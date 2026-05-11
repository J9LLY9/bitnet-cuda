#ifndef __CUDACC__
// This block tricks the IDE's eyes so it stops showing red lines.
// The real compiler (nvcc) will ignore everything inside this block.
#define __global__
#define __shared__
#define __half float
#define __launch_bounds__(...)
#define __restrict__
#include <cuda_runtime.h>
typedef float half;
#endif
/*
 * bitnet_forward.cu  —  v2: Shared-Memory Tiled Kernel
 *
 * Matrix layout (unchanged from v1)
 * -----------------------------------
 *   A         : (M, K)   fp16   — input activations (RMSNorm'd)
 *   B_packed  : (N, K/4) int8   — ternary weights, 4 per byte, 2 bits each
 *   C         : (M, N)   fp16   — output
 *
 * Ternary encoding (2 bits per weight)
 * --------------------------------------
 *   0b00 →  0    0b01 → +1    0b10 → -1
 *   bits [2b+1 : 2b] hold weight at K-offset b within each packed byte.
 *
 * What changed from v1
 * ---------------------
 *   v1 had every thread stream all K values independently from global memory.
 *   v2 breaks K into BLOCK_K-wide tiles. The whole thread-block cooperates to
 *   stage each tile into __shared__ before any thread does arithmetic on it.
 *   This converts repeated long-latency global loads into cheap shared-mem
 * reads.
 *
 * RTX 3050 (GA107 / Ampere) target tuning
 * -----------------------------------------
 *   BLOCK    = 16  →  256 threads/block  =  8 warps
 *   BLOCK_K  = 16  →  16 K-elements / tile  (= 4 packed bytes of B per output
 * col)
 *
 *   Shared memory per block
 *     s_A        : 16 × 16 × 2 B (fp16) =  512 B
 *     s_B_packed : 16 × 4  × 1 B (int8) =   64 B
 *     total                              =  576 B   ← well under the 48 KB/SM
 * budget
 *
 *   Theoretical occupancy
 *     48 KB shmem limit → 48 KB / 576 B = 83 blocks max (shmem-limited)
 *     2048 thread limit → 2048 / 256    =  8 blocks max (thread-limited)
 *     → 8 blocks × 256 = 2048 threads = 100 % theoretical occupancy on GA107
 *
 *   Read path
 *     __ldg() routes global loads through the 128 KB read-only L1 cache,
 *     which is distinct from the coherent L1 on Ampere. Helpful when the same
 *     A-row is reused across N tiles and the same B column across M tiles.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define BLOCK 16
#define BLOCK_K 16 // K-tile width; gives BK4 = 4 packed bytes of B per tile row

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// __launch_bounds__ tells the compiler we expect 256 threads and up to 8
// concurrent blocks per SM — lets it tune register allocation for occupancy.
__global__ __launch_bounds__(BLOCK *BLOCK, 8) void bitnet_forward_kernel(
    const __half *__restrict__ A,        // (M, K)    row-major
    const int8_t *__restrict__ B_packed, // (N, K/4)  row-major, packed ternary
    __half *__restrict__ C,              // (M, N)    row-major output
    int M, int K, int N) {
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * BLOCK + ty; // global output row   [0, M)
  const int col = blockIdx.x * BLOCK + tx; // global output col   [0, N)
  const int K4 = K >> 2;                   // K / 4

  // ------------------------------------------------------------------
  // Shared memory shelves (see sizing calculation in the file header)
  // ------------------------------------------------------------------
  __shared__ __half s_A[BLOCK][BLOCK_K];            // input activation tile
  __shared__ int8_t s_B_packed[BLOCK][BLOCK_K / 4]; // packed weight tile

  float acc = 0.0f;

  const int BK4 = BLOCK_K / 4; // = 4 packed bytes per tile-row
  const int num_tiles = (K + BLOCK_K - 1) / BLOCK_K;

  for (int t = 0; t < num_tiles; ++t) {

    // ==================================================================
    // Phase 1 — Cooperative load: global memory → shared memory
    // ==================================================================

    // --- Load s_A ---
    // Thread (ty, tx) is responsible for s_A[ty][tx].
    // All 256 threads participate; out-of-bounds slots get 0 so the
    // inner compute loop can run without per-element bounds checks.
    {
      const int k_a = t * BLOCK_K + tx; // global K index for this thread
      s_A[ty][tx] =
          (row < M && k_a < K) ? __ldg(&A[row * K + k_a]) : __float2half(0.0f);
    }

    // --- Load s_B_packed ---
    // We have BLOCK × BK4 = 16 × 4 = 64 bytes to fill.
    // Assign each byte to a unique thread via a flat index [0, 64).
    // Threads 64..255 sit idle here — the load is tiny vs. compute.
    {
      const int flat = ty * BLOCK + tx; // 0 … 255
      if (flat < BLOCK * BK4) {
        const int b_row = flat / BK4; // which output col in this block
        const int b_col = flat % BK4; // which packed byte in the tile
        const int g_col = blockIdx.x * BLOCK + b_row; // global output col
        const int g_kg = t * BK4 + b_col; // global packed-byte index
        s_B_packed[b_row][b_col] = (g_col < N && g_kg < K4)
                                       ? __ldg(&B_packed[g_col * K4 + g_kg])
                                       : (int8_t)0;
      }
    }

    // All threads must finish loading before anyone reads shared memory.
    __syncthreads();

    // ==================================================================
    // Phase 2 — Compute: dot product from shared memory
    // ==================================================================
    // Only live output threads accumulate (inactive threads already have
    // acc == 0 and we skip the store below).
    if (row < M && col < N) {
// BLOCK_K is a compile-time constant → compiler fully unrolls.
#pragma unroll
      for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
        const int kg_local = k_local >> 2; // which packed byte  [0, BK4)
        const int b = k_local & 3;         // bit-pair index      [0, 4)
        const uint8_t packed = (uint8_t)s_B_packed[tx][kg_local];
        const int code = (packed >> (b * 2)) & 0x3;

        // Skip zero weights — they contribute nothing and avoid a multiply.
        if (code != 0) {
          const float w = (code == 1) ? 1.0f : -1.0f;
          acc += __half2float(s_A[ty][k_local]) * w;
        }
      }
    }

    // Guard before next tile's load overwrites shared memory.
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Host-side C++ wrapper
// ---------------------------------------------------------------------------
torch::Tensor bitnet_forward(torch::Tensor A,        // (M, K) float16,  CUDA
                             torch::Tensor B_packed, // (N, K/4) int8,   CUDA
                             int M, int K, int N) {
  TORCH_CHECK(A.is_cuda(), "A must be on a CUDA device");
  TORCH_CHECK(B_packed.is_cuda(), "B_packed must be on a CUDA device");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_packed.dtype() == torch::kInt8, "B_packed must be int8");
  TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 (2-bit ternary packing)");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");

  auto C = torch::zeros(
      {M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

  const dim3 block(BLOCK, BLOCK);
  const dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

  bitnet_forward_kernel<<<grid, block>>>(
      reinterpret_cast<const __half *>(A.data_ptr<at::Half>()),
      B_packed.data_ptr<int8_t>(),
      reinterpret_cast<__half *>(C.data_ptr<at::Half>()), M, K, N);

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "bitnet_forward_kernel failed: ", cudaGetErrorString(err));

  return C;
}

// ---------------------------------------------------------------------------
// pybind11 module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bitnet_forward", &bitnet_forward,
        "BitNet-1.58 tiled ternary-weight forward pass (shared-memory v2).\n"
        "Args: A (M,K fp16), B_packed (N,K/4 int8), M, K, N -> C (M,N fp16)");
}
