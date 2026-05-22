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
 * bitnet_forward.cu  —  v3: Vectorized uint4 Memory Access
 *
 * Matrix layout (unchanged from v2)
 * -----------------------------------
 *   A         : (M, K)   fp16   — input activations (RMSNorm'd)
 *   B_packed  : (N, K/4) int8   — ternary weights, 4 per byte, 2 bits each
 *   C         : (M, N)   fp16   — output
 *
 * Ternary encoding (2 bits per weight, LSB-first within each byte)
 * -----------------------------------------------------------------
 *   0b00 →  0    0b01 → +1    0b10 → -1
 *   bits [2b+1 : 2b] hold weight at K-offset b within each packed byte.
 *
 * What changed from v2 → v3
 * --------------------------
 *   v2: scalar uint8_t loads for B_packed — 1 byte per thread.
 *       Each 1-byte load triggers a 128-byte cache-line transaction, so
 *       only ~0.78% of fetched bandwidth was useful weight data.
 *
 *   v3: B_packed cast to const uint4*; each participating thread issues a
 *       single 16-byte __ldg load (1 uint4 = 64 2-bit ternary weights).
 *       Per cache-line transaction:  16 B useful  /  128 B fetched ≈ 12.5%.
 *
 *   BLOCK_K raised 16 → 64 to match the natural uint4 width:
 *       16 bytes × 4 weights/byte = 64 ternary weights per tile slice.
 *
 * Grid / Block dimensions — UNCHANGED from v2
 * --------------------------------------------
 *   block = dim3(BLOCK, BLOCK) = dim3(16, 16) = 256 threads / block
 *   grid  = dim3((N+15)/16, (M+15)/16)
 *
 * Shared memory per block
 * -----------------------
 *   s_A     : 16 × 64 × 2 B (fp16 tile)  = 2048 B
 *   s_B_u4  : 16 × 16 B   (uint4 tile)   =  256 B
 *   total                                 = 2304 B  ← well under 48 KB / SM
 *
 * Theoretical occupancy (GA107 / RTX 3050, Ampere sm_86)
 * -------------------------------------------------------
 *   Shared-memory limit  : 48 KB / 2304 B ≈ 21 blocks max
 *   Thread limit         : 2048  / 256     =  8 blocks max  ← binding
 *   → 8 blocks × 256 = 2048 threads = 100 % theoretical occupancy (same as v2)
 *
 * uint4 alignment
 * ---------------
 *   uint4 loads require 16-byte alignment.
 *   PyTorch CUDA allocations are ≥ 256-byte aligned, so B_packed.data_ptr()
 *   is always safe to reinterpret as const uint4*.
 *
 * K divisibility requirement (UPGRADED from K % 4 → K % 64)
 * ----------------------------------------------------------
 *   One uint4 covers 16 packed bytes = 64 2-bit weights.
 *   K must therefore be a multiple of 64.
 *   Standard BitNet dimensions (256, 512, 1024, 2048, 4096 …) all satisfy this.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// BLOCK_K = 64: one uint4 of B (16 bytes) covers exactly 64 ternary weights.
// Changing this constant changes the tile geometry — the uint4 load logic
// below is written specifically for BLOCK_K == 64.
#define BLOCK 16
#define BLOCK_K 64

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// __launch_bounds__(maxThreads, minBlocks):
//   maxThreads = 256  — hard cap; compiler can allocate registers freely up to
//   this. minBlocks  =   4  — conservative hint: guarantee ≥4 concurrent CTAs /
//   SM.
//                       Lowered from v2's 8 to avoid ptxas "out of range" on
//                       sm_86 when register pressure from the expanded
//                       BLOCK_K=64 tile prevents achieving 8 × 256 = 2048
//                       threads / SM simultaneously.
__global__ __launch_bounds__(BLOCK *BLOCK, 4) void bitnet_forward_kernel(
    const __half *__restrict__ A,        // (M, K)    row-major fp16
    const int8_t *__restrict__ B_packed, // (N, K/4)  row-major, packed ternary
    __half *__restrict__ C,              // (M, N)    row-major fp16 output
    int M, int K, int N) {
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * BLOCK + ty; // global output row  [0, M)
  const int col = blockIdx.x * BLOCK + tx; // global output col  [0, N)

  // K_u4: number of uint4 elements per row of B_packed.
  //   B_packed row width = K/4 bytes.
  //   K/4 bytes  /  16 bytes per uint4  =  K/64.
  const int K_u4 = K >> 6; // K / 64

  // ------------------------------------------------------------------
  // Shared memory
  //   s_A     : fp16 activation tile  — BLOCK rows × BLOCK_K cols
  //   s_B_u4  : ternary weight tile   — one uint4 (64 weights) per
  //             output column in this block (BLOCK entries total)
  // ------------------------------------------------------------------
  __shared__ __half s_A[BLOCK][BLOCK_K]; // 2048 bytes
  __shared__ uint4 s_B_u4[BLOCK];        //  256 bytes

  float acc = 0.0f;

  // Cast once, outside the loop — zero runtime cost, pure type annotation.
  // __restrict__ goes on the variable, not the cast's type (nvcc #191-D).
  // Aliasing is safe: we never write through B_u4 and the original int8_t
  // pointer is __restrict__, so the compiler knows there is no conflict.
  const uint4 *__restrict__ B_u4 = reinterpret_cast<const uint4 *>(B_packed);

  // Since K % 64 == 0 is enforced by the host, num_tiles == K / BLOCK_K
  // exactly.
  const int num_tiles = K_u4; // same value: K / 64

  for (int t = 0; t < num_tiles; ++t) {

// ==================================================================
// Phase 1 — Cooperative load: global memory → shared memory
// ==================================================================

// --- Load s_A (activation tile) ---
//
// With BLOCK_K = 64 and 256 threads (BLOCK × BLOCK), each thread
// is responsible for 64/16 = 4 fp16 elements in its own row of s_A.
//
// Thread (ty, tx) writes:
//   s_A[ty][ 0*16 + tx ]  ←  A[row][t*64 +  0 + tx]
//   s_A[ty][ 1*16 + tx ]  ←  A[row][t*64 + 16 + tx]
//   s_A[ty][ 2*16 + tx ]  ←  A[row][t*64 + 32 + tx]
//   s_A[ty][ 3*16 + tx ]  ←  A[row][t*64 + 48 + tx]
//
// Within a warp, consecutive tx → consecutive K addresses
// → fully coalesced reads from global memory (LDG.U16 × 4).
#pragma unroll
    for (int i = 0; i < BLOCK_K / BLOCK; ++i) { // 4 iterations
      const int k_a = t * BLOCK_K + i * BLOCK + tx;
      s_A[ty][i * BLOCK + tx] =
          (row < M && k_a < K) ? __ldg(&A[row * K + k_a]) : __float2half(0.0f);
    }

    // --- Load s_B_u4 (ternary weight tile) — vectorized uint4 ---
    //
    // We need exactly BLOCK (16) uint4 values: one per output column in
    // this block.  Only the BLOCK threads with ty == 0 participate;
    // the remaining 240 threads are idle during this phase (the B tile
    // is tiny — 256 bytes — so the latency is hidden by the A load above,
    // which all 256 threads issued in parallel through the __ldg cache).
    //
    // Thread (ty=0, tx):
    //   g_col = blockIdx.x * 16 + tx        — global output column
    //   loads 16 contiguous bytes from row g_col of B_packed, at the
    //   uint4 offset for this tile (t), covering 64 ternary weights:
    //     B_packed[ g_col * (K/4) + t*16 ]  …  [ + 15 ]
    //
    // The 16 threads access 16 different rows of B_packed — these are
    // strided by K/4 bytes in global memory.  Each individual uint4 load
    // still brings in 16 bytes of *useful* data per 128-byte cache line
    // (12.5% efficiency), vs. 1 byte per cache line (0.78%) in v2.
    if (ty == 0) {
      const int g_col = blockIdx.x * BLOCK + tx; // global output col
      s_B_u4[tx] = (g_col < N) ? __ldg(&B_u4[g_col * K_u4 + t])
                               : make_uint4(0u, 0u, 0u, 0u);
    }

    // All threads must finish loading before any thread reads shared memory.
    __syncthreads();

    // ==================================================================
    // Phase 2 — Compute: dot product from shared memory
    // ==================================================================
    //
    // Only live output threads accumulate; idle threads keep acc == 0
    // and the store below is skipped for them.
    if (row < M && col < N) {

      // Fetch this output column's packed weights from shared memory.
      // s_B_u4[tx] is broadcast to all ty-rows in the block (same tx).
      // uint4 layout:
      //   .x  →  weights for K-offsets  0 .. 15  (bits  0..31)
      //   .y  →  weights for K-offsets 16 .. 31  (bits  0..31)
      //   .z  →  weights for K-offsets 32 .. 47  (bits  0..31)
      //   .w  →  weights for K-offsets 48 .. 63  (bits  0..31)
      const uint4 bw = s_B_u4[tx];

      // The four 32-bit chunks are named explicitly to ensure nvcc keeps
      // them in registers and does not spill a runtime-indexed array.
      const uint32_t chunk0 =
          bw.x; // K-offsets  0..15  (2 bits each, LSB-first)
      const uint32_t chunk1 = bw.y; // K-offsets 16..31
      const uint32_t chunk2 = bw.z; // K-offsets 32..47
      const uint32_t chunk3 = bw.w; // K-offsets 48..63

// Fully unrolled decode-and-accumulate: 4 chunks × 16 weights = 64 MACs.
//
// KEY OPTIMISATION — compile-time-constant s_A index:
//   After #pragma unroll expands both loops, c and i are literal
//   integers.  The expression (c * 16 + i) therefore folds to a
//   single constant at compile time, and nvcc emits an LDS with a
//   constant byte offset — zero runtime arithmetic on the s_A address.
//
// Branch strategy — skip zero and undefined weights:
//   code 0 (0b00) → weight  0  → no contribution
//   code 1 (0b01) → weight +1  → add activation
//   code 2 (0b10) → weight -1  → subtract activation
//   code 3 (0b11) → undefined  → treated as -1  (matches v2 behaviour;
//                                real BitNet-1.58b weights never use this)
//   Branching on (code != 0) skips the ~25% zero weights in real models,
//   reducing total FMAs.  With --use_fast_math the branch overhead is
//   negligible when the branch is mostly coherent within a warp.
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        // After unrolling, c is a compile-time constant (0/1/2/3) so this
        // ternary chain reduces to one of chunk0/1/2/3 with no runtime cost.
        const uint32_t chunk = (c == 0)   ? chunk0
                               : (c == 1) ? chunk1
                               : (c == 2) ? chunk2
                                          : chunk3;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const unsigned code = (chunk >> (i * 2)) & 0x3u;
          if (code != 0u) {
            // (c * 16 + i) is a compile-time constant — constant LDS offset.
            const float a = __half2float(s_A[ty][c * 16 + i]);
            acc += (code == 1u) ? a : -a;
          }
        }
      }
    } // end if (row < M && col < N)

    // Guard before the next tile's load overwrites shared memory.
    __syncthreads();

  } // end for (t)

  if (row < M && col < N)
    C[row * N + col] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Host-side C++ wrapper
// ---------------------------------------------------------------------------
torch::Tensor bitnet_forward(torch::Tensor A,        // (M, K) float16, CUDA
                             torch::Tensor B_packed, // (N, K/4) int8,  CUDA
                             int M, int K, int N) {
  TORCH_CHECK(A.is_cuda(), "A must be on a CUDA device");
  TORCH_CHECK(B_packed.is_cuda(), "B_packed must be on a CUDA device");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_packed.dtype() == torch::kInt8, "B_packed must be int8");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");

  // v3 requires K % 64 == 0:
  //   uint4 = 16 bytes = 64 ternary weights.
  //   Standard BitNet dims (256, 512, 1024, 2048, 4096 …) all satisfy this.
  TORCH_CHECK(K % 64 == 0,
              "v3 vectorized kernel requires K divisible by 64 "
              "(one uint4 = 16 bytes = 64 packed 2-bit weights). "
              "Got K=",
              K, ". Pad your weight matrix to the next multiple of 64.");

  auto C = torch::zeros(
      {M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

  // Grid / block dimensions unchanged from v2.
  const dim3 block(BLOCK, BLOCK); // 16×16 = 256 threads
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
        "BitNet-1.58b uint4-vectorized ternary-weight forward pass (v3).\n"
        "Args: A (M,K fp16), B_packed (N,K/4 int8), M, K, N -> C (M,N fp16)\n"
        "Requires: K % 64 == 0  (one uint4 = 16 bytes = 64 2-bit weights).");
}
