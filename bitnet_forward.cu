#define __STRICT_ANSI__
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
 * bitnet_forward.cu  —  v7: W2A8 quantized activations + __dp4a integer math
 *
 * Matrix layout
 * -----------------------------------
 *   A         : (M, K)          int8   — quantized input activations
 *   B_packed  : (K/64, N, 16)   int8   — ternary weights, transposed for
 *                                        coalesced access
 *   C         : (M, N)          int32  — output (de-quantized to fp16 in Python)
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
 *       BLOCK_K raised 16 → 64 to match the natural uint4 width:
 *       16 bytes × 4 weights/byte = 64 ternary weights per tile slice.
 *
 *   v4: Transposed B layout + coalesced B load.
 *       v3's formula B_u4[g_col * K_u4 + t] strides consecutive threads by
 *       K_u4 uint4 elements (= K/4 bytes) in memory — the GPU sees N
 *       independent cache lines per warp, giving ~0.53% bandwidth utilisation.
 *
 *       Fix: Python pre-packs B into physical layout (K_u4, N, 16 bytes) so
 *       that tile t is the outer dimension and the neuron index n is inner.
 *       The kernel now reads B_u4[t * N + g_col]: consecutive threads
 *       (consecutive g_col / tx values) hit consecutive uint4 addresses
 *       → one 128-byte cache line serves all 8 threads in a warp sub-group
 *       → 100% coalesced LDG, theoretical 12.5× bandwidth gain over v3.
 *
 *   v5: Shared Memory Unpacking Station — eliminates redundant decode work.
 *       In v4, every one of the 16 ty-rows in a block decoded the same uint4
 *       bucket for its output column: 16× redundant bitwise shifts/masks per
 *       tile step, making the kernel compute-bound on bitwise instructions
 *       during prefill (M = 1024, many warps active).
 *
 *       Fix: only ty == 0 (16 threads) decodes the uint4.  It writes 64 fp16
 *       scalars {-1, 0, +1} into s_B_unpacked[tx][0..63] (the "Unpacking
 *       Station").  After __syncthreads(), all 16 ty-rows read the decoded
 *       weights via cheap LDS and execute a branchless 64-iteration FMA loop.
 *       Decode work: 16× reduction.  Compute path: scalar FP16 MACs, no
 *       branches, fully pipelined by the hardware scheduler.
 *
 *   v7: W2A8 + __dp4a — integer-only inner loop.
 *       Activations arrive as int8 (quantized per-token in Python).
 *       Phase 2 uses __dp4a to compute 4 int8×int8 multiply-adds per
 *       instruction, accumulating into int32.  The inner loop iterates
 *       k in steps of 4 (16 iterations for BLOCK_K=64).
 *       De-quantization (int32 → fp16 via gamma/scale) happens in Python.
 *
 * Grid / Block dimensions — UNCHANGED from v2
 * --------------------------------------------
 *   block = dim3(BLOCK, BLOCK) = dim3(16, 16) = 256 threads / block
 *   grid  = dim3((N+15)/16, (M+15)/16)
 *
 * Shared memory per block
 * -----------------------
 *   s_A           : 16 × 64 × 1 B  (int8 activation tile)          = 1024 B
 *   s_B_unpacked  : 64 × 16 × 1 B  (int8 decoded weight tile)     = 1024 B
 *   total                                                          = 2048 B  ← well under 48 KB / SM
 *
 * Theoretical occupancy (GA107 / RTX 3050, Ampere sm_86)
 * -------------------------------------------------------
 *   Shared-memory limit  : 48 KB / 2048 B = 24 blocks max
 *   Thread limit         : 2048  / 256    =  8 blocks max  ← binding (unchanged)
 *   → 8 blocks × 256 = 2048 threads = 100 % theoretical occupancy
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
    const int8_t *__restrict__ A,        // (M, K)    row-major int8
    const int8_t *__restrict__ B_packed, // (K/64, N, 16)  pre-packed ternary (v4 transpose)
    int32_t *__restrict__ C,             // (M, N)    row-major int32 output
    int M, int K, int N) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      printf("KERNEL IS RUNNING\n");
  }
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
  //   s_A          : int8 activation tile   — BLOCK rows  × BLOCK_K cols
  //   s_B_unpacked : int8 decoded weight tile — BLOCK_K rows × BLOCK cols
  // ------------------------------------------------------------------
  __shared__ int8_t  s_A[BLOCK][BLOCK_K];           // 1024 bytes — activation tile
  __shared__ int32_t s_B_packed[BLOCK_K / 4][BLOCK]; // 1024 bytes — packed decoded weights (4 x int8 in int32)

  int acc = 0;

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
          (row < M && k_a < K) ? __ldg(&A[row * K + k_a]) : (int8_t)0;
    }

    // --- Fetch + Unpack: global B → s_B_unpacked (Shared Memory Unpacking Station) ---
    //
    // Only ty == 0 (16 threads, one per output column in this block) participates.
    // Each thread:
    //   1. Issues ONE coalesced __ldg for its output column's uint4 (unchanged
    //      from v4: B_packed layout is (K_u4, N) uint4, so consecutive tx values
    //      map to consecutive addresses → fully coalesced 256-byte transaction).
    //   2. Unpacks the 4 × uint32 chunks into 64 fp16 scalars {-1, 0, +1} and
    //      writes them to s_B_unpacked[0..63][tx].
    //
    // The 240 ty > 0 threads are idle here, but the latency is hidden by the
    // s_A load above which all 256 threads issued through the __ldg cache.
    //
    // After __syncthreads() every ty row reads the decoded weights via cheap LDS
    // in Phase 2 — decode cost drops from 16× (v4) to 1× per tile step.
    if (ty == 0) {
      const int g_col = blockIdx.x * BLOCK + tx;
      const uint4 bw = (g_col < N) ? __ldg(&B_u4[t * N + g_col])
                                   : make_uint4(0u, 0u, 0u, 0u);

      // Unpack 4 chunks × 16 codes = 64 ternary weights into s_B_unpacked.
      // After #pragma unroll, c is a compile-time constant (0/1/2/3), so the
      // ternary chain collapses to one of bw.x/y/z/w with zero runtime cost.
      // The inner index (c * 16 + i) is also compile-time-constant → nvcc emits
      // STS with a literal byte offset (no runtime address arithmetic).
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t chunk = (c == 0)   ? bw.x
                               : (c == 1) ? bw.y
                               : (c == 2) ? bw.z
                                          : bw.w;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          int32_t packed_val = 0;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const unsigned code = (chunk >> ((g * 4 + i) * 2)) & 0x3u;
            // Encoding: 0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → -1 (undefined).
            int8_t w = (code == 1u) ? (int8_t) 1
                     : (code == 2u) ? (int8_t)-1
                                    : (int8_t) 0;
            packed_val |= ((int32_t)w & 0xFF) << (i * 8);
          }
          s_B_packed[c * 4 + g][tx] = packed_val;
        }
      }
    }

    // All threads must finish loading before any thread reads shared memory.
    __syncthreads();

    // ==================================================================
    // Phase 2 — Compute: __dp4a integer dot-product accumulation
    // ==================================================================
    //
    // s_A[ty][k..k+3]          : 4 contiguous int8 activations
    // s_B_unpacked[k..k+3][tx] : 4 strided int8 weights {-1, 0, +1}
    //
    // __dp4a(a_val, b_val, acc) computes:
    //   acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    // where a and b are reinterpreted as 4 × int8 packed in int32.
    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K / 4; ++k) {
        int a_val = reinterpret_cast<const int*>(s_A[ty])[k];
        int b_val = s_B_packed[k][tx];
        acc = __dp4a(a_val, b_val, acc);
      }
    }

    // Guard before the next tile's load overwrites shared memory.
    __syncthreads();

  } // end for (t)

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Host-side C++ wrapper
// ---------------------------------------------------------------------------
torch::Tensor bitnet_forward(torch::Tensor A,        // (M, K)        int8, CUDA
                             torch::Tensor B_packed, // (K/64, N, 16) int8, CUDA (pre-packed)
                             int M, int K, int N) {
  TORCH_CHECK(A.is_cuda(), "A must be on a CUDA device");
  TORCH_CHECK(B_packed.is_cuda(), "B_packed must be on a CUDA device");
  TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
  TORCH_CHECK(B_packed.dtype() == torch::kInt8, "B_packed must be int8");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");

  TORCH_CHECK(K % 64 == 0,
              "Kernel requires K divisible by 64 "
              "(one uint4 = 16 bytes = 64 packed 2-bit weights). "
              "Got K=",
              K, ". Pad your weight matrix to the next multiple of 64.");

  auto C = torch::zeros(
      {M, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

  const dim3 block(BLOCK, BLOCK);
  const dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

  bitnet_forward_kernel<<<grid, block>>>(
      A.data_ptr<int8_t>(),
      B_packed.data_ptr<int8_t>(),
      C.data_ptr<int32_t>(), M, K, N);

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
        "BitNet-1.58b W2A8 forward pass (v7, __dp4a).\n"
        "Args: A (M,K int8), B_packed (K/64,N,16 int8 pre-packed), M, K, N -> C (M,N int32)\n"
        "Activations are quantized to int8 in Python; output is int32, de-quantized in Python.\n"
        "Requires: K % 64 == 0.");
}
