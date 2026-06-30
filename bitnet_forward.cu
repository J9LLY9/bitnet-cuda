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
 * bitnet_forward.cu  —  v8: Register-tiled GEMM (BM=64, BN=64, BK=64, TM=4, TN=4)
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
 * v8 changes from v7
 * ------------------
 *   v7: 16×16 output tile per block, 1 output element per thread.
 *       Shared-memory bandwidth bound at large batch sizes: each thread
 *       issues 16 shared-memory reads per K-group, giving only 0.5 MACs
 *       per byte read from shared memory.
 *
 *   v8: 64×64 output tile per block, 4×4 output elements per thread.
 *       Each thread accumulates 16 values in registers across the outer
 *       product of a[4] × b[4].  Arithmetic intensity at the shared-memory
 *       level rises to 8 MACs per shared-memory load (4× over v7).
 *
 *       Thread mapping (BM=64, BN=64, block=16×16):
 *         Thread (tx, ty) computes C rows {ty, ty+16, ty+32, ty+48}
 *                                    cols {tx, tx+16, tx+32, tx+48}
 *         relative to the block's 64×64 tile origin.
 *
 *       Cooperative loads:
 *         s_A[2][64][64]  — all 256 threads, one uint4 each (fully coalesced).
 *         s_B_packed[2][16][64] — first 64 threads only (one uint4 = 64 weights).
 *
 *       Bank-conflict analysis:
 *         s_A reads: all threads in a warp share ty → same row → broadcast,
 *                    zero bank conflicts.
 *         s_B reads: s_B_packed[b][k][tx + 16*j] — tx=0..15 maps to banks 0..15
 *                    (one int32 per bank), no conflict within a warp.
 *
 *   v9 adds double-buffered (ping-pong) shared memory tiling:
 *     Tile 0 is prefetched before the loop.  Each iteration loads tile t+1
 *     into the write buffer while computing from the read buffer, then issues
 *     a single __syncthreads() at the end of the loop body.
 *
 * Grid / Block dimensions
 * -----------------------
 *   block = dim3(16, 16) = 256 threads / block
 *   grid  = dim3((N+63)/64, (M+63)/64)
 *
 * Shared memory per block
 * -----------------------
 *   s_A[2]        : 2 × 64 × 64 × 1 B  (double-buffered activation tile)  = 8192 B
 *   s_B_packed[2] : 2 × 16 × 64 × 4 B  (double-buffered weight tile)      = 8192 B
 *   total                                                                  = 16384 B (16 KB)
 *
 * Theoretical occupancy (GA107 / RTX 3050, Ampere sm_86)
 * -------------------------------------------------------
 *   Shared-memory limit  : 48 KB / 16 KB =  3 blocks max
 *   Thread limit         : 2048  / 256   =  8 blocks max
 *   → 3 blocks × 256 = 768 threads = 37.5% theoretical occupancy
 *     The occupancy drop vs v8 is offset by hiding global-memory latency
 *     through overlap of tile prefetch with register-tile computation.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4) void bitnet_forward_kernel(
    const int8_t *__restrict__ A,        // (M, K)    row-major int8
    const int8_t *__restrict__ B_packed, // (K/64, N, 16)  pre-packed ternary
    int32_t *__restrict__ C,             // (M, N)    row-major int32 output
    int M, int K, int N) {

  const int ty  = threadIdx.y;
  const int tx  = threadIdx.x;
  const int tid = ty * 16 + tx; // 0..255

  // Double-buffered shared memory tiles (ping-pong buffers).
  alignas(16) __shared__ int8_t  s_A[2][64][64];        // activation tile
  alignas(16) __shared__ int32_t s_B_packed[2][16][64]; // packed ternary weight tile

  // Per-thread 4×4 accumulator in registers.
  // acc[i][j] accumulates the dot product for output element
  //   C[blockIdx.y*64 + ty + 16*i][blockIdx.x*64 + tx + 16*j].
  int acc[4][4] = {0};

  const uint4 *__restrict__ B_u4 = reinterpret_cast<const uint4 *>(B_packed);
  const int num_tiles = K >> 6; // K / 64

  // Thread roles for cooperative loads:
  //   s_A:        each of the 256 threads loads one uint4 (16 int8 values).
  //   s_B_packed: first 64 threads each load one uint4 and decode 64 weights.
  const int load_row    = tid / 4; // row in s_A this thread fills:  0..63
  const int load_col_u4 = tid % 4; // uint4 column offset:           0..3

  // ----------------------------------------------------------------
  // Prefetch tile 0 into buffer 0 before the main loop begins.
  // ----------------------------------------------------------------
  {
    const int g_row    = blockIdx.y * 64 + load_row;
    const int g_col_u4 = 0 * 4 + load_col_u4;
    uint4 val_A = (g_row < M)
        ? __ldg(reinterpret_cast<const uint4 *>(A + g_row * K) + g_col_u4)
        : make_uint4(0u, 0u, 0u, 0u);
    reinterpret_cast<uint4 *>(s_A[0])[tid] = val_A;
  }

  if (tid < 64) {
    const int col_b = blockIdx.x * 64 + tid;
    const uint4 bw  = (col_b < N)
        ? __ldg(&B_u4[0 * N + col_b])
        : make_uint4(0u, 0u, 0u, 0u);

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
          int8_t w = (code == 1u) ? (int8_t) 1
                   : (code == 2u) ? (int8_t)-1
                                  : (int8_t) 0;
          packed_val |= ((int32_t)w & 0xFF) << (i * 8);
        }
        s_B_packed[0][c * 4 + g][tid] = packed_val;
      }
    }
  }
  __syncthreads(); // Wait for prefetch tile 0 to land.

  for (int t = 0; t < num_tiles; ++t) {

    const int read_idx  = t % 2;
    const int write_idx = (t + 1) % 2;

    // ----------------------------------------------------------------
    // Prefetch tile t+1 into write_idx while compute runs on read_idx.
    // ----------------------------------------------------------------
    if (t < num_tiles - 1) {
      {
        const int g_row    = blockIdx.y * 64 + load_row;
        const int g_col_u4 = (t + 1) * 4 + load_col_u4;
        uint4 val_A = (g_row < M)
            ? __ldg(reinterpret_cast<const uint4 *>(A + g_row * K) + g_col_u4)
            : make_uint4(0u, 0u, 0u, 0u);
        reinterpret_cast<uint4 *>(s_A[write_idx])[tid] = val_A;
      }

      if (tid < 64) {
        const int col_b = blockIdx.x * 64 + tid;
        const uint4 bw  = (col_b < N)
            ? __ldg(&B_u4[(t + 1) * N + col_b])
            : make_uint4(0u, 0u, 0u, 0u);

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
              // 0b00 → 0, 0b01 → +1, 0b10 → -1
              int8_t w = (code == 1u) ? (int8_t) 1
                       : (code == 2u) ? (int8_t)-1
                                      : (int8_t) 0;
              packed_val |= ((int32_t)w & 0xFF) << (i * 8);
            }
            s_B_packed[write_idx][c * 4 + g][tid] = packed_val;
          }
        }
      }
    }

    // ----------------------------------------------------------------
    // Compute outer product from read_idx buffer.
    //
    // For each K-group k (covering 4 consecutive K-elements):
    //   a[i] = 4 int8 activations from row (ty + 16*i) of the tile
    //   b[j] = 4 int8 weights   for col (tx + 16*j) of the tile
    //   acc[i][j] += __dp4a(a[i], b[j])
    // ----------------------------------------------------------------
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = reinterpret_cast<const int *>(s_A[read_idx][ty + 16 * i])[k];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        b[j] = s_B_packed[read_idx][k][tx + 16 * j];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
      }
    }

    // Single sync: ensures prefetch into write_idx has landed (safe for next
    // iteration's compute) and compute from read_idx is done (safe to overwrite).
    __syncthreads();

  } // end for (t)

  // ----------------------------------------------------------------
  // Write 4×4 register tile to global output C
  // ----------------------------------------------------------------
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int global_row = blockIdx.y * 64 + ty + 16 * i;
    if (global_row < M) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int global_col = blockIdx.x * 64 + tx + 16 * j;
        if (global_col < N) {
          C[global_row * N + global_col] = acc[i][j];
        }
      }
    }
  }
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

  const dim3 block(16, 16);
  const dim3 grid((N + 63) / 64, (M + 63) / 64);

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
        "BitNet-1.58b W2A8 forward pass (v9, double-buffered register-tiled, __dp4a).\n"
        "Args: A (M,K int8), B_packed (K/64,N,16 int8 pre-packed), M, K, N -> C (M,N int32)\n"
        "Activations are quantized to int8 in Python; output is int32, de-quantized in Python.\n"
        "Block tile: BM=64, BN=64, BK=64. Thread tile: TM=4, TN=4.\n"
        "Requires: K % 64 == 0.");
}
