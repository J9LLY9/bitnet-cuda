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
 * bitnet_forward.cu  —  v9: Warp-shuffle decoded weight distribution
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
 * v9 changes from v8
 * ------------------
 *   v8: Weight decode (Phase 1b) used only the first 64 threads (tid < 64).
 *       The remaining 192 threads were idle during this phase, creating a
 *       synchronization stall before Phase 2 could begin.
 *
 *   v9: All 256 threads participate in weight decode via warp-shuffle.
 *       8 warps × 8 columns per warp = 64 columns (covers the full BN=64 tile).
 *       Within each warp:
 *         - Lanes 0..7 each issue one coalesced LDG for their column's uint4.
 *         - Four __shfl_sync calls (one per uint32 component) broadcast each
 *           chunk to the 24 non-loading lanes that need it.
 *         - Each of the 32 lanes decodes one (chunk, column) pair and writes
 *           4 packed int32 values to s_B_packed.
 *       The decode work (previously 64 threads × 16 writes = 1024 writes)
 *       is now spread across 256 threads × 4 writes = 1024 writes with
 *       all threads active, hiding the __syncthreads() barrier latency better.
 *
 *       Why 4 shuffles, not 1:
 *         A single __shfl_sync(mask, loaded_val, src_lane) reads loaded_val
 *         from lane src_lane. Since src_lane is always 0..7 (chunk_idx=0),
 *         a single shuffle only distributes chunk 0 (bw.x). Four separate
 *         shuffles — one each for bw.x, bw.y, bw.z, bw.w — are required to
 *         correctly broadcast all four chunks to the 32 lanes of each warp.
 *
 * Grid / Block dimensions — unchanged from v8
 * -------------------------------------------
 *   block = dim3(16, 16) = 256 threads / block
 *   grid  = dim3((N+63)/64, (M+63)/64)
 *
 * Shared memory per block — unchanged from v8
 * -------------------------------------------
 *   s_A          : 64 × 64 × 1 B  (int8 activation tile)       = 4096 B
 *   s_B_packed   : 16 × 64 × 4 B  (int32 packed weight tile)   = 4096 B
 *   total                                                       = 8192 B  (8 KB)
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

  // Warp decomposition for the weight decode phase.
  const int lane_id = tid % 32; // 0..31 within warp
  const int warp_id = tid / 32; // 0..7  (8 warps per block)

  // Shared memory tiles — 16-byte aligned for safe uint4 reinterpret casts.
  alignas(16) __shared__ int8_t  s_A[64][64];        // activation tile
  alignas(16) __shared__ int32_t s_B_packed[16][64]; // packed ternary weight tile

  // Per-thread 4×4 accumulator in registers.
  // acc[i][j] accumulates the dot product for output element
  //   C[blockIdx.y*64 + ty + 16*i][blockIdx.x*64 + tx + 16*j].
  int acc[4][4] = {0};

  const uint4 *__restrict__ B_u4 = reinterpret_cast<const uint4 *>(B_packed);
  const int num_tiles = K >> 6; // K / 64

  // Thread roles for the s_A cooperative load (unchanged from v8):
  //   Each of the 256 threads loads one uint4 (16 int8 values).
  const int load_row    = tid / 4; // s_A row this thread fills: 0..63
  const int load_col_u4 = tid % 4; // uint4 column offset:       0..3

  for (int t = 0; t < num_tiles; ++t) {

    // ----------------------------------------------------------------
    // Phase 1a — Load activation tile s_A (all 256 threads)
    //
    // Thread tid loads s_A[load_row][load_col_u4*16 .. +15].
    // Flattened reinterpret: reinterpret_cast<uint4*>(s_A)[tid] maps to
    // byte offset tid*16, which equals row=tid/4, uint4-col=tid%4. ✓
    // Consecutive tid values → consecutive g_col_u4 → coalesced LDG.
    // ----------------------------------------------------------------
    {
      const int g_row    = blockIdx.y * 64 + load_row;
      const int g_col_u4 = t * 4 + load_col_u4;
      uint4 val_A = (g_row < M)
          ? __ldg(reinterpret_cast<const uint4 *>(A + g_row * K) + g_col_u4)
          : make_uint4(0u, 0u, 0u, 0u);
      reinterpret_cast<uint4 *>(s_A)[tid] = val_A;
    }

    // ----------------------------------------------------------------
    // Phase 1b — Load and unpack weight tile s_B_packed (all 256 threads)
    //
    // Each warp (warp_id 0..7) is responsible for 8 weight columns:
    //   columns [warp_id*8 .. warp_id*8+7] of the BN=64 tile.
    //
    // Lanes 0..7 of each warp issue one LDG each for their column.
    // Lanes 8..31 hold bw={0,0,0,0} and receive data via shuffle.
    //
    // Shuffle distribution: each lane l is assigned
    //   chunk_idx = l / 8  (which uint32 chunk of the uint4: 0=.x, 1=.y, 2=.z, 3=.w)
    //   src_lane  = l % 8  (which loading lane holds the data for this column)
    //
    // A single __shfl_sync(mask, loaded_val, src_lane) cannot distribute all
    // four chunks because loaded_val for any lane in 0..7 always reflects
    // chunk_idx=0 (i.e., bw.x only). Four component-wise shuffles are used:
    //   cx = shfl(bw.x, src_lane) — chunk 0 from the source column
    //   cy = shfl(bw.y, src_lane) — chunk 1 from the source column
    //   cz = shfl(bw.z, src_lane) — chunk 2 from the source column
    //   cw = shfl(bw.w, src_lane) — chunk 3 from the source column
    // Each lane then selects the right cx/cy/cz/cw for its chunk_idx.
    //
    // LDG coalescing: lanes 0..7 of each warp access consecutive addresses
    // → fully coalesced (lanes 0..7 of warp 0 fetch cols 0..7, etc.).
    //
    // All 256 threads are active here vs. 64 in v8 → 4× more decode
    // parallelism and better latency hiding before __syncthreads().
    // ----------------------------------------------------------------
    {
      // Load: only lanes 0..7 issue LDG; others hold zero.
      uint4 bw = make_uint4(0u, 0u, 0u, 0u);
      if (lane_id < 8) {
        const int col_b = blockIdx.x * 64 + warp_id * 8 + lane_id;
        bw = (col_b < N) ? __ldg(&B_u4[t * N + col_b])
                         : make_uint4(0u, 0u, 0u, 0u);
      }

      // Each lane's role: decode chunk `chunk_idx` of column `src_lane`.
      const int chunk_idx = lane_id / 8; // 0..3
      const int src_lane  = lane_id % 8; // 0..7

      // Broadcast all four uint32 components from src_lane to current lane.
      const uint32_t cx = __shfl_sync(0xFFFFFFFF, bw.x, src_lane);
      const uint32_t cy = __shfl_sync(0xFFFFFFFF, bw.y, src_lane);
      const uint32_t cz = __shfl_sync(0xFFFFFFFF, bw.z, src_lane);
      const uint32_t cw = __shfl_sync(0xFFFFFFFF, bw.w, src_lane);

      // Select the chunk matching this lane's assigned chunk_idx.
      const uint32_t chunk = (chunk_idx == 0) ? cx
                           : (chunk_idx == 1) ? cy
                           : (chunk_idx == 2) ? cz
                                              : cw;

      // Decode 16 ternary weights from chunk into 4 packed int32 values
      // and write them to the correct column of s_B_packed.
      const int col_local = warp_id * 8 + src_lane; // 0..63
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
        s_B_packed[chunk_idx * 4 + g][col_local] = packed_val;
      }
    }

    __syncthreads();

    // ----------------------------------------------------------------
    // Phase 2 — Compute outer product accumulation in registers
    //
    // For each K-group k (covering 4 consecutive K-elements):
    //   a[i] = 4 int8 activations from row (ty + 16*i) of the tile
    //   b[j] = 4 int8 weights   for col (tx + 16*j) of the tile
    //   acc[i][j] += __dp4a(a[i], b[j])
    //
    // s_A reads: all threads in a warp share ty → same row → broadcast.
    // s_B reads: tx=0..15 → banks 0..15 (one int32 per bank) → no conflict.
    // ----------------------------------------------------------------
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = reinterpret_cast<const int *>(s_A[ty + 16 * i])[k];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        b[j] = s_B_packed[k][tx + 16 * j];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
      }
    }

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
        "BitNet-1.58b W2A8 forward pass (v9, warp-shuffle decode, __dp4a).\n"
        "Args: A (M,K int8), B_packed (K/64,N,16 int8 pre-packed), M, K, N -> C (M,N int32)\n"
        "Activations are quantized to int8 in Python; output is int32, de-quantized in Python.\n"
        "Block tile: BM=64, BN=64, BK=64. Thread tile: TM=4, TN=4.\n"
        "Weight decode: all 256 threads active via 4x __shfl_sync per warp.\n"
        "Requires: K % 64 == 0.");
}
