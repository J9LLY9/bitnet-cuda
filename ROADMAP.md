# ROADMAP.md — BitNet 1.58b Custom CUDA Engine

This file tracks short-term engineering tasks, completed milestones, and immediate blockers. Updated daily during pair-programming sessions.

---

## 1. Current State (As of June 26, 2026)

* **Status:** v9 warp-shuffle decoded weight kernel is fully implemented, verified, and benchmarked.
  * **v9 Performance:** **2.14× faster than fp32 cuBLAS** at M=K=N=4096, **12.34 TOPS** on RTX 3050. Largest gains at small batch sizes: M=1 improved from 1.31× (v8) to **1.37×**, M=8 from 1.47× to **1.55×**.
  * **Design:** All 256 threads now participate in weight decode via `__shfl_sync`. 8 warps × 8 columns per warp; lanes 0..7 load, 4 `__shfl_sync` calls broadcast all 4 `uint4` components; each lane decodes one (chunk, column) pair. Eliminates the 192-thread idle stall of v8's `tid < 64` decode path.
  * **Why 4 shuffles:** A single `__shfl_sync(mask, loaded_val, src_lane)` only distributes the source lane's `loaded_val`, which is always `bw.x` (chunk 0) since src_lane ∈ 0..7 forces chunk_idx=0. Four component-wise shuffles are required to broadcast all chunks correctly.
  * **Build & Correctness:** All 4 unit tests pass on GCC 15 + CUDA 13 (RTX 3050 / sm_86).

* **Benchmark sweep (K=N=4096):**

  | M (batch) | fp16 cuBLAS (ms) | fp32 cuBLAS (ms) | Custom v9 (ms) | Speedup vs fp32 |
  |----------:|----------------:|-----------------:|---------------:|----------------:|
  |         1 |           0.160 |            0.317 |          0.232 |          1.37×  |
  |         8 |           0.162 |            0.361 |          0.233 |          1.55×  |
  |        32 |           0.172 |            0.362 |          0.235 |          1.54×  |
  |       128 |           0.284 |            0.788 |          0.423 |          1.86×  |
  |       512 |           0.958 |            2.975 |          1.511 |          1.97×  |
  |      2048 |           3.589 |           12.106 |          5.673 |          2.13×  |
  |      4096 |           7.319 |           23.546 |         11.061 |          2.13×  |

---

## 2. Immediate Blocker (Where We Left Off)

* **None.** v9 is complete. Next step is double-buffered shared memory to pipeline tile loads with compute.

---

## 3. Next Tasks (In Order)

### **Task 1: Double-Buffered Shared Memory**
Allocate two ping-pong copies of `s_A` and `s_B_packed` (8 KB → 16 KB per block). While Phase 2 compute runs on the current buffer, asynchronously prefetch the next tile into the alternate buffer. This eliminates the load→compute `__syncthreads()` barrier, replacing it with a lighter async-copy completion fence. Expected tradeoff: 3 blocks per SM (down from 6) vs. fewer stalls.

### **Task 2: Profile v9 vs v8**
Run `ncu` on both kernels to confirm the decode idle-time reduction is reflected in SM utilization metrics, and to identify whether the remaining `__syncthreads()` barriers or the compute loop is the next bottleneck.

