# ROADMAP.md — BitNet 1.58b Custom CUDA Engine

This file tracks short-term engineering tasks, completed milestones, and immediate blockers. Updated daily during pair-programming sessions.

---

## 1. Current State (As of June 26, 2026)

* **Status:** v8 register-tiled kernel is fully implemented, verified, and benchmarked.
  * **v8 Performance:** The kernel now runs **2.12× faster than fp32 cuBLAS** at M=K=N=4096, achieving **12.32 TOPS** on the RTX 3050. This is a dramatic turnaround from v7, which was 6.55× *slower* than fp32 cuBLAS.
  * **Design:** BM=64, BN=64, BK=64, TM=4, TN=4. Each of 256 threads computes a 4×4 register sub-tile. Arithmetic intensity at the shared-memory level is 8 MACs per load (4× over v7). `alignas(16)` tiles, `uint4` cooperative loads, 100% bank-conflict-free reads.
  * **Build & Correctness:** All 4 unit tests pass on GCC 15 + CUDA 13 (RTX 3050 / sm_86).

* **Benchmark sweep (K=N=4096):**

  | M (batch) | fp16 cuBLAS (ms) | fp32 cuBLAS (ms) | Custom v8 (ms) | Speedup vs fp32 |
  |----------:|----------------:|-----------------:|---------------:|----------------:|
  |         1 |           0.160 |            0.317 |          0.242 |          1.31×  |
  |         8 |           0.162 |            0.361 |          0.246 |          1.47×  |
  |        32 |           0.172 |            0.362 |          0.244 |          1.48×  |
  |       128 |           0.283 |            0.787 |          0.437 |          1.80×  |
  |       512 |           0.954 |            2.941 |          1.493 |          1.97×  |
  |      2048 |           3.568 |           12.042 |          5.672 |          2.12×  |
  |      4096 |           7.285 |           23.718 |         11.199 |          2.12×  |

---

## 2. Immediate Blocker (Where We Left Off)

* **None.** v8 is complete. Next step is warp shuffle intrinsics to reduce `__syncthreads()` overhead.

---

## 3. Next Tasks (In Order)

### **Task 1: Warp Shuffle Intrinsics**
Replace the two `__syncthreads()` barriers per tile step with `__shfl_sync()` for intra-warp communication. Focus on the weight-decode phase where only 64 of 256 threads write to `s_B_packed` — this is where barrier overhead is highest relative to work done.

### **Task 2: Profile v8 vs v7**
Run `ncu` (Nsight Compute) on both kernels to confirm the arithmetic intensity improvement is reflected in SM utilization, and to identify the next bottleneck for v9.

