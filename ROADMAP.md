# ROADMAP.md — BitNet 1.58b Custom CUDA Engine

This file tracks short-term engineering tasks, completed milestones, and immediate blockers. Updated daily during pair-programming sessions.

---

## 1. Current State (As of June 30, 2026)

* **Status:** Double-buffered shared memory tiling is fully implemented and integrated. All unit correctness tests pass.
* **Performance:** Hiding global memory latency via double-buffered prefetching yields a performance improvement across all batch sizes, despite the reduction in block occupancy (from 8 blocks per SM down to 6).
* **Build & Correctness:** All 4 unit tests pass on GCC 15 + CUDA 13 (RTX 3050 / sm_86).

* **Benchmark sweep (K=N=4096):**

  | M (batch) | fp16 cuBLAS (ms) | fp32 cuBLAS (ms) | Custom v8 (ms) | Custom DB (ms) | Speedup vs fp32 |
  |----------:|----------------:|-----------------:|---------------:|---------------:|----------------:|
  |         1 |           0.160 |            0.317 |          0.232 |          0.212 |          1.49×  |
  |         8 |           0.162 |            0.361 |          0.233 |          0.213 |          1.70×  |
  |        32 |           0.172 |            0.362 |          0.235 |          0.217 |          1.67×  |
  |       128 |           0.284 |            0.788 |          0.423 |          0.393 |          2.01×  |
  |       512 |           0.958 |            2.975 |          1.511 |          1.406 |          2.11×  |
  |      2048 |           3.589 |           12.106 |          5.673 |          5.351 |          2.25×  |
  |      4096 |           7.319 |           23.546 |         11.061 |         10.463 |          2.26×  |

---

## 2. Completed Milestones
* **Milestone 1:** Double-Buffered Shared Memory Tiling (Integrated on June 30, 2026). Yields a ~5.4% to 8.6% latency reduction compared to the single-buffered v8 kernel.

---

## 3. Next Tasks (In Order)

### **Task 1: Profile Double-Buffered Kernel vs Baseline**
Run `ncu` on both kernels to verify that the stall cycles on memory dependency (`LDG`) have decreased, and to determine the next bottleneck (whether arithmetic pipelines or register pressure is now limiting further speedups).

### **Task 2: Warp-Shuffle Decode Re-Integration**
Now that double-buffering is verified, re-integrate the v9 warp-shuffle decoding optimization to see if we can combine both benefits for a further latency reduction at $M=1$.
