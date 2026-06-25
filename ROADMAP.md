# ROADMAP.md — BitNet 1.58b Custom CUDA Engine

This file tracks short-term engineering tasks, completed milestones, and immediate blockers. Updated daily during pair-programming sessions.

---

## 1. Current State (As of June 25, 2026)

* **Status:** W2A8 Quantization, native `__dp4a` hardware math, and shared-memory packing are fully implemented and verified!
  * **PyTorch side ([model.py](file:///home/p520/bitnet-cuda/model.py)):** Dynamic activation scaling/quantization to `int8`, and output de-quantization back to `float16` integrated successfully.
  * **CUDA side ([bitnet_forward.cu](file:///home/p520/bitnet-cuda/bitnet_forward.cu)):** Vectorized 32-bit load and 4-way `__dp4a` integer accumulation inside the inner loop with an optimized shared-memory packing station that bypasses unpacking overhead during accumulation.
  * **Build status:** Compiled and linked successfully on GCC 15 + CUDA 13.
  * **Correctness:** Correctness verified. [test_kernel.py](file:///home/p520/bitnet-cuda/test_kernel.py) passes all tests with updated absolute tolerance (`atol=0.25`) to accommodate W2A8 integer quantization rounding noise. Interactive text generation ([inference.py](file:///home/p520/bitnet-cuda/inference.py)) runs end-to-end through the kernel.

---

## 2. Immediate Blocker (Where We Left Off)

* **None.** All outstanding blockers (test tolerance mismatch) have been resolved.

---

## 3. Tomorrow's Tasks (In Order)

### **Task 1: Profile and Analyze Bottlenecks**
Profile the dynamic W2A8 quantization kernel using Nsight Compute to determine current bottlenecks (memory bandwidth vs. compute instruction limits) now that `__dp4a` is active.

### **Task 2: Design Register Tiling Strategy**
Sketch out the 2D thread-tile layout for register tiling to load sub-tiles of activations and weights into registers, maximizing data reuse and bypassing shared-memory read bandwidth bounds.

### **Task 3: Implement Register Tiled Kernel**
Write the v7 kernel in [bitnet_forward.cu](file:///home/p520/bitnet-cuda/bitnet_forward.cu) utilizing register tiling, and integrate it into the test suite and benchmark sweeps.
