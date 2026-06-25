# ROADMAP.md — BitNet 1.58b Custom CUDA Engine

This file tracks short-term engineering tasks, completed milestones, and immediate blockers. Updated daily during pair-programming sessions.

---

## 1. Current State (As of June 25, 2026)

* **Status:** W2A8 Quantization and native `__dp4a` hardware math have been implemented!
  * **PyTorch side (`model.py`):** Added dynamic activation scaling/quantization to `int8`, and output de-quantization back to `float16`.
  * **CUDA side (`bitnet_forward.cu`):** Replaced the branchless `FSEL` float GEMM loop with a vectorized 32-bit load and 4-way `__dp4a` integer accumulation, resulting in a pure integer pipeline.
  * **Build status:** Compiled and linked successfully on GCC 15 + CUDA 13.
  * **Correctness:** Verified that KV caching outputs match full sequence outputs 100% identically. The custom integer GEMM matches PyTorch float reference within ~1% average error.

---

## 2. Immediate Blocker (Where We Left Off)

* **Tolerance mismatch in `test_kernel.py`:**
  * `TEST 2` (Random) and `TEST 3` (Boundary) failed because their tolerances are set to `atol=0.1` (designed for the unquantized float16 kernel).
  * Because W2A8 quantization introduces a tiny amount of rounding noise (expected ~1% error), the max error was `0.164`.
  * **Solution:** We need to increase `atol` to `0.25` in those tests.

---

## 3. Tomorrow's Tasks (In Order)

### **Task 1: Adjust tolerances in test_kernel.py**
Run the following prompt in Claude Code to apply the fix:
```text
Please edit test_kernel.py to adjust the absolute tolerances (atol) in test_random and test_boundary to 0.25 to accommodate the expected W2A8 integer quantization noise.
```

### **Task 2: Verify and Benchmark**
Run the test suite to verify correctness and see the performance speedup from `__dp4a`:
```bash
venv/bin/python test_kernel.py
```

### **Task 3: Run Interactive Text Generation**
Verify that interactive text generation runs correctly using the W2A8 integer pipeline:
```bash
venv/bin/python inference.py
```
