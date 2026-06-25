# spec.md — BitNet 1.58b Custom CUDA Engine

## 1. Vision

This project is a from-scratch CUDA C++ inference engine for ternary (1.58-bit) large language models based on the BitNet b1.58 architecture (Ma et al., 2024). It replaces floating-point matrix multiplication with pure addition and subtraction by exploiting the constraint that every weight is exactly {-1, 0, +1}.

The project exists to answer a focused engineering question: **how fast can a single-GPU ternary inference engine run when every layer of the stack — packing, memory layout, kernel design, and pipeline integration — is purpose-built for 2-bit weights?**

It serves as the central portfolio piece for an Autumn 2026 transfer application to UW and MIT, demonstrating applied GPU systems engineering, low-level CUDA optimization, and quantized AI inference — built entirely from scratch on consumer hardware (RTX 3050, 8 GB VRAM).

## 2. Goals

- **End-to-end ternary inference:** Accept a text prompt and generate tokens using the custom CUDA kernel for all weight-quantized matrix multiplications — no cuBLAS fallback on the critical path.
- **Measurable speedup:** Demonstrate wall-clock and memory-bandwidth advantages over equivalent PyTorch FP16 inference on the same hardware, validated by reproducible benchmarks.
- **Production-quality kernel engineering:** Vectorized memory coalescing, shared-memory tiling, branchless ternary accumulation, bank-conflict-free layouts — each optimization motivated by profiling data.
- **Portfolio clarity:** Every design decision is documented, benchmarked, and explainable in an interview setting.

## 3. Success Criteria

The project is complete when:

1. `app.py` / `inference.py` generate coherent text using the custom CUDA kernel for all `BitLinear` layers — no `torch.nn.Linear` on the inference path.
2. Kernel correctness is verified by a test suite covering hand-crafted values, random matrices, boundary shapes, and edge cases (all-zero weights, non-tile-aligned dimensions).
3. A reproducible benchmark compares custom-kernel inference against PyTorch FP16 baseline on identical inputs, reporting latency, throughput (tok/s), memory traffic, and bandwidth utilization.
4. Generation quality (coherent TinyStories-level output) is preserved after kernel integration.

## 4. Current State

### Implemented

- **Model:** `BitNet158` transformer — 12 layers, 512 embed dim, 8 heads, RMSNorm + SubLN + GELU, ternary `BitLinear` with STE. Trained to loss 1.04 on TinyStories.
- **Kernel (v6):** Tiled shared-memory GEMM in `bitnet_forward.cu`. 2-bit weight packing (4 weights per byte), `uint4` vectorized loads, `[K, N]` pre-transposed layout for coalesced access, shared-memory unpacking station, branchless `ISETP+FSEL` accumulation (zero FMUL). Bank-conflict fixes applied (transposed `s_B_unpacked`, padded `s_A`).
- **Inference pipeline:** `inference.py` (CLI) and `app.py` (Gradio UI) with top-p sampling, repetition penalty, temperature control, streaming generation, and GPU telemetry.
- **Benchmark harness:** `benchmark_uw.py` and sweep in `test_kernel.py` comparing custom kernel vs cuBLAS FP16/FP32.
- **Weights:** `BitNet_UW_Final_Gold_1.04.safetensors` — trained independently on a ThinkStation P520.
- **Kernel Integration:** Custom CUDA kernel successfully integrated into `BitLinear.forward()` (ISSUE-07) and verified matching mathematical correctness across all shapes.
- **KV Cache:** `past_key_values` caching fully implemented (ISSUE-08) for linear-cost $O(T)$ autoregressive generation, dropping step-cost to $O(1)$ and verified mathematically equivalent to full sequence generation.

### In Progress

- `__dp4a` integer dot-product accumulation (W2A8 quantization) (ISSUE-09).

### Not Yet Implemented

- Warp-level shuffle intrinsics to replace `__syncthreads()` barriers.
- Register tiling (2D thread-tile accumulation in registers).
- Full unpack-phase thread utilization (currently 16 of 256 threads active during unpack).

## 5. Current Priority

**`__dp4a` integer dot-product accumulation (W2A8 quantization)**

Migrate the kernel from float16 activations to 8-bit integer activations (INT8), enabling the use of native GPU `__dp4a` instructions (4-way integer dot-product with 32-bit accumulation). This will replace the float calculations in the inner loop with high-throughput integer arithmetic, breaking through the floating-point instruction throughput limit. This requires adding scaling/quantization layers for activations prior to GEMM.

## 6. Non-Goals

- **Training framework.** Training code exists for producing weights but is not an optimization target.
- **Multi-GPU / distributed inference.** Single-GPU, single-model scope only.
- **General-purpose GEMM library.** The kernel is purpose-built for ternary weights; it does not aim to replace cuBLAS for arbitrary matrix shapes or dtypes.
- **Model architecture research.** The transformer architecture is fixed; this project optimizes the *execution* of that architecture, not its design.
- **Deployment infrastructure.** No serving framework, containerization, or API layer.
- **Side-projects.** Scope is locked to the kernel and inference pipeline through Summer 2026.

## 7. Engineering Principles

1. **Measure before optimizing.** Every kernel change is preceded by profiling (nsight, CUDA events) and followed by a benchmark to confirm the hypothesis.
2. **Profile before rewriting.** Identify the actual bottleneck (memory, compute, latency) before writing new code.
3. **Prefer reproducible benchmarks.** Fixed seeds, deterministic inputs, warmup iterations, CUDA event timing — no `time.time()` on the host.
4. **Optimize for learning and explainability.** Every optimization must be understood well enough to explain in an interview. Clever tricks that can't be articulated are rejected.
5. **Correctness before performance.** The test suite must pass before any optimization is merged.
6. **Minimize abstraction.** Prefer explicit CUDA C over wrapper libraries. The learning value is in the low-level details.

## 8. Long-Term Roadmap

1. **Fix test suite** — correct dimension mismatches and add the memory transpose to `pack_ternary`.
2. **Kernel integration** — wire `bitnet_cuda.bitnet_forward` into `BitLinear.forward()` for end-to-end inference.
3. **KV cache** — implement `past_key_values` for linear-cost autoregressive generation.
4. **`__dp4a` migration** — move to W2A8 quantization with native integer dot-product instructions.
5. **Register tiling** — 2D thread-tile accumulation in registers to break the shared-memory bandwidth ceiling.
6. **Warp shuffle intrinsics** — replace `__syncthreads()` with `__shfl_sync()` for intra-warp communication.

## 9. Project Organization

| Path | Purpose |
|------|---------|
| `README.md` | Public-facing project summary, benchmark highlights, and setup instructions. Optimized for GitHub visitors and recruiters. |
| `spec.md` | Long-term engineering direction, goals, constraints, and roadmap. The canonical reference for both human contributors and AI coding assistants. Stable — not a scratchpad. |
| `ROADMAP.md` | *(planned)* Detailed, ordered task breakdown derived from this spec. Updated as milestones are completed. |
| `benchmarks/` | *(planned)* Standalone benchmark scripts and saved results, separated from test code. |
| `docs/` | *(planned)* Deep-dive writeups on specific optimizations (bank conflicts, vectorized loads, etc.) for portfolio and interview prep. |
| `tests/` | *(planned)* Dedicated test directory. Currently `test_kernel.py` lives in the project root. |
