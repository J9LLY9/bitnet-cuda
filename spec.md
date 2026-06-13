# Architecture & Context Specification: BitNet 1.58b CUDA Engine

## System Architecture Baseline
- **Environment:** Native Linux (Ubuntu 26.04 LTS) | Lenovo ThinkStation P520
- **Hardware Profile:** 8GB VRAM (RTX 3050) | Target footprint requires optimization for bare-metal execution
- **Local Model Engine:** Ollama running custom `qwen-coder-8k` (Context Window: 8192 tokens)

## Project Objective
A from-scratch, highly optimized CUDA C++ forward-pass inference engine for 1.58B Parameter Ternary LLMs, eliminating heavy floating-point multiplication in favor of pure addition/subtraction optimizations.

## Tech Stack & Project Footprint
- **Language Standard:** Modern C++, CUDA C, Python (PyBind11 for C++ extensions)
- **Core Frameworks:** PyTorch (for training simulation validation), CUDA 12.x
- **Key Mathematical Constraint:** 2-bit Weight Quantization packing 4 ternary weights (-1, 0, 1) into a single uint8_t byte.

## Strategic Constraints & Guardrails
- **Scope Lock:** Absolutely NO secondary features or side-projects outside the core client architecture this summer.
- **Academic Overlay:** This project serves as a key pillar for the Autumn 2026 academic reset at Bellevue College and elite university transfer portfolios (UW, MIT). Every design decision must show professional-grade system design.

---

## Gemini 2.5 Flash — Teaching Role

Gemini 2.5 Flash operates exclusively as a **teacher and concept explainer** in this project. It runs inside a terminal CLI.

### What Gemini does
- Explains GPU architecture concepts, CUDA programming model, and paper math in plain language
- Answers the "why" behind a fix so the engineer understands before any code is written
- Walks through CUDA docs, research papers (e.g. BitNet b1.58), and hardware specs on demand
- Asks Socratic follow-up questions to confirm understanding

### What Gemini never does
- **Never writes, edits, or suggests code changes** — that is Claude Code's job
- **Never modifies any file** in the project
- **Never proposes implementation details** — it teaches the concept; Claude Code owns the design and execution

### How to use it (Step 2 of the workflow)
When an issue is documented by Claude Code and handed off with a "concept to learn" note, bring that concept to Gemini in the terminal. Ask it to explain until the idea is fully clear. Then return to Claude Code and deliver the brief (Step 3).

Gemini is the tutor between Step 1 and Step 3. Nothing more.

---

## Collaboration Workflow (6-Step Process)

This is the standard operating procedure for every engineering issue tackled in this project. Both Claude Code and Gemini CLI follow this protocol.

| Step | Who  | Action |
|------|------|--------|
| 1    | Claude | **Find & document the issue** — identify the root cause, affected file/line, and why it is wrong |
| 2    | You  | **Learn outside the terminal** — research the concept (GPU architecture, CUDA docs, papers, YouTube, etc.) until you understand the "why" behind the fix |
| 3    | You  | **Return and brief Claude** — describe what you learned; Claude confirms understanding before writing any code |
| 4    | Claude | **Explain the proposed fix** — describe what will be built, why each design decision was made, and what the expected outcome is |
| 5    | Claude | **Implement the fix** — code is written only after step 4 is agreed upon |
| 6    | Both | **Update spec.md** — mark the issue resolved, record what changed and what the new baseline is |

---

## Launching the Inference Server

### Prerequisites
- Activate the venv with `. venv_p520_native/bin/activate` (dot-space, not `./` — the script lacks the execute bit)
- Trained weights file `BitNet_UW_Final_Gold_1.04.safetensors` must be in the project root

### Gradio Web UI (primary)
```bash
. venv_p520_native/bin/activate
python app.py
# Opens at http://127.0.0.1:7860
```

### CLI inference (quick test)
```bash
. venv_p520_native/bin/activate
python inference.py
```

### Weight loading priority (auto-detected)
1. `bitnet_sft.pt` + `sft_tokenizer/` (SFT fine-tune)
2. `bitnet_weights.pt` (base PyTorch)
3. `BitNet_UW_Final_Gold_1.04.safetensors` ← current primary weights
4. `BITNET_1.05_HERO_WEIGHTS.safetensors`
5. `bitnet_weights_final.safetensors`
6. Most recent `checkpoint_step*.safetensors`

### Inference Status — Confirmed Working (2026-06-09)
- `app.py` and `inference.py` import `BitNet158` from `model.py` (stale inline `BitNetLanguageModel` removed)
- Weights load cleanly from `BitNet_UW_Final_Gold_1.04.safetensors`
- Confirmed throughput: **22.9 tok/s** on RTX 3050 8GB (pure PyTorch — custom kernel not yet wired in)
- Model architecture: `BitNet158`, 12 layers, 512 embed dim, 8 heads, `max_seq_len=256`, RMSNorm + SubLN + GELU, no biases

---

## Core Workspace State

### Completed Features (MVP Stable)
1. **Vectorized Memory Coalescing:** Upgraded scalar byte-fetches to 128-bit vectorized loads (`uint4`), driving memory bus saturation via a single `LDG.E.128` instruction per thread.
2. **Matrix Pre-packing Layout:** Transposed raw row-major layouts `[N, K]` to `[K, N]` physical memory alignment prior to execution, forcing coalesced access lines.
3. **Shared Memory Unpacking Station:** Implemented a `__shared__` memory array where a single row of threads unpacks `uint4` data, casting to `__half` floats, reducing redundant bitwise shifts via block synchronization.

### Active Engineering Bottlenecks & Code Gaps
- **Prefill Latency (M=1024) Collapse:** Latency spikes wildly to 34.093 ms under Matrix-Matrix operations due to shared memory bank conflicts and a lack of register-level accumulation tiling.
- ~~**FP16 Floating-Point Relapse**~~ **Resolved (ISSUE-02):** FMUL eliminated from inner loop; weights now stored as `int8_t` and accumulated via branchless ISETP+FSEL.
- **Warp Synchronization Barriers:** Heavy reliance on global `__syncthreads()` forces explicit hardware pipeline stalls within the streaming multiprocessors.

### Summer Execution Blueprint
1. **The __dp4a Shift:** Move to W2A8 (2-bit weight, 8-bit activation) quantization, utilizing native CUDA `__dp4a` instructions to compute four packed 8-bit integer dot products in a single clock cycle.
2. **2D Register Tiling Implementation:** Restructure the kernel so individual threads handle an 8x8 sub-tile accumulation entirely inside high-speed hardware registers, lifting the shared memory bandwidth wall.
3. **Warp-Level Intrinsics:** Replace the block-sync barriers with warp shuffle instructions (`__shfl_sync()`) to broadcast unpacked bytes across threads directly through registers.

---

## Open Issue Tracker

Issues are listed in recommended fix order. Each issue follows the 6-step workflow above.
`[ ]` = open  `[x]` = resolved

---

### Kernel Issues (`bitnet_forward.cu`)

#### `[x]` ISSUE-01 — 16-way Shared Memory Bank Conflict (Phase 2 reads)
- **File:** `bitnet_forward.cu:144`
- **Root cause:** `s_B_unpacked[BLOCK][BLOCK_K]` is declared with rows of 64 `__half` = 128 bytes. Ampere has 32 banks × 4 bytes = 128 bytes per full rotation. Threads with different `tx` values (stride = 1 row = 128 bytes) all map to the same bank. Every Phase 2 read (`s_B_unpacked[tx][i]` with 16 different `tx` values active) triggers a 16-way bank conflict — 16× latency penalty on every shared memory load in the hot path.
- **Concept to learn:** CUDA shared memory bank conflict mechanics; how bank index is computed (`byteOffset / 4) % 32`); row-vs-column stride access patterns.
- **Proposed fix:** Declare as `s_B_unpacked[BLOCK_K][BLOCK]` (transposed) so Phase 2 reads `s_B_unpacked[i][tx]` — consecutive `tx` values hit consecutive banks with zero conflicts.
- **Expected gain:** Up to 16× reduction in Phase 2 shared memory latency; directly addresses the prefill spike.
- **Resolution (2026-06-11):** Transposed `s_B_unpacked` to `[BLOCK_K][BLOCK]` to eliminate 16-way bank conflicts during both writes and reads. Padded `s_A` to `[BLOCK][BLOCK_K + 2]` to eliminate 2-way bank conflicts during reads. Verified correctness of execution.

---

#### `[x]` ISSUE-02 — FP16 Multiply Relapse in Accumulation Loop
- **File:** `bitnet_forward.cu:249`
- **Root cause:** `acc += __half2float(s_A[ty][i]) * __half2float(s_B_unpacked[tx][i])` — weights are decoded into `{-1.0, 0.0, +1.0}` as `__half` then multiplied as `float`. This re-introduces the floating-point multiply that BitNet b1.58 is specifically designed to eliminate. A ternary weight can only be +1, -1, or 0; no multiply is needed.
- **Concept to learn:** BitNet b1.58 paper (Ma et al., 2024) §3 — the theoretical basis for replacing FMA with conditional add/subtract. Read about branchless integer accumulation techniques.
- **Proposed fix (path A — simple):** In Phase 2, read weight as `int8` and use a branchless `acc += (w == 1) ? a : (w == -1) ? -a : 0.0f`. No multiply. \
  **Proposed fix (path B — blueprint):** The `__dp4a` shift (ISSUE-05) — store activations as int8 (W2A8) and use `__dp4a` for four dot products per clock with zero float multiplies.
- **Expected gain:** Removes all float multiplies from the inner loop; aligns implementation with the published BitNet thesis.
- **Resolution (2026-06-11):** Retyped `s_B_unpacked` from `__half[BLOCK_K][BLOCK]` to `int8_t[BLOCK_K][BLOCK]` (1024 B, down from 2048 B). Unpack writes now store `int8_t {-1, 0, +1}`. Phase 2 inner loop changed from `acc += half2float(A) * half2float(B)` to `acc += (w==1)?a:(w==-1)?-a:0.0f` — compiler emits ISETP+FSEL, zero FMUL. Kernel bumped to v6. Verified correctness. Known trade-off: int8 storage introduces 4-way bank conflicts on s_B_unpacked reads (vs 2-way with __half); acceptable until ISSUE-05 replaces this layout with __dp4a.

---

#### `[ ]` ISSUE-03 — 240 Idle Threads per Tile (Unpack Phase)
- **File:** `bitnet_forward.cu:199`
- **Root cause:** Only `ty == 0` (16 of 256 threads) performs the uint4 unpack. The remaining 240 threads (ty 1–15) sit completely idle during every tile's unpack phase. This is 94% warp occupancy waste during the most memory-intensive step.
- **Concept to learn:** CUDA warp shuffle intrinsics (`__shfl_sync`, `__shfl_xor_sync`) — how threads can broadcast register values to other lanes in a warp without touching shared memory or needing all threads to do the same work.
- **Proposed fix:** Distribute the unpack work across all warps using warp shuffles to broadcast decoded bytes, eliminating the `if (ty == 0)` dead zone.
- **Expected gain:** Eliminates the 15/16 idle-thread penalty; directly addresses "Warp Synchronization Barriers" from the Summer Blueprint.

---

### Test Suite Issues (`test_kernel.py`)

#### `[ ]` ISSUE-04 — `test_fixed` passes wrong tensor dimensions to kernel
- **File:** `test_kernel.py:99`
- **Root cause:** `A_vals` is created as shape `(1, 4)` but `K=64` is passed to the kernel. The kernel computes `A[row * 64 + k_a]` for `k_a` up to 63 — 60 GPU memory reads are fully out of bounds. `B_packed` is 2 bytes; the kernel expects at least 32 bytes.
- **Fix:** Pad `A_vals` to `(1, 64)` with zeros (only positions 0–3 carry the actual test values). Rebuild `W_int` to `(2, 64)` with zeros filling the unused weights, so the hand-computed expected output `[1.0, 3.0]` still holds.

---

#### `[ ]` ISSUE-05 — `test_boundary` and `test_all_zero_weights` use K not divisible by 64
- **File:** `test_kernel.py:165` (`K=48`) and `test_kernel.py:195` (`K=16`)
- **Root cause:** Both tests use `K` values that fail `TORCH_CHECK(K % 64 == 0)`. They crash before testing anything. `test_boundary` (the most important test for tile edge handling) has never actually run.
- **Fix:** Change `test_boundary` to `K=128` (two full tiles with M/N non-multiples of 16). Change `test_all_zero_weights` to `K=64`.

---

#### `[ ]` ISSUE-06 — `pack_ternary` never applies the v4 memory transpose
- **File:** `test_kernel.py:25–47`
- **Root cause:** `pack_ternary` returns `(N, K//4)` byte layout. The kernel (since v4) requires the pre-transposed layout `(K//64, N, 16)`. `benchmark_uw.py:50–53` applies `.view(N, K//64, 16).permute(1,0,2).contiguous()` correctly; `test_kernel.py` does not. For `K=64` (1 tile), the layouts are accidentally identical, masking the bug. For `K=128+` the kernel reads wrong weight data and silently produces incorrect output.
- **Fix:** Add the transpose step inside `pack_ternary` after packing, or apply it at every call site inside the test file.

---

### Architecture / Pipeline Gaps

#### `[ ]` ISSUE-07 — Custom CUDA kernel not connected to the inference pipeline
- **Files:** `inference.py`, `app.py`, `model.py`
- **Root cause:** `bitnet_cuda.bitnet_forward` is only called from benchmarks and tests. `BitNet158` uses standard `torch.nn.Linear` for all matrix multiplications. The 22.9 tok/s figure is pure PyTorch/cuBLAS — the optimized kernel contributes nothing to actual inference.
- **Fix:** Replace `torch.nn.Linear` forward calls in `model.py` with the custom kernel for weight-quantized layers. Requires solving ISSUE-01 and ISSUE-02 first so the kernel is actually correct.
- **Note:** This is the integration milestone — everything before this is kernel R&D.

---

#### `[ ]` ISSUE-08 — No KV cache; generation cost is quadratic
- **Files:** `inference.py:47`, `app.py:120`
- **Root cause:** Each token generation runs `model(input_ids)` over the full growing sequence from token 1 to token N. At `max_seq_len=256`, the 256th token pays 256× the attention compute of the first. A proper autoregressive engine computes key/value projections only once per token and caches them.
- **Concept to learn:** KV cache design — how past key/value tensors are stored and reused; how the attention mask shifts each step.
- **Fix:** Implement a `past_key_values` cache in `BitNet158.forward()` and update `generate()` to pass and extend it.
- **Expected gain:** Linear generation cost instead of quadratic; meaningful TPS improvement on long sequences.
