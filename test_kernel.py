"""
test_kernel.py — correctness test for the bitnet_cuda shared-memory kernel.

What this verifies
------------------
1. Gate logic  : does ternary unpacking (0b00→0, 0b01→+1, 0b10→-1) produce
                 the right weight values?
2. Shared-memory tiling : does the tiled accumulation give the same numbers
                          as a plain PyTorch matmul on the unpacked weights?

Run:
    python test_kernel.py
"""

import sys
import struct
import torch
import bitnet_cuda   # the compiled extension (run setup.py first)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pack_ternary(W_int: torch.Tensor) -> torch.Tensor:
    """
    Pack a (N, K) int8 weight tensor of values in {-1, 0, +1} into the
    2-bit-per-weight format expected by the kernel.

    Encoding:  0 → 0b00,  +1 → 0b01,  -1 → 0b10
    Returns a (N, K//4) int8 tensor.
    """
    assert W_int.shape[1] % 4 == 0, "K must be divisible by 4"
    N, K = W_int.shape
    K4 = K // 4
    packed = torch.zeros(N, K4, dtype=torch.int8)

    # Encode each weight value into its 2-bit code.
    code = torch.zeros_like(W_int, dtype=torch.uint8)
    code[W_int ==  1] = 0b01
    code[W_int == -1] = 0b10
    # code[W_int ==  0] = 0b00  (already zero)

    # Pack four consecutive codes into one byte.
    for b in range(4):
        packed |= (code[:, b::4] << (b * 2)).to(torch.int8)

    return packed


def apply_kernel_layout(B_packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Transpose packed weights into the tiled layout expected by kernel v6."""
    return B_packed.view(N, K // 64, 16).permute(1, 0, 2).contiguous()


def unpack_ternary(B_packed: torch.Tensor, K: int) -> torch.Tensor:
    """
    Inverse of pack_ternary. Returns a (N, K) float32 weight matrix.
    Used to build the reference PyTorch matmul.
    """
    N, K4 = B_packed.shape
    W = torch.zeros(N, K, dtype=torch.float32)
    raw = B_packed.cpu().to(torch.int32) & 0xFF   # treat as unsigned

    for b in range(4):
        codes = (raw >> (b * 2)) & 0x3            # shape (N, K4)
        col = b                                    # absolute K offset starts at b
        W[:, col::4][codes == 1] =  1.0
        W[:, col::4][codes == 2] = -1.0

    return W


def reference_matmul(A_fp16: torch.Tensor, W_float: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ W^T in float32, round-trip through fp16 to match
    the kernel's store precision, and return an fp16 tensor.
    """
    A_f32 = A_fp16.float().cpu()
    C_f32 = A_f32 @ W_float.T          # (M, N)
    return C_f32.half()                 # match the kernel's output dtype


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_fixed():
    """
    Tiny hand-crafted case where we can reason about every number.

    A   = [[1, 0, -1, 2, 0, ...]]   shape (1, 64)
    W   = [[1, 1,  0, 0, 0, ...],   shape (2, 64)
           [0, 1, -1, 1]]
    C   = A @ W^T
        row 0 = [1*1+0*1+(-1)*0+2*0,  1*0+0*1+(-1)*(-1)+2*1]
              = [1,                    3]
    """
    print("=" * 60)
    print("TEST 1 — Fixed hand-crafted values")
    print("=" * 60)

    M, K, N = 1, 64, 2

    A_vals = torch.zeros((M, K), dtype=torch.float16, device="cuda")
    A_vals[:, :4] = torch.tensor([[1.0, 0.0, -1.0, 2.0]], dtype=torch.float16, device="cuda")
    W_int = torch.zeros((N, K), dtype=torch.int8)
    W_int[:, :4] = torch.tensor([[1,  1,  0, 0],
                                 [0,  1, -1, 1]], dtype=torch.int8)
    B_packed = apply_kernel_layout(pack_ternary(W_int), N, K).cuda()

    print(f"  [DEBUG] B_packed.shape = {B_packed.shape}")
    print(f"  [DEBUG] M={M}, K={K}, N={N}")
    print(f"  [DEBUG] B_packed.dtype is torch.int8: {B_packed.dtype == torch.int8} (actual: {B_packed.dtype})")

    scale = 127.0 / A_vals.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    A_quant = (A_vals * scale).round().clamp(-128, 127).to(torch.int8)
    C_kernel = bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N)
    C_kernel = C_kernel.half() / scale
    C_ref    = torch.tensor([[1.0, 3.0]], dtype=torch.float16)

    print(f"  Expected : {C_ref}")
    print(f"  Kernel   : {C_kernel.cpu()}")

    if torch.allclose(C_kernel.cpu(), C_ref, atol=1e-2):
        print("  PASS\n")
    else:
        diff = (C_kernel.cpu() - C_ref).abs()
        print(f"  FAIL — max abs error: {diff.max().item():.6f}\n")
        sys.exit(1)


def test_random(M=64, K=128, N=64, seed=42):
    """
    Random ternary weights and fp16 activations.
    Checks that the kernel output matches the unpacked-weight PyTorch matmul
    within fp16 rounding tolerance.
    """
    print("=" * 60)
    print(f"TEST 2 — Random  M={M}  K={K}  N={N}  seed={seed}")
    print("=" * 60)

    torch.manual_seed(seed)

    # Random fp16 activations in [-2, 2]
    A_fp16 = (torch.rand(M, K, dtype=torch.float32) * 4 - 2).half().cuda()

    # Random ternary weights drawn from {-1, 0, +1} with roughly 50% zeros
    W_int    = torch.randint(-1, 2, (N, K), dtype=torch.int8)   # uniform {-1,0,+1}
    B_packed = pack_ternary(W_int)

    # Reference uses the canonical (N, K/4) packing; v6 uses tiled layout.
    W_float = unpack_ternary(B_packed, K)
    B_packed = apply_kernel_layout(B_packed, N, K).cuda()

    print(f"  [DEBUG] B_packed.shape = {B_packed.shape}")
    print(f"  [DEBUG] M={M}, K={K}, N={N}")
    print(f"  [DEBUG] B_packed.dtype is torch.int8: {B_packed.dtype == torch.int8} (actual: {B_packed.dtype})")

    # Kernel output
    scale = 127.0 / A_fp16.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    A_quant = (A_fp16 * scale).round().clamp(-128, 127).to(torch.int8)
    C_kernel = bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N)
    C_kernel = C_kernel.half() / scale

    # Reference: unpack weights, do fp32 matmul, store as fp16
    C_ref   = reference_matmul(A_fp16, W_float)

    # fp16 accumulation can drift; allow a small absolute tolerance.
    # The kernel accumulates in fp32 but A is stored as fp16, so small
    # representational differences vs. the fp32 reference are expected.
    atol = 0.25   # accommodate W2A8 quantization noise
    rtol = 1e-2

    passed = torch.allclose(C_kernel.cpu(), C_ref, atol=atol, rtol=rtol)

    max_err  = (C_kernel.cpu() - C_ref).abs().max().item()
    mean_err = (C_kernel.cpu() - C_ref).abs().mean().item()
    print(f"  Max  abs error : {max_err:.6f}")
    print(f"  Mean abs error : {mean_err:.6f}")
    print(f"  Tolerance      : atol={atol}")

    if passed:
        print("  PASS\n")
    else:
        print("  FAIL — outputs diverge beyond tolerance\n")
        sys.exit(1)


def test_boundary(M=17, K=128, N=33, seed=7):
    """
    Non-power-of-two shapes that stress the tile boundary handling.
    M=17 and N=33 are not multiples of BLOCK=16, so the last tile column/row
    contains out-of-bounds threads that must be masked off correctly.
    """
    print("=" * 60)
    print(f"TEST 3 — Boundary shapes  M={M}  K={K}  N={N}  seed={seed}")
    print("=" * 60)

    torch.manual_seed(seed)

    A_fp16   = (torch.rand(M, K, dtype=torch.float32) * 2 - 1).half().cuda()
    W_int    = torch.randint(-1, 2, (N, K), dtype=torch.int8)
    B_packed = pack_ternary(W_int)
    W_float = unpack_ternary(B_packed, K)
    B_packed = apply_kernel_layout(B_packed, N, K).cuda()

    print(f"  [DEBUG] B_packed.shape = {B_packed.shape}")
    print(f"  [DEBUG] M={M}, K={K}, N={N}")
    print(f"  [DEBUG] B_packed.dtype is torch.int8: {B_packed.dtype == torch.int8} (actual: {B_packed.dtype})")

    scale = 127.0 / A_fp16.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    A_quant = (A_fp16 * scale).round().clamp(-128, 127).to(torch.int8)
    C_kernel = bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N)
    C_kernel = C_kernel.half() / scale
    C_ref    = reference_matmul(A_fp16, W_float)

    max_err = (C_kernel.cpu() - C_ref).abs().max().item()
    print(f"  Max abs error : {max_err:.6f}")

    if torch.allclose(C_kernel.cpu(), C_ref, atol=0.25, rtol=1e-2):
        print("  PASS\n")
    else:
        print("  FAIL — boundary handling is broken\n")
        sys.exit(1)


def test_all_zero_weights(M=16, K=64, N=16):
    """
    All weights zero → C must be all zeros regardless of A.
    Verifies that the 'skip zero' branch doesn't accidentally accumulate.
    """
    print("=" * 60)
    print("TEST 4 — All-zero weights (C must be zero)")
    print("=" * 60)

    A_fp16   = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W_int    = torch.zeros(N, K, dtype=torch.int8)
    B_packed = apply_kernel_layout(pack_ternary(W_int), N, K).cuda()

    print(f"  [DEBUG] B_packed.shape = {B_packed.shape}")
    print(f"  [DEBUG] M={M}, K={K}, N={N}")
    print(f"  [DEBUG] B_packed.dtype is torch.int8: {B_packed.dtype == torch.int8} (actual: {B_packed.dtype})")

    scale = 127.0 / A_fp16.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    A_quant = (A_fp16 * scale).round().clamp(-128, 127).to(torch.int8)
    C_kernel = bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N)

    if torch.all(C_kernel == 0):
        print("  PASS\n")
    else:
        print(f"  FAIL — got non-zero outputs: {C_kernel.abs().max().item()}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, n_warmup: int, n_iters: int) -> float:
    """
    Run `fn()` for `n_warmup` iterations (discarded), then time `n_iters`
    iterations with CUDA Events and return the average milliseconds per call.

    torch.cuda.Event is the correct tool here: it timestamps work on the GPU
    timeline rather than on the host, so PCIe round-trip latency and Python
    overhead are excluded from the measurement.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)

    t_start.record()
    for _ in range(n_iters):
        fn()
    t_end.record()

    torch.cuda.synchronize()
    return t_start.elapsed_time(t_end) / n_iters   # ms per call


def benchmark(M: int = 4096, K: int = 4096, N: int = 4096,
              n_warmup: int = 5, n_iters: int = 50) -> None:
    """
    Compare three execution paths on a single large matrix multiplication:

      1. PyTorch fp16 nn.Linear  — cuBLAS + Tensor Cores (the "gold standard")
      2. PyTorch fp32 nn.Linear  — cuBLAS without Tensor Cores (reference)
      3. Custom BitNet kernel     — tiled shared-memory ternary kernel

    TOPS metric
    -----------
    We count 2·M·K·N operations (the dense-matmul convention) for all three
    paths so the numbers sit on the same ruler.  The custom kernel skips zero
    weights at runtime, so its *effective* work is lower, but 2MKN lets you
    compare raw throughput directly against cuBLAS specs for the RTX 3050.

    RTX 3050 peak numbers for context
    ----------------------------------
      fp16 Tensor Core : ~57 TOPS
      fp16 CUDA Core   :  ~9 TOPS
      Memory bandwidth :  112 GB/s
    Our kernel uses CUDA cores + shared-memory tiling, so peak is ~9 TOPS.
    cuBLAS uses Tensor Cores, so it can reach multiples of that.
    """
    print("=" * 60)
    print(f"BENCHMARK  M={M}  K={K}  N={N}  ({n_iters} timed iterations)")
    print("=" * 60)

    # ── Inputs ──────────────────────────────────────────────────────────────

    torch.manual_seed(0)

    A_fp16 = (torch.rand(M, K, dtype=torch.float32) * 4 - 2).half().cuda()
    A_fp32 = A_fp16.float()

    # Ternary weight matrix {-1, 0, +1} — same weights used everywhere.
    W_int    = torch.randint(-1, 2, (N, K), dtype=torch.int8)
    B_packed = pack_ternary(W_int)

    # Unpacked fp16 and fp32 weight matrices for the PyTorch baselines.
    # unpack_ternary() returns float32; casting to fp16 matches nn.Linear input.
    W_f32 = unpack_ternary(B_packed, K).cuda()   # (N, K)
    W_f16 = W_f32.half()
    B_packed = apply_kernel_layout(B_packed, N, K).cuda()

    # ── PyTorch fp16 baseline (cuBLAS + Tensor Cores) ───────────────────────

    def run_torch_fp16():
        torch.nn.functional.linear(A_fp16, W_f16)   # C = A @ W^T

    ms_torch_fp16 = _cuda_time_ms(run_torch_fp16, n_warmup, n_iters)

    # ── PyTorch fp32 baseline (cuBLAS, no Tensor Cores) ─────────────────────

    def run_torch_fp32():
        torch.nn.functional.linear(A_fp32, W_f32)

    ms_torch_fp32 = _cuda_time_ms(run_torch_fp32, n_warmup, n_iters)

    # ── Custom BitNet kernel (W2A8, __dp4a) ────────────────────────────────

    scale = 127.0 / A_fp16.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    A_quant = (A_fp16 * scale).round().clamp(-128, 127).to(torch.int8)

    def run_custom():
        bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N)

    ms_custom = _cuda_time_ms(run_custom, n_warmup, n_iters)

    # ── Reporting ────────────────────────────────────────────────────────────

    # Dense-equivalent op count: one multiply + one add per output element per K step.
    total_ops  = 2.0 * M * K * N            # flops / call
    tops_f16   = total_ops / (ms_torch_fp16 * 1e-3) / 1e12
    tops_f32   = total_ops / (ms_torch_fp32 * 1e-3) / 1e12
    tops_cust  = total_ops / (ms_custom     * 1e-3) / 1e12

    speedup_vs_f16 = ms_torch_fp16 / ms_custom
    speedup_vs_f32 = ms_torch_fp32 / ms_custom

    # Memory traffic for the custom kernel (helps assess bandwidth utilisation).
    bytes_A      = M * K * 1                          # int8
    bytes_Bpack  = N * (K // 4) * 1                  # int8 packed
    bytes_C      = M * N * 4                          # int32
    gb_custom    = (bytes_A + bytes_Bpack + bytes_C) / 1e9
    bw_custom_gbs = gb_custom / (ms_custom * 1e-3)   # GB/s achieved

    print(f"\n  {'Kernel':<30}  {'Time (ms)':>10}  {'TOPS':>8}  {'vs fp32':>8}  {'vs fp16':>8}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'PyTorch fp16 (cuBLAS+TC)':<30}  {ms_torch_fp16:>10.3f}  {tops_f16:>8.2f}       ---       ---")
    print(f"  {'PyTorch fp32 (cuBLAS)':<30}  {ms_torch_fp32:>10.3f}  {tops_f32:>8.2f}       ---       ---")
    print(f"  {'Custom BitNet (ternary tile)':<30}  {ms_custom:>10.3f}  {tops_cust:>8.2f}  {speedup_vs_f32:>7.2f}x  {speedup_vs_f16:>7.2f}x")

    print(f"\n  Memory traffic (custom kernel)")
    print(f"    A         : {bytes_A   /1e6:6.1f} MB  (int8, quantized activation)")
    print(f"    B_packed  : {bytes_Bpack/1e6:6.1f} MB  (int8, 4x compressed vs fp16 dense)")
    print(f"    C         : {bytes_C   /1e6:6.1f} MB  (int32 output)")
    print(f"    Total     : {gb_custom*1e3:6.1f} MB  →  {bw_custom_gbs:.1f} GB/s achieved")
    print(f"    RTX 3050 peak bandwidth : 112 GB/s")
    print(f"    Bandwidth utilisation   : {100*bw_custom_gbs/112:.1f}%\n")

    if speedup_vs_f32 >= 1.0:
        print(f"  Custom kernel is {speedup_vs_f32:.2f}x faster than fp32 cuBLAS.")
    else:
        print(f"  Custom kernel is {1/speedup_vs_f32:.2f}x SLOWER than fp32 cuBLAS.")
        print("  (Expected: cuBLAS uses Tensor Cores; our kernel uses CUDA cores.)")
        print("  The BitNet advantage is the 4x smaller weight footprint and skipped zeros,")
        print("  which matters more at inference-scale batch sizes, not M=K=N=4096 benchmarks.")
    print()


def benchmark_sweep() -> None:
    """
    Run the benchmark across a range of problem sizes to show how the
    performance gap evolves.  Small M (single-token inference) is where
    our kernel is memory-bandwidth-bound and its compressed B pays off most.
    """
    print("\n" + "=" * 60)
    print("SWEEP — speedup vs fp32 cuBLAS across batch sizes (K=N=4096)")
    print("=" * 60)
    print(f"\n  {'M (batch)':>10}  {'fp16 (ms)':>10}  {'fp32 (ms)':>10}  {'custom (ms)':>12}  {'speedup vs fp32':>16}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*16}")

    K, N = 4096, 4096
    for M in [1, 8, 32, 128, 512, 2048, 4096]:
        torch.manual_seed(0)
        A_fp16   = (torch.rand(M, K) * 4 - 2).half().cuda()
        A_fp32   = A_fp16.float()
        W_int    = torch.randint(-1, 2, (N, K), dtype=torch.int8)
        B_packed = pack_ternary(W_int)
        W_f32    = unpack_ternary(B_packed, K).cuda()
        W_f16    = W_f32.half()
        B_packed = apply_kernel_layout(B_packed, N, K).cuda()

        iters = 200 if M <= 32 else 50

        scale = 127.0 / A_fp16.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        A_quant = (A_fp16 * scale).round().clamp(-128, 127).to(torch.int8)

        ms_f16  = _cuda_time_ms(lambda: torch.nn.functional.linear(A_fp16, W_f16), 5, iters)
        ms_f32  = _cuda_time_ms(lambda: torch.nn.functional.linear(A_fp32, W_f32), 5, iters)
        ms_cust = _cuda_time_ms(lambda: bitnet_cuda.bitnet_forward(A_quant, B_packed, M, K, N), 5, iters)
        spd     = ms_f32 / ms_cust

        print(f"  {M:>10}  {ms_f16:>10.3f}  {ms_f32:>10.3f}  {ms_cust:>12.3f}  {spd:>14.2f}x")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Tests require a GPU.")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice : {dev}")
    print(f"PyTorch: {torch.__version__}\n")

    test_fixed()
    test_random()
    test_boundary()
    test_all_zero_weights()

    print("All tests passed.\n")

    benchmark()
    benchmark_sweep()
