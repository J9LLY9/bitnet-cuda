"""
bitnet_kernel_loader.py

Shows how to JIT-compile bitnet_forward.cu and plug it into the existing
BitNet158 / BitLinear pipeline from model.py.

Usage
-----
    python bitnet_kernel_loader.py

Requirements
------------
    pip install torch  (with CUDA support)
    CUDA toolkit + nvcc on PATH  (must match the PyTorch CUDA version)
    On Windows: MSVC "x64 Native Tools" environment must be active
"""

import torch
# from torch.utils.cpp_extension import load  # DISABLED — kernel not compiled for this run

# ---------------------------------------------------------------------------
# 1. JIT-compile the CUDA kernel  (DISABLED for overnight training run)
#    Uncomment the block below once bitnet_forward.cu is complete and tested.
# ---------------------------------------------------------------------------
# bitnet_cuda = load(
#     name="bitnet_cuda",
#     sources=["bitnet_forward.cu"],
#     extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
#     verbose=True,
# )
bitnet_cuda = None  # placeholder so references below don't crash on import

# ---------------------------------------------------------------------------
# 2. Helper: pack a BitLinear weight tensor into 2-bit ternary int8 format
#
#    Packing layout (matches bitnet_forward.cu):
#      byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
#      encoding: 0→0b00, +1→0b01, -1→0b10
# ---------------------------------------------------------------------------
def pack_ternary_weights(weight: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Quantize and pack a (N, K) float weight tensor to (N, K//4) int8.

    Args:
        weight : raw floating-point weights from BitLinear, shape (N, K)
        gamma  : per-layer scale factor (scalar), i.e. weight.abs().mean()

    Returns:
        packed int8 tensor of shape (N, K//4) on CUDA, ready for the kernel
    """
    N, K = weight.shape
    assert K % 4 == 0, f"K={K} must be divisible by 4 for ternary 2-bit packing"

    # Quantize to {-1, 0, +1}  (mirrors BitLinear.forward's STE step)
    w_q = (weight / gamma.clamp(min=1e-5)).round().clamp(-1, 1).to(torch.int32)

    # Encode: 0→0, +1→1, -1→2
    encoded = torch.where(w_q > 0,
                  torch.ones_like(w_q),
                  torch.where(w_q < 0,
                      torch.full_like(w_q, 2),
                      torch.zeros_like(w_q)))        # (N, K)

    # Pack 4 consecutive weights per byte: shape (N, K//4, 4) → (N, K//4)
    e = encoded.view(N, K // 4, 4)
    packed = (e[:, :, 0]
              | (e[:, :, 1] << 2)
              | (e[:, :, 2] << 4)
              | (e[:, :, 3] << 6)).to(torch.int8)   # (N, K//4)

    return packed.cuda().contiguous()


# ---------------------------------------------------------------------------
# 3. Thin wrapper: mirrors BitLinear.forward but uses the CUDA kernel
#    instead of torch.nn.functional.linear.
# ---------------------------------------------------------------------------
def bitlinear_cuda_forward(
    x: torch.Tensor,           # (..., K) float16  — already RMSNorm'd
    B_packed: torch.Tensor,    # (N, K//4) int8
    gamma: float,              # original layer scale for magnitude restoration
) -> torch.Tensor:
    """Run the custom CUDA kernel and restore weight magnitude."""
    leading = x.shape[:-1]
    K = x.shape[-1]
    N = B_packed.shape[0]
    M = x[..., 0].numel()      # product of all leading dimensions

    x_2d = x.reshape(M, K).half().contiguous()

    # C = bitnet_cuda.bitnet_forward(x_2d, B_packed, M, K, N)  # DISABLED — kernel not ready
    raise RuntimeError("bitlinear_cuda_forward: CUDA kernel disabled for this run. Use BitLinear in model.py instead.")

    # Restore magnitude (gamma rescaling, same as BitLinear.forward)
    return (C.float() * gamma).half().reshape(*leading, N)


# ---------------------------------------------------------------------------
# 4. Smoke test — run against an actual BitLinear layer from model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model import BitNet158, get_device

    device = get_device()
    assert str(device) == "cuda", "This kernel requires a CUDA GPU"

    # Build the model (architecture must match any checkpoint you load)
    model = BitNet158(
        vocab_size=50257,
        embed_size=512,
        num_heads=8,
        num_layers=12,
        max_seq_len=256,
    ).to(device)
    model.eval()

    # Pick a layer to test — first attention Q-projection (512 → 512)
    layer = model.blocks[0].attention.W_q
    W = layer.weight.detach()           # (512, 512)
    gamma_val = W.abs().mean().clamp(min=1e-5)

    # Pack weights offline (do this once; cache B_packed for repeated inference)
    B_packed = pack_ternary_weights(W, gamma_val)
    print(f"Weight tensor  : {W.shape}  {W.dtype}")
    print(f"Packed weights : {B_packed.shape}  {B_packed.dtype}  "
          f"({W.numel()} values → {B_packed.numel()} bytes, "
          f"{W.numel() / B_packed.numel():.1f}× compression)")

    # Dummy input: batch=2, seq=32  →  M = 64 tokens, K = 512 features
    x_raw = torch.randn(2, 32, 512, dtype=torch.float32, device=device)
    x_normed = layer.act_norm(x_raw).half()   # SubLN normalisation (matches BitLinear)

    # --- Custom CUDA kernel output ---
    C_kernel = bitlinear_cuda_forward(x_normed, B_packed, gamma_val.item())

    # --- PyTorch reference output (BitLinear.forward, no STE needed at eval) ---
    w_q = (W / gamma_val).round().clamp(-1, 1)
    x_2d = x_normed.reshape(-1, 512).float()
    C_ref = (x_2d @ w_q.t() * gamma_val).half().reshape(2, 32, 512)

    max_err = (C_kernel.float() - C_ref.float()).abs().max().item()
    print(f"\nOutput shape   : {C_kernel.shape}  {C_kernel.dtype}")
    print(f"Max |error| vs PyTorch reference: {max_err:.6f}")
    print("PASS" if max_err < 0.05 else "MISMATCH — check packing")
