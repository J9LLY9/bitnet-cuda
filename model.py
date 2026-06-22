"""
model.py — BitNet-1.58b Transformer (canonical definition).

All other scripts (trainer.py, sft_trainer.py, inference.py, app.py) should
import from here rather than re-defining the architecture inline.

Key design choices vs. the original MLP:
  • RMSNorm everywhere instead of LayerNorm — no mean subtraction, numerically
    stabler with ternary weights, matches LLaMA / BitNet-b1.58 paper style.
  • SubLN inside BitLinear — each linear layer normalises its *input* activations
    with RMSNorm before the ternary projection, keeping the input in the range
    where the {-1, 0, 1} mapping is meaningful.  This is the primary fix for
    semantic drift at loss ~1.3.
  • bias=False throughout — redundant once inputs are RMSNorm'd.
  • Context window: 1 024 (hardware ceiling for T1000 8 GB at 12 L × 512 d).
  • Device: auto-selects cuda → mps → cpu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Custom CUDA kernel (optional — falls back to PyTorch if unavailable)
# ---------------------------------------------------------------------------
try:
    import bitnet_cuda
    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-select cuda (NVIDIA) > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Ternary packing utilities
# ---------------------------------------------------------------------------
def _pack_ternary(w_int: torch.Tensor) -> torch.Tensor:
    """Pack (N, K) int8 ternary weights {-1, 0, +1} into (N, K//4) int8."""
    N, K = w_int.shape
    code = torch.zeros(N, K, dtype=torch.uint8, device=w_int.device)
    code[w_int == 1] = 0b01
    code[w_int == -1] = 0b10
    packed = torch.zeros(N, K // 4, dtype=torch.int8, device=w_int.device)
    for b in range(4):
        packed |= (code[:, b::4] << (b * 2)).to(torch.int8)
    return packed


# ---------------------------------------------------------------------------
# BitNetFunction — torch.autograd.Function wrapping the CUDA kernel
# ---------------------------------------------------------------------------
class BitNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weights, M, K, N):
        return bitnet_cuda.bitnet_forward(x, packed_weights, M, K, N)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "BitNetFunction backward is not implemented (inference only)"
        )


# ---------------------------------------------------------------------------
# BitLinear  (1.58-bit ternary weights + SubLN activation normalisation)
# ---------------------------------------------------------------------------
class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with ternary weight quantisation.

    Training path (PyTorch fallback):
        SubLN → STE quantisation → F.linear → gamma rescale

    Inference path (custom CUDA kernel):
        SubLN → pre-packed ternary kernel → gamma rescale
        Activated by calling pack_for_inference() after loading weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.act_norm = RMSNorm(in_features)
        self.K_padded = ((in_features + 63) // 64) * 64
        self._pack_weights()

    @torch.no_grad()
    def _pack_weights(self):
        w = self.weight
        N, K = w.shape
        gamma = w.abs().mean().clamp(min=1e-5)
        w_quant = (w / gamma).round().clamp(-1, 1).to(torch.int8)

        if K < self.K_padded:
            w_quant = F.pad(w_quant, (0, self.K_padded - K), value=0)

        packed = _pack_ternary(w_quant)
        packed = (packed.view(N, self.K_padded // 64, 16)
                        .permute(1, 0, 2)
                        .contiguous())

        self.register_buffer('packed_weights', packed, persistent=False)
        self.register_buffer('weight_gamma', gamma.detach().clone(), persistent=False)

    def pack_for_inference(self):
        self._pack_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_norm(x)

        if _KERNEL_AVAILABLE and x.is_cuda and self.packed_weights is not None:
            orig_shape = x.shape
            N = self.out_features
            K = self.in_features
            M = x.numel() // K

            x_flat = x.contiguous().view(M, K)

            if K < self.K_padded:
                x_flat = F.pad(x_flat, (0, self.K_padded - K))

            input_dtype = x_flat.dtype
            if input_dtype != torch.float16:
                x_flat = x_flat.half()

            pw = self.packed_weights
            if pw.device != x_flat.device:
                pw = pw.to(x_flat.device)

            out = BitNetFunction.apply(x_flat, pw, M, self.K_padded, N)

            gamma = self.weight_gamma
            if gamma.device != out.device:
                gamma = gamma.to(out.device)

            out = out.to(input_dtype) * gamma
            return out.view(*orig_shape[:-1], N)
        else:
            w = self.weight
            gamma = w.abs().mean().clamp(min=1e-5)
            w_quant = (w / gamma).round().clamp(-1, 1)
            w_final = w + (w_quant - w).detach()
            return F.linear(x, w_final, self.bias) * gamma


# ---------------------------------------------------------------------------
# Causal Multi-Head Bit-Attention
# ---------------------------------------------------------------------------
class BitAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert embed_size % num_heads == 0, \
            f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim  = embed_size // num_heads

        self.W_q = BitLinear(embed_size, embed_size)
        self.W_k = BitLinear(embed_size, embed_size)
        self.W_v = BitLinear(embed_size, embed_size)
        self.W_o = BitLinear(embed_size, embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        scale = D ** -0.5
        attn  = (Q @ K.transpose(-2, -1)) * scale
        attn  = attn.masked_fill(mask == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# BitBlock  (Pre-Norm residual block)
# ---------------------------------------------------------------------------
class BitBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        self.norm1     = RMSNorm(embed_size)
        self.attention = BitAttention(embed_size, num_heads)
        self.norm2     = RMSNorm(embed_size)
        self.ffn_0     = BitLinear(embed_size, 4 * embed_size)
        self.ffn_2     = BitLinear(4 * embed_size, embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        h = F.gelu(self.ffn_0(self.norm2(x)))
        x = x + self.ffn_2(h)
        return x


# ---------------------------------------------------------------------------
# BitNet158  (full model)
# ---------------------------------------------------------------------------
class BitNet158(nn.Module):
    """
    BitNet-1.58b Transformer language model.

    Default spec (T1000 / P520 hardware ceiling):
        layers      : 12
        embed_size  : 512
        num_heads   : 8   (head_dim = 64)
        max_seq_len : 1 024
        vocab_size  : set by tokenizer (GPT-2 base = 50 257, +special tokens)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_size:  int = 512,
        num_heads:   int = 8,
        num_layers:  int = 12,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding   = nn.Embedding(max_seq_len, embed_size)

        self.blocks = nn.ModuleList([
            BitBlock(embed_size, num_heads) for _ in range(num_layers)
        ])

        self.norm    = RMSNorm(embed_size)
        self.lm_head = BitLinear(embed_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def prepare_for_inference(self):
        """Repack all BitLinear weights for the custom CUDA kernel.
        Call after loading weights and moving to device."""
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.pack_for_inference()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.token_embedding(x) + self.pos_embedding(pos)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(self.norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    print(f"CUDA kernel: {'available' if _KERNEL_AVAILABLE else 'not available (PyTorch fallback)'}")

    VOCAB  = 50_259
    model  = BitNet158(vocab_size=VOCAB).to(device)

    if _KERNEL_AVAILABLE and device.type == "cuda":
        model.prepare_for_inference()
        print("Packed weights prepared for CUDA kernel")

    model.eval()
    params = model.num_parameters()
    print(f"Parameters: {params:,}  ({params/1e6:.1f} M)")

    dummy = torch.randint(0, VOCAB, (2, 128)).to(device)
    with torch.no_grad():
        logits = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Logits shape: {logits.shape}   ← expect (2, 128, {VOCAB})")

    sample_block = model.blocks[0]
    print(f"\nBlock 0 attention W_q act_norm weight: "
          f"mean={sample_block.attention.W_q.act_norm.weight.mean():.4f}  "
          f"(ones at init → model is untrained)")

    print("\nBitNet-1.58b sanity check PASSED.")
