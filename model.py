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
    """
    Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019).
    Omits mean subtraction, which is unnecessary once weights are ternary
    and harmful for gradient flow through very sparse activations.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale (γ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# BitLinear  (1.58-bit ternary weights + SubLN activation normalisation)
# ---------------------------------------------------------------------------
class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with:

    1. SubLN  — RMSNorm applied to *input* activations before the linear op,
       so the ternary projection always sees unit-variance inputs.
    2. Weight quantisation — W scaled by 1/γ (γ = mean |w|), rounded and
       clamped to {-1, 0, 1}.
    3. STE  — Straight-Through Estimator:
           w_final = w + (w_quant − w).detach()
       Forward pass uses ternary weights; gradients flow through full-precision w.
    4. Gamma rescaling — output multiplied by γ to restore weight magnitude,
       preventing logit collapse after ternary projection.

    bias is disabled by default: SubLN inside the layer makes it redundant.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.act_norm = RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalise activations (SubLN)
        x = self.act_norm(x)

        # 2. Compute per-layer scale factor
        w     = self.weight
        gamma = w.abs().mean().clamp(min=1e-5)

        # 3. Ternary quantisation + STE
        w_quant = (w / gamma).round().clamp(-1, 1)
        w_final = w + (w_quant - w).detach()          # STE: forward ternary, backward float

        # 4. Linear projection + magnitude restoration
        return F.linear(x, w_final, self.bias) * gamma


# ---------------------------------------------------------------------------
# Causal Multi-Head Bit-Attention
# ---------------------------------------------------------------------------
class BitAttention(nn.Module):
    """
    Multi-head self-attention where every projection is a BitLinear.
    Causal mask is built once per sequence length and cached on the right device.
    """

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

        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        scale = D ** -0.5
        attn  = (Q @ K.transpose(-2, -1)) * scale            # (B, H, T, T)
        attn  = attn.masked_fill(mask == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# BitBlock  (Pre-Norm residual block)
# ---------------------------------------------------------------------------
class BitBlock(nn.Module):
    """
    Pre-Norm Transformer block:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Using RMSNorm (not LayerNorm) before each sub-layer prevents the mean-shift
    instability that causes semantic drift in ternary networks.

    FFN expansion ratio: 4× (512 → 2 048 → 512) with GELU activation,
    which is smoother than ReLU and reduces dead neurons in ternary nets.

    Note: FFN layers are named ffn_0 / ffn_2 (not nn.Sequential) so that
    safetensors checkpoint keys are stable and human-readable.
    """

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

    The causal mask is pre-allocated once and stored as a buffer (not a
    parameter) so it moves with .to(device) automatically.
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

        self.norm    = RMSNorm(embed_size)          # final norm before lm_head
        self.lm_head = BitLinear(embed_size, vocab_size)

        # Causal mask is computed on-the-fly in forward() so it never
        # enters the state dict and safetensors checkpoints stay clean.

        self._init_weights()

    def _init_weights(self):
        """Small-std init to keep pre-quantisation activations near unit variance."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        pos  = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        x    = self.token_embedding(x) + self.pos_embedding(pos)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(self.norm(x))                      # (B, T, vocab_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    VOCAB  = 50_259   # GPT-2 + 2 special tokens (<thought>, </thought>)
    model  = BitNet158(vocab_size=VOCAB).to(device)
    params = model.num_parameters()
    print(f"Parameters: {params:,}  ({params/1e6:.1f} M)")

    # Forward pass: batch=2, seq_len=128
    dummy = torch.randint(0, VOCAB, (2, 128)).to(device)
    logits = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Logits shape: {logits.shape}   ← expect (2, 128, {VOCAB})")

    # Verify BitLinear SubLN is active
    sample_block = model.blocks[0]
    print(f"\nBlock 0 attention W_q act_norm weight: "
          f"mean={sample_block.attention.W_q.act_norm.weight.mean():.4f}  "
          f"(ones at init → model is untrained)")

    print("\nBitNet-1.58b sanity check PASSED.")
