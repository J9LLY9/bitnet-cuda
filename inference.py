import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --- ARCHITECTURE (must match trainer.py exactly) ---
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        gamma = w.abs().mean()
        w_quant = (w / (gamma + 1e-5)).round().clamp(-1, 1)
        w_final = w + (w_quant - w).detach()
        # MUST have this * gamma at the end!
        return F.linear(x, w_final, self.bias) * gamma


class BitAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.W_q = BitLinear(embed_size, embed_size)
        self.W_k = BitLinear(embed_size, embed_size)
        self.W_v = BitLinear(embed_size, embed_size)
        self.W_o = BitLinear(embed_size, embed_size)

    def forward(self, x, mask):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class BitBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = BitAttention(embed_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            BitLinear(embed_size, 4 * embed_size),
            nn.ReLU(),
            BitLinear(4 * embed_size, embed_size),
        )

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BitNetLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads=8, num_layers=4, max_seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        self.blocks = nn.ModuleList([BitBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.lm_head = BitLinear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

# --- GENERATION ---
def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """
    Penalise tokens that already appear in input_ids.
    Positive logits are divided by `penalty`; negative logits are multiplied.
    This consistently pushes seen tokens away from the top regardless of sign.
    penalty=1.0 is a no-op; values like 1.2–1.5 work well in practice.
    """
    seen_token_ids = input_ids[0].unique()
    score = logits[0, seen_token_ids]
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[0, seen_token_ids] = score
    return logits


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (Top-P) filter: keep the smallest set of tokens whose cumulative
    softmax probability exceeds `top_p`, zero out the rest.
    The shift-before-compare ensures the token that crosses the threshold is kept.
    top_p=1.0 is a no-op (full vocabulary).
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right: remove the current token's own probability before comparing,
    # so the token that first pushes cumulative_probs over top_p is retained.
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50,
             temperature: float = 1.0, top_p: float = 0.9,
             repetition_penalty: float = 1.3, device: str = "cuda:0") -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits      = model(input_ids)           # (1, seq_len, vocab_size)
            next_logits = logits[:, -1, :]           # (1, vocab_size)

            # 1. Repetition penalty
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, input_ids, repetition_penalty)

            # 2. Temperature scaling (must come before Top-P so the distribution
            #    is already sharpened/flattened when we measure probability mass)
            next_logits = next_logits / max(temperature, 1e-6)

            # 3. Nucleus (Top-P) filter
            next_logits = top_p_filter(next_logits, top_p)

            # 4. Sample
            probs      = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


# --- MAIN ---
if __name__ == "__main__":
    import os
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Prefer the SFT tokenizer (includes <thought> token) when available
    SFT_TOKENIZER_DIR = "sft_tokenizer"
    SFT_WEIGHTS       = "bitnet_sft.pt"
    BASE_WEIGHTS      = "bitnet_weights.pt"

    if os.path.isdir(SFT_TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(SFT_TOKENIZER_DIR)
        weights_path = SFT_WEIGHTS
        print(f"SFT tokenizer loaded from {SFT_TOKENIZER_DIR}/")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        weights_path = BASE_WEIGHTS

    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)   # use len() to include any added special tokens

    model = BitNetLanguageModel(vocab_size, embed_size=512, num_heads=8, num_layers=12).to(device)

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Weights loaded from {weights_path}")
    except FileNotFoundError:
        print(f"WARNING: {weights_path} not found — running with random weights.")
    except RuntimeError as e:
        raise RuntimeError(
            f"Architecture mismatch loading {weights_path} — "
            "check that num_layers/embed_size match the checkpoint.\n"
            f"Original error: {e}"
        )

    # The Interactive Prompt (Make sure this line is EXACTLY like this)
    prompt = input("\nEnter your prompt: ")
    
    print(f"\nPrompt: {prompt!r}")
    print("Generating 50 tokens...\n")

    output = generate(
        model, tokenizer, prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        device=device,
    )

    print("Generated text:")
    print(output)