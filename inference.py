import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import BitNet158, get_device

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
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, past_key_values = model(input_ids, use_cache=True)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, input_ids, repetition_penalty)

            next_logits = next_logits / max(temperature, 1e-6)
            next_logits = top_p_filter(next_logits, top_p)

            probs      = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)

            next_logits, past_key_values = model(
                next_token, past_key_values=past_key_values, use_cache=True
            )
            next_logits = next_logits[:, -1, :]

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


# --- MAIN ---
if __name__ == "__main__":
    import os
    import re
    import glob
    device = str(get_device())
    print(f"Running on: {device}")

    SFT_TOKENIZER_DIR      = "sft_tokenizer"
    SFT_WEIGHTS            = "bitnet_sft.pt"
    BASE_WEIGHTS           = "bitnet_weights.pt"
    SAFETENSORS_CANDIDATES = [
        "BitNet_UW_Final_Gold_1.04.safetensors",
        "BITNET_1.05_HERO_WEIGHTS.safetensors",
        "bitnet_weights_final.safetensors",
    ]

    def _find_weights():
        if os.path.isdir(SFT_TOKENIZER_DIR) and os.path.isfile(SFT_WEIGHTS):
            return SFT_WEIGHTS, False
        if os.path.isfile(BASE_WEIGHTS):
            return BASE_WEIGHTS, False
        for name in SAFETENSORS_CANDIDATES:
            if os.path.isfile(name):
                return name, True
        ckpts = sorted(glob.glob("checkpoint_step*.safetensors"))
        return (ckpts[-1], True) if ckpts else (None, False)

    if os.path.isdir(SFT_TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(SFT_TOKENIZER_DIR)
        print(f"SFT tokenizer loaded from {SFT_TOKENIZER_DIR}/")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    model = BitNet158(vocab_size, embed_size=512, num_heads=8, num_layers=12, max_seq_len=256).to(device)

    weights_path, is_st = _find_weights()
    if weights_path is None:
        print("WARNING: no weights found — running with random weights.")
    else:
        try:
            if is_st:
                from safetensors.torch import load_file as st_load
                ckpt = st_load(weights_path, device=device)
                ckpt = {re.sub(r'^(blocks\.\d+)\.block\.', r'\1.', k): v for k, v in ckpt.items()}
                model.load_state_dict(ckpt, strict=True)
            else:
                model.load_state_dict(torch.load(weights_path, map_location=device))
            model.prepare_for_inference()
            print(f"Weights loaded from {weights_path}")
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