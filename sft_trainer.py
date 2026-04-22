"""
sft_trainer.py — Supervised Fine-Tuning (SFT) for BitNet 1.58-bit Transformer.

Pipeline:
  1. Load pre-trained weights from bitnet_weights.pt
  2. Add <thought> / </thought> special tokens and resize embeddings
  3. Pull 1,000 examples from tatsu-lab/alpaca (Stanford Alpaca instruction set)
     Note: replace DATASET_NAME/DATASET_SPLIT if you have access to a different
     instruction corpus (e.g. microsoft/phi-ct-tiny if it becomes public).
  4. Fine-tune with causal LM loss masked on the instruction portion — only the
     model's *response* (including the <thought> token) contributes to the loss.
  5. Save tokenizer + weights to bitnet_sft.pt / sft_tokenizer/

Effective batch: MICRO_BATCH * ACCUM_STEPS = 4 * 8 = 32
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_WEIGHTS    = "bitnet_weights.pt"
SFT_WEIGHTS_OUT = "bitnet_sft.pt"
TOKENIZER_OUT   = "sft_tokenizer"

DATASET_NAME  = "tatsu-lab/alpaca"
DATASET_SPLIT = "train[:1000]"          # exactly 1 000 examples

EMBED_SIZE  = 512
NUM_HEADS   = 8
NUM_LAYERS  = 12
MAX_SEQ_LEN = 256

SFT_LR       = 5e-5   # 10× lower than pre-training (5e-4)
MIN_LR       = 1e-6
MICRO_BATCH  = 4
ACCUM_STEPS  = 8       # effective batch = 32
SFT_EPOCHS   = 4       # 1 000 examples / 32 effective = ~31 steps/epoch → 125 total
LOG_EVERY    = 25
GRAD_CLIP    = 1.0

THOUGHT_START = "<thought>"
THOUGHT_END   = "</thought>"

# ---------------------------------------------------------------------------
# Architecture (identical to trainer.py — must not diverge)
# ---------------------------------------------------------------------------
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        gamma = w.abs().mean()
        w_quant = (w / (gamma + 1e-5)).round().clamp(-1, 1)
        w_final = w + (w_quant - w).detach()
        return F.linear(x, w_final, self.bias) * gamma


class BitAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_size // num_heads
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
        attn  = (Q @ K.transpose(-2, -1)) * scale
        attn  = attn.masked_fill(mask == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        out   = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
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
    def __init__(self, vocab_size, embed_size, num_heads=8, num_layers=12, max_seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding   = nn.Embedding(max_seq_len, embed_size)
        self.blocks  = nn.ModuleList([BitBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.norm    = nn.LayerNorm(embed_size)
        self.lm_head = BitLinear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.token_embedding(x) + self.pos_embedding(pos)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# Embedding resize: extend token_embedding + lm_head for new special tokens
# ---------------------------------------------------------------------------
def resize_model_embeddings(model, new_vocab_size, embed_size):
    old_vocab = model.token_embedding.num_embeddings
    if new_vocab_size <= old_vocab:
        return
    n_new = new_vocab_size - old_vocab
    print(f"Resizing embeddings: {old_vocab} → {new_vocab_size} (+{n_new} tokens)")

    # Token embedding
    old_emb = model.token_embedding
    new_emb = nn.Embedding(new_vocab_size, embed_size)
    new_emb.weight.data[:old_vocab] = old_emb.weight.data
    # Init new rows near zero so they start neutral
    nn.init.normal_(new_emb.weight.data[old_vocab:], mean=0.0, std=0.02)
    model.token_embedding = new_emb

    # LM head (BitLinear)
    old_head = model.lm_head
    has_bias = old_head.bias is not None
    new_head = BitLinear(embed_size, new_vocab_size, bias=has_bias)
    new_head.weight.data[:old_vocab] = old_head.weight.data
    nn.init.normal_(new_head.weight.data[old_vocab:], mean=0.0, std=0.02)
    if has_bias:
        new_head.bias.data[:old_vocab] = old_head.bias.data
        new_head.bias.data[old_vocab:] = 0.0
    model.lm_head = new_head


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
PROMPT_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


class InstructionDataset(Dataset):
    """
    Tokenizes each example as:
        [PROMPT tokens] [<thought>\n RESPONSE tokens EOS]
                         ↑ loss starts here ↑

    The prompt portion is masked with -100 so the model only learns
    to predict its own response, not to memorise the instruction format.
    """

    def __init__(self, examples, tokenizer, max_len):
        self.items = []
        eos_id = tokenizer.eos_token_id

        for ex in examples:
            instruction = ex.get("instruction", "").strip()
            inp         = ex.get("input", "").strip()
            output      = ex.get("output", "").strip()

            if not instruction or not output:
                continue

            # Format prompt
            if inp:
                prompt_text = PROMPT_TEMPLATE_WITH_INPUT.format(
                    instruction=instruction, input=inp
                )
            else:
                prompt_text = PROMPT_TEMPLATE_NO_INPUT.format(
                    instruction=instruction
                )

            # Response always begins with <thought> so the model learns to use it
            response_text = f"{THOUGHT_START}\n{output}"

            prompt_ids   = tokenizer.encode(prompt_text,   add_special_tokens=False)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)
            response_ids.append(eos_id)   # teach the model when to stop

            full_ids = (prompt_ids + response_ids)[:max_len]

            # Labels: -100 for the prompt, real token ids for the response
            prompt_len   = min(len(prompt_ids), max_len)
            response_len = len(full_ids) - prompt_len
            labels       = [-100] * prompt_len + full_ids[prompt_len:]

            # Pad to max_len
            pad_len  = max_len - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels   = labels   + [-100] * pad_len

            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels":    torch.tensor(labels,   dtype=torch.long),
            })

        print(f"Dataset built: {len(self.items)} usable examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Tokenizer + special tokens ─────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [THOUGHT_START, THOUGHT_END]
    })
    print(f"Added {num_added} special token(s): {THOUGHT_START}  {THOUGHT_END}")
    print(f"  <thought> id = {tokenizer.convert_tokens_to_ids(THOUGHT_START)}")
    print(f"  </thought> id = {tokenizer.convert_tokens_to_ids(THOUGHT_END)}")

    new_vocab_size = len(tokenizer)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Building model (vocab_size={new_vocab_size})...")
    # Build with original vocab first so we can load pre-trained weights cleanly
    model = BitNetLanguageModel(
        vocab_size  = tokenizer.vocab_size,   # original GPT-2 vocab
        embed_size  = EMBED_SIZE,
        num_heads   = NUM_HEADS,
        num_layers  = NUM_LAYERS,
        max_seq_len = MAX_SEQ_LEN,
    ).to(device)

    if os.path.exists(BASE_WEIGHTS):
        model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
        print(f"Pre-trained weights loaded from {BASE_WEIGHTS}")
    else:
        print(f"WARNING: {BASE_WEIGHTS} not found — starting from random init.")

    # Extend embeddings for the two new tokens
    resize_model_embeddings(model, new_vocab_size, EMBED_SIZE)
    model = model.to(device)

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET_NAME}  split={DATASET_SPLIT}")
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset = InstructionDataset(raw, tokenizer, max_len=MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=True,
                         num_workers=2, pin_memory=True)

    # ── Optimiser / scheduler ──────────────────────────────────────────────
    # Total optimiser steps = epochs * (dataset / effective_batch)
    steps_per_epoch = max(1, len(dataset) // (MICRO_BATCH * ACCUM_STEPS))
    total_steps     = SFT_EPOCHS * steps_per_epoch
    print(f"Training plan: {SFT_EPOCHS} epochs × {steps_per_epoch} steps = {total_steps} total optimiser steps")
    print(f"LR: {SFT_LR} → {MIN_LR}  (CosineAnnealing)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer_step   = 0
    accum_count      = 0
    running_loss     = 0.0
    optimizer.zero_grad()

    for epoch in range(SFT_EPOCHS):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)   # (B, T)
            labels    = batch["labels"].to(device)       # (B, T)  — -100 on prompt

            logits = model(input_ids)                    # (B, T, V)

            # Shift: predict token[t+1] from token[t]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, new_vocab_size),
                shift_labels.view(-1),
            )

            (loss / ACCUM_STEPS).backward()
            running_loss += loss.item()
            accum_count  += 1

            if accum_count == ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
                accum_count = 0

                if optimizer_step % LOG_EVERY == 0:
                    avg_loss = running_loss / (LOG_EVERY * ACCUM_STEPS)
                    lr_now   = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch+1}/{SFT_EPOCHS}  step {optimizer_step}/{total_steps}"
                          f"  loss={avg_loss:.4f}  lr={lr_now:.2e}")
                    running_loss = 0.0

        print(f"Epoch {epoch+1} complete.")

    # ── Save ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), SFT_WEIGHTS_OUT)
    tokenizer.save_pretrained(TOKENIZER_OUT)
    print(f"\nSFT weights saved to:  {SFT_WEIGHTS_OUT}")
    print(f"Tokenizer saved to:    {TOKENIZER_OUT}/")
    print("Done.")


if __name__ == "__main__":
    main()
