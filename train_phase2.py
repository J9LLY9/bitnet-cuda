"""
train_phase2.py — BitNet-1.58b  Phase 1 Validation + Phase 2 Overnight Grind

Usage:
    python train_phase2.py

Phase 1
-------
  • Loads bitnet_m1_recovery_01_step00500.safetensors into BitNet158 (12L 512d 8H).
  • Expands pos_embedding from [256, 512] → [1024, 512] by copying known rows and
    small-noise-initialising the new 768 positions.
  • Runs greedy generation on the entity-tracking probe.  Prints PASS / FAIL.
  • Aborts with a clear message if FAIL — does not start Phase 2.

Phase 2
-------
  • Full TinyStories train split tokenised once and pinned in RAM (≤ 4 GB for 80 GB host).
  • Pack-and-chunk at context_len=1024 → zero padding waste, maximum data density.
  • Tries micro_batch=32 (effective 128 via 4-step accumulation) on the RTX 3050.
    Auto-falls back to micro_batch=16 + 8 accumulation steps on OOM.
  • Gradient checkpointing on every BitBlock to stay well inside 8 GB VRAM.
  • CosineAnnealingLR: 1e-5 → 1e-7.
  • Saves bitnet_1.1_milestone_step_X.safetensors every 1 000 optimiser steps.
  • Exits cleanly once training loss EMA drops below TARGET_LOSS = 1.1.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint as grad_ckpt
from transformers import AutoTokenizer
from datasets import load_dataset

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
except ImportError:
    sys.exit("ERROR: run  pip install safetensors  then retry.")

from model import BitNet158, get_device

# ─── Paths ────────────────────────────────────────────────────────────────────
M1_CHECKPOINT   = "bitnet_m1_recovery_01_step00500.safetensors"
CKPT_PREFIX     = "bitnet_1.1_milestone_step"

# ─── Model spec ───────────────────────────────────────────────────────────────
VOCAB_SIZE   = 50_257   # GPT-2 base (no special tokens added for Phase 2)
EMBED_SIZE   = 512
NUM_HEADS    = 8
NUM_LAYERS   = 12
CONTEXT_LEN  = 1024

# ─── Training hypers ──────────────────────────────────────────────────────────
TARGET_LOSS  = 1.1
LR           = 2e-4
MIN_LR       = 1e-6
GRAD_CLIP    = 1.0
CKPT_EVERY   = 1_000   # optimiser steps
LOG_EVERY    = 10

# Warm-up: if the first-step loss lands above this, the recovered weights
# need a brief stabilisation ramp before the full 2e-4 LR kicks in.
WARMUP_TRIGGER_LOSS = 1.8   # activate if loss > this on step 1
WARMUP_STEPS        = 50    # linear ramp: LR/50 → LR over 50 steps

# Batching — Phase 2 will try FAST first, fall back to SAFE on OOM
FAST_MICRO   = 32;  FAST_ACCUM  = 4    # effective batch = 128
SAFE_MICRO   = 16;  SAFE_ACCUM  = 8    # effective batch = 128
VRAM_CEIL_GB = 7.5                      # RTX 3050 safety threshold

# ─── Probe ────────────────────────────────────────────────────────────────────
PROBE_PROMPT   = "Lily has a red ball and a blue toy. She gives the red ball to Tim. Now, Tim has the"
PROBE_KEYWORD  = "red ball"
PROBE_TOKENS   = 20     # generate this many tokens for the probe

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(1, sorted_idx, sorted_logits)
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int,
             temperature: float = 0.1, top_p: float = 0.95,
             device: str = "cuda") -> str:
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new):
        logits     = model(ids)
        next_l     = logits[:, -1, :] / max(temperature, 1e-6)
        next_l     = top_p_filter(next_l, top_p)
        probs      = F.softmax(next_l, dim=-1)
        next_tok   = torch.multinomial(probs, 1)
        ids        = torch.cat([ids, next_tok], dim=1)
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading  (handles pos_embedding expansion 256 → 1024)
# ══════════════════════════════════════════════════════════════════════════════

def load_m1_checkpoint(path: str, model: BitNet158, device: torch.device) -> None:
    print(f"Loading checkpoint: {path}")
    ckpt = st_load(path, device="cpu")

    # ── pos_embedding: expand 256 → CONTEXT_LEN ───────────────────────────
    old_pe = ckpt["pos_embedding.weight"]          # [256, 512]
    old_ctx, d = old_pe.shape
    if old_ctx < CONTEXT_LEN:
        new_pe = torch.zeros(CONTEXT_LEN, d)
        new_pe[:old_ctx] = old_pe                  # copy known positions
        # small noise for the new positions so they aren't identical at init
        nn.init.normal_(new_pe[old_ctx:], mean=0.0, std=0.01)
        ckpt["pos_embedding.weight"] = new_pe
        print(f"  pos_embedding expanded {old_ctx} → {CONTEXT_LEN} "
              f"(positions {old_ctx}–{CONTEXT_LEN-1} noise-initialised)")

    # ── strict load (all other keys match model.py exactly) ───────────────
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if unexpected:
        print(f"  WARNING — unexpected keys (ignored): {unexpected}")
    if missing:
        print(f"  WARNING — missing keys (random init kept): {missing}")

    total    = sum(p.numel() for p in model.parameters())
    loaded   = sum(ckpt[k].numel() for k in ckpt if k in dict(model.named_parameters()))
    print(f"  {loaded:,} / {total:,} parameters loaded from checkpoint "
          f"({100*loaded/total:.1f} %)")

    model.to(device)
    print(f"  Model on {next(model.parameters()).device}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Validation
# ══════════════════════════════════════════════════════════════════════════════

def phase1_validate(model, tokenizer, device) -> bool:
    print("\n" + "═" * 60)
    print("PHASE 1 — VALIDATION")
    print("═" * 60)
    print(f"Prompt : {PROBE_PROMPT!r}")

    output = generate(model, tokenizer, PROBE_PROMPT,
                      max_new=PROBE_TOKENS, device=str(device))
    completion = output[len(PROBE_PROMPT):]

    print(f"Output : {output!r}")
    print(f"Completion: {completion!r}")

    passed = PROBE_KEYWORD.lower() in completion.lower()
    if passed:
        print(f'\n  ✓  PASS — "{PROBE_KEYWORD}" found in completion.')
        print("  Proceeding to Phase 2.\n")
    else:
        print(f'\n  ✗  FAIL — "{PROBE_KEYWORD}" not found.')
        print("  The M1 checkpoint may not have learned entity tracking yet.")
        print("  Aborting. Retrain or use a later checkpoint.\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Dataset (RAM cache)
# ══════════════════════════════════════════════════════════════════════════════

def build_ram_cache(tokenizer) -> TensorDataset:
    """
    Loads the full TinyStories train split, tokenises every story,
    concatenates into one flat token stream, then chunks into
    (context_len + 1) windows for next-token prediction.

    With 80 GB RAM: the flat stream is ~450–550 M int32 tokens ≈ 2 GB.
    The chunked TensorDataset is another ~2 GB.  Well within budget.
    """
    print("Building RAM cache of full TinyStories train split…")
    t0 = time.time()

    raw = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(raw):,} stories loaded from HuggingFace  ({time.time()-t0:.0f}s)")

    # Tokenise in batches — fast, no padding needed (we concatenate everything)
    sep = [tokenizer.eos_token_id]   # story boundary marker

    def tok_batch(batch):
        out = []
        for text in batch["text"]:
            out.extend(tokenizer.encode(text, add_special_tokens=False))
            out.extend(sep)
        return {"ids": [out]}        # one big list per batch, returned as list

    t1 = time.time()
    print("  Tokenising…")
    all_ids: list[int] = []
    chunk_size = 10_000              # stories per tokenisation chunk
    for start in range(0, len(raw), chunk_size):
        end    = min(start + chunk_size, len(raw))
        texts  = raw[start:end]["text"]
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            all_ids.extend(sep)
        if (start // chunk_size) % 20 == 0:
            pct = 100 * end / len(raw)
            print(f"    {end:>7,} / {len(raw):,}  ({pct:.0f}%)  "
                  f"tokens so far: {len(all_ids):,}")

    print(f"  Tokenisation complete: {len(all_ids):,} tokens  "
          f"({time.time()-t1:.0f}s)")

    # Chunk into (CONTEXT_LEN + 1) windows; discard the last partial chunk
    win    = CONTEXT_LEN + 1
    n_seqs = len(all_ids) // win
    flat   = torch.tensor(all_ids[:n_seqs * win], dtype=torch.int32).view(n_seqs, win)

    inputs  = flat[:, :-1].long()   # (N, CONTEXT_LEN)
    targets = flat[:, 1:].long()    # (N, CONTEXT_LEN)  — next-token labels

    ds = TensorDataset(inputs, targets)
    gb = flat.numel() * 4 / 1024 ** 3
    print(f"  RAM cache: {n_seqs:,} sequences × {CONTEXT_LEN} tokens  "
          f"({gb:.2f} GB)  [{time.time()-t0:.0f}s total]\n")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Training
# ══════════════════════════════════════════════════════════════════════════════

class GradCkptBlock(nn.Module):
    """Thin wrapper so gradient checkpointing works without modifying model.py."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, mask):
        return grad_ckpt(self.block, x, mask, use_reentrant=False)


def wrap_grad_checkpointing(model: BitNet158) -> None:
    """Wrap every BitBlock with gradient checkpointing in-place."""
    for i, block in enumerate(model.blocks):
        model.blocks[i] = GradCkptBlock(block)
    print("  Gradient checkpointing enabled on all 12 blocks.")


def make_loader(ds: TensorDataset, micro_batch: int) -> DataLoader:
    # num_workers=0: data is already in RAM, no disk I/O to parallelise
    return DataLoader(ds, batch_size=micro_batch, shuffle=True,
                      num_workers=0, pin_memory=False, drop_last=True)


def save_checkpoint(model: BitNet158, step: int) -> None:
    path = f"{CKPT_PREFIX}_{step:06d}.safetensors"
    # safetensors requires contiguous float tensors
    sd = {k: v.contiguous().float() for k, v in model.state_dict().items()}
    st_save(sd, path)
    print(f"  Checkpoint saved → {path}")


def phase2_train(model: BitNet158, ds: TensorDataset, device: torch.device) -> None:
    print("═" * 60)
    print("PHASE 2 — OVERNIGHT 1.1 LOSS GRIND")
    print("═" * 60)

    # ── Gradient checkpointing (frees ~60 % of activation VRAM) ──────────
    wrap_grad_checkpointing(model)
    model.train()

    # ── Optimiser / scheduler ─────────────────────────────────────────────
    # Total steps unknown (run until TARGET_LOSS); T_max is a soft horizon
    T_MAX = 50_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # ── Try FAST batch config; OOM → fall back to SAFE ───────────────────
    micro_batch = FAST_MICRO
    accum_steps = FAST_ACCUM
    loader      = make_loader(ds, micro_batch)
    data_iter   = iter(loader)

    print(f"  Attempting micro_batch={micro_batch}, accum={accum_steps} "
          f"(effective batch 128)")
    print(f"  LR: {LR:.0e} → {MIN_LR:.0e}  (CosineAnnealing over {T_MAX:,} steps)  [grind LR for 1.3→1.1]")
    print(f"  Target loss: {TARGET_LOSS}")
    print(f"  Checkpoint every {CKPT_EVERY} steps\n")

    def next_batch():
        nonlocal data_iter, loader
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    opt_step      = 0
    accum_count   = 0
    accum_loss    = 0.0
    ema_loss      = None
    ema_alpha     = 0.98       # heavy smoothing for overnight display
    batch_adapted = False
    t_log         = time.time()

    # Warm-up state (activated lazily after step 1 if loss is high)
    in_warmup   = False
    warmup_step = 0

    optimizer.zero_grad()

    while True:
        x, y = next_batch()
        x, y = x.to(device), y.to(device)

        try:
            logits = model(x)                              # (B, T, V)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            (loss / accum_steps).backward()
            accum_loss  += loss.item()
            accum_count += 1

        except torch.cuda.OutOfMemoryError:
            if batch_adapted:
                sys.exit("FATAL: OOM even at SAFE batch config. Reduce context or layers.")
            print(f"\n  OOM at micro_batch={micro_batch}. "
                  f"Switching to micro_batch={SAFE_MICRO} / accum={SAFE_ACCUM}…")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            micro_batch   = SAFE_MICRO
            accum_steps   = SAFE_ACCUM
            loader        = make_loader(ds, micro_batch)
            data_iter     = iter(loader)
            accum_count   = 0
            accum_loss    = 0.0
            batch_adapted = True
            continue

        if accum_count < accum_steps:
            continue

        # ── Optimiser step ────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Warm-up gate: hold cosine scheduler back during the LR ramp.
        # Each warm-up step linearly increases LR from LR/WARMUP_STEPS → LR.
        # Scheduler is only advanced (and its internal counter incremented)
        # once warm-up is done, so the cosine curve starts from the right place.
        if in_warmup:
            warmup_step += 1
            warmup_lr = LR * warmup_step / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            if warmup_step >= WARMUP_STEPS:
                in_warmup = False
                for pg in optimizer.param_groups:
                    pg['lr'] = LR
                print(f"  [warm-up] complete at opt step {opt_step + 1} "
                      f"— cosine schedule takes over at {LR:.1e}")
        else:
            scheduler.step()

        optimizer.zero_grad()

        avg_loss    = accum_loss / accum_steps
        ema_loss    = avg_loss if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * avg_loss
        accum_loss  = 0.0
        accum_count = 0
        opt_step   += 1

        # ── Warm-up trigger: check loss on first completed step ───────────
        # If the recovered weights land far above 1.3, a sudden 2e-4 update
        # could shatter the ternary structure.  Ramp gently instead.
        if opt_step == 1 and avg_loss > WARMUP_TRIGGER_LOSS:
            in_warmup = True
            for pg in optimizer.param_groups:
                pg['lr'] = LR / WARMUP_STEPS   # near-zero start for ramp
            print(f"  [warm-up] loss {avg_loss:.4f} > {WARMUP_TRIGGER_LOSS:.1f} on step 1 "
                  f"— activating {WARMUP_STEPS}-step ramp "
                  f"({LR/WARMUP_STEPS:.1e} → {LR:.1e})")

        # ── Logging ───────────────────────────────────────────────────────
        # Read LR from param groups (correct during warm-up AND cosine phases)
        if opt_step % LOG_EVERY == 0:
            elapsed  = time.time() - t_log
            lr_now   = optimizer.param_groups[0]['lr']
            vram     = vram_gb()
            warmup_tag = f"  [warm-up {warmup_step}/{WARMUP_STEPS}]" if in_warmup else ""
            print(f"  step {opt_step:>6,}  loss {avg_loss:.4f}  "
                  f"ema {ema_loss:.4f}  lr {lr_now:.2e}  "
                  f"vram {vram:.2f} GB  ({elapsed:.1f}s){warmup_tag}")
            t_log = time.time()

        # ── Ternary weight health check (every 500 steps) ─────────────────
        # Verifies that BitLinear layers are using genuine ternary weights
        # (W ∈ {-1, 0, 1}) in the forward pass, not the latent float weights.
        # A healthy zero-fraction is 30–70 %; outside that range investigate.
        if opt_step % 500 == 0:
            with torch.no_grad():
                zero_fracs = []
                for m in model.modules():
                    if isinstance(m, nn.Linear) and m.weight is not None:
                        g = m.weight.abs().mean().clamp(min=1e-5)
                        q = (m.weight / g).round().clamp(-1, 1)
                        zero_fracs.append((q == 0).float().mean().item())
                if zero_fracs:
                    avg_zero = sum(zero_fracs) / len(zero_fracs)
                    status   = "OK" if 0.3 <= avg_zero <= 0.7 else "WARN — outside 30–70 % range"
                    print(f"  [ternary] zero-weight fraction: {avg_zero:.1%}  [{status}]")

        # ── Checkpoint ────────────────────────────────────────────────────
        if opt_step % CKPT_EVERY == 0:
            save_checkpoint(model, opt_step)

        # ── Target check ─────────────────────────────────────────────────
        if ema_loss is not None and ema_loss <= TARGET_LOSS:
            print(f"\n  TARGET REACHED — EMA loss {ema_loss:.4f} ≤ {TARGET_LOSS}")
            save_checkpoint(model, opt_step)
            print("  Phase 2 complete.")
            break


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BitNet Phase 1 → Phase 2 pipeline")
    parser.add_argument(
        "--force-phase2", action="store_true",
        help="Skip the Phase 1 probe gate and go straight to overnight training. "
             "Use this when loading an early checkpoint (e.g. step 500) that hasn't "
             "learned entity tracking yet but is a valid warm-start for Phase 2."
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")
    if device.type != "cuda":
        print("WARNING: CUDA not found — training will be very slow on CPU/MPS.")

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Build model ───────────────────────────────────────────────────────
    model = BitNet158(
        vocab_size  = VOCAB_SIZE,
        embed_size  = EMBED_SIZE,
        num_heads   = NUM_HEADS,
        num_layers  = NUM_LAYERS,
        max_seq_len = CONTEXT_LEN,
    )

    # ── Load M1 weights ───────────────────────────────────────────────────
    load_m1_checkpoint(M1_CHECKPOINT, model, device)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    passed = phase1_validate(model, tokenizer, device)
    if not passed:
        if args.force_phase2:
            print("  --force-phase2 set: overriding Phase 1 gate and continuing.\n")
        else:
            print("  Hint: if this is an early checkpoint and you still want to")
            print("  run Phase 2 as a warm-start, re-run with --force-phase2.\n")
            sys.exit(1)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    dataset = build_ram_cache(tokenizer)
    model.train()
    phase2_train(model, dataset, device)


if __name__ == "__main__":
    main()
