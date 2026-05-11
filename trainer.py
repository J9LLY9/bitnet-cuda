import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
    SAFETENSORS_OK = True
except ImportError:
    SAFETENSORS_OK = False

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    NVML_OK = False

from model import BitNet158, get_device


def vram_used_gb() -> float:
    if NVML_OK:
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1024 ** 3
    return torch.cuda.memory_allocated() / 1024 ** 3


class GradCkptBlock(nn.Module):
    """Thin wrapper applied AFTER checkpoint loading so state-dict keys stay clean."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, mask):
        return grad_checkpoint(self.block, x, mask, use_reentrant=False)


# ---------------------------------------------------------------------------
# 1. ARGS  (parsed early so --resume_from is available before model init)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume_from", type=str, default="BitNet_TS_1.27_Final.safetensors",
                    help="Path to a .safetensors checkpoint to resume from")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 2. MODEL  — BitNet158 from model.py (RMSNorm + SubLN + GELU, no biases)
# ---------------------------------------------------------------------------
torch.cuda.empty_cache()
device    = get_device()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size  = tokenizer.vocab_size
MAX_SEQ_LEN = 256   # matches the M1 checkpoint's pos_embedding [256, 512]

model = BitNet158(
    vocab_size  = vocab_size,
    embed_size  = 512,
    num_heads   = 8,
    num_layers  = 12,
    max_seq_len = MAX_SEQ_LEN,
).to(device)

# ---------------------------------------------------------------------------
# 3. RESUME  — load before gradient-checkpointing wrapper so keys match 100 %
# ---------------------------------------------------------------------------
if args.resume_from:
    import re as _re
    if not SAFETENSORS_OK:
        raise RuntimeError("safetensors not installed — run: pip install safetensors")
    print(f"Resuming from: {args.resume_from}")
    ckpt = st_load(args.resume_from, device="cpu")

    # Checkpoints saved while GradCkptBlock was active have keys like
    # "blocks.N.block.<rest>".  The unwrapped model expects "blocks.N.<rest>".
    # Remap before loading so strict=True can verify a 100% match.
    remapped = {_re.sub(r'^(blocks\.\d+)\.block\.', r'\1.', k): v
                for k, v in ckpt.items()}
    if remapped != ckpt:
        print("  Remapped GradCkptBlock wrapper keys  "
              "(blocks.N.block.X → blocks.N.X)")
    ckpt = remapped

    # Strict load — crashes immediately with a clear message on any mismatch.
    try:
        model.load_state_dict(ckpt, strict=True)
    except RuntimeError as _e:
        raise RuntimeError(
            f"\n\nCheckpoint weight mismatch — {args.resume_from}:\n{_e}\n\n"
            "Fix: ensure the checkpoint was saved from the same BitNet158 "
            "architecture (vocab_size=50257, embed=512, heads=8, layers=12, "
            "max_seq_len=256)."
        ) from None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: 100% of parameters ({total_params:,}) — strict match confirmed")

# ---------------------------------------------------------------------------
# 4. GRADIENT CHECKPOINTING  — wrap AFTER loading (keeps key names clean)
# ---------------------------------------------------------------------------
for i, block in enumerate(model.blocks):
    model.blocks[i] = GradCkptBlock(block)
print(f"BitNet158 ready on {device} | grad-checkpointing ON | "
      f"params: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 5. HYPERPARAMETERS
# ---------------------------------------------------------------------------
TOTAL_STEPS        = 50_000
MICRO_BATCH        = 8      # forward-pass batch size
ACCUM_STEPS        = 16     # 8 × 16 = 128 effective batch
CKPT_EVERY         = 2_000  # weekend warrior: save every 2k steps
TARGET_LOSS        = 1.05   # stretch goal: saves BITNET_1.05_HERO_WEIGHTS.safetensors
DIVERGE_THRESHOLD  = 2.5    # emergency stop if loss spikes above this after step 100
VRAM_MONITOR_STEPS = 50
VRAM_LIMIT_GB      = 7.8
vram_adapted       = False

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                               betas=(0.9, 0.95))
criterion = nn.CrossEntropyLoss(reduction="mean")
scheduler = CosineAnnealingLR(optimizer, T_max=50_000, eta_min=1e-6)

# ---------------------------------------------------------------------------
# 6. DATA
# ---------------------------------------------------------------------------
def build_loader(batch_size: int):
    raw = load_dataset("roneneldan/TinyStories", split="train", keep_in_memory=True)
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=MAX_SEQ_LEN)
    tok = raw.map(tokenize, batched=True, num_proc=1, remove_columns=["text"])
    tok.set_format(type="torch", columns=["input_ids"])
    dl = DataLoader(tok, batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    def _infinite():
        while True:
            yield from dl
    return _infinite()

# ---------------------------------------------------------------------------
# 6b. LOG FILE  — opened in append + line-buffered mode so every line is
#     flushed to disk immediately; safe to tail -f overnight_grind_log.txt
# ---------------------------------------------------------------------------
LOG_PATH = "uw_application_grind.txt"
_log_file = open(LOG_PATH, "a", buffering=1)

import datetime as _dt
_log_file.write(f"\n{'='*60}\n"
                f"Run started: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n")

def log(msg: str) -> None:
    print(msg)
    _log_file.write(msg + "\n")

print("Loading and tokenising TinyStories (25%)…")
data_iter = build_loader(MICRO_BATCH)

# ---------------------------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------------------------
log(f"Starting grind — {TOTAL_STEPS} steps on {device}")
log(f"  Micro-batch  : {MICRO_BATCH}  (effective = {MICRO_BATCH * ACCUM_STEPS})")
log(f"  LR           : 1e-4 → 1e-6  (CosineAnnealing T_max=50,000 — Weekend Warrior)")
log(f"  Weight decay : 0.01")
log(f"  Ckpt every   : {CKPT_EVERY} steps  (.safetensors)")
log(f"  Target loss  : {TARGET_LOSS}  → saves BITNET_1.05_HERO_WEIGHTS.safetensors then exits")
log(f"  Diverge guard: loss > {DIVERGE_THRESHOLD} after step 100 → emergency stop")
log(f"  Log file     : {LOG_PATH}")

model.train()
optimizer.zero_grad()
optimizer_step    = 0
accumulation_loss = 0.0
accum_count       = 0

while optimizer_step < TOTAL_STEPS:
    batch     = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    inputs    = input_ids[:, :-1]
    targets   = input_ids[:, 1:]

    logits = model(inputs)
    loss   = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
    (loss / ACCUM_STEPS).backward()
    accumulation_loss += loss.item()
    accum_count       += 1

    if accum_count < ACCUM_STEPS:
        continue

    # ── Optimiser step ────────────────────────────────────────────────────
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    optimizer_step += 1
    accum_count     = 0

    # ── VRAM safety (silent unless threshold is breached) ─────────────────
    if optimizer_step <= VRAM_MONITOR_STEPS and not vram_adapted:
        gb = vram_used_gb()
        if gb > VRAM_LIMIT_GB:
            old_m, old_a = MICRO_BATCH, ACCUM_STEPS
            MICRO_BATCH  = 4
            ACCUM_STEPS  = 32        # 4 × 32 = 128 effective batch preserved
            vram_adapted = True
            del data_iter
            torch.cuda.empty_cache()
            data_iter         = build_loader(MICRO_BATCH)
            accumulation_loss = 0.0
            print(f"  [VRAM ALERT] {gb:.2f} GB > {VRAM_LIMIT_GB} GB — "
                  f"batch {old_m}→{MICRO_BATCH}, accum {old_a}→{ACCUM_STEPS}")

    # ── Per-step log (console + file) ─────────────────────────────────────
    avg_loss   = accumulation_loss / ACCUM_STEPS
    current_lr = optimizer.param_groups[0]['lr']
    gb         = vram_used_gb()
    log(f"Step: {optimizer_step} | Loss: {avg_loss:.4f} | "
        f"LR: {current_lr:.2e} | VRAM: {gb:.2f} GB")
    accumulation_loss = 0.0

    # ── Target loss auto-stop ─────────────────────────────────────────────
    if avg_loss <= TARGET_LOSS:
        log(f"\n  TARGET REACHED — Loss {avg_loss:.4f} ≤ {TARGET_LOSS} at step {optimizer_step}")
        sd = {k: v.contiguous().float() for k, v in model.state_dict().items()}
        st_save(sd, "BITNET_1.05_HERO_WEIGHTS.safetensors")
        log("  Saved: BITNET_1.05_HERO_WEIGHTS.safetensors")
        log("  Stopping training. The grind is over.")
        break

    # ── Divergence guard ──────────────────────────────────────────────────
    if optimizer_step > 100 and avg_loss > DIVERGE_THRESHOLD:
        log(f"\n  [DIVERGE GUARD] Loss {avg_loss:.4f} > {DIVERGE_THRESHOLD} at step {optimizer_step}.")
        log("  Weights may be diverging. Stopping to protect last checkpoint.")
        log("  Resume from the most recent checkpoint_stepXXXXX.safetensors.")
        break

    # ── Checkpoint (.safetensors) ─────────────────────────────────────────
    if optimizer_step % CKPT_EVERY == 0:
        if SAFETENSORS_OK:
            ckpt_path = f"checkpoint_step{optimizer_step:05d}.safetensors"
            sd = {k: v.contiguous().float() for k, v in model.state_dict().items()}
            st_save(sd, ckpt_path)
        else:
            ckpt_path = f"checkpoint_step{optimizer_step:05d}.pt"
            torch.save(model.state_dict(), ckpt_path)
        log(f"  Checkpoint saved: {ckpt_path}")

# ---------------------------------------------------------------------------
# 8. FINAL SAVE
# ---------------------------------------------------------------------------
log("Training complete!")
if SAFETENSORS_OK:
    sd = {k: v.contiguous().float() for k, v in model.state_dict().items()}
    st_save(sd, "bitnet_weights_final.safetensors")
    log("Final weights saved: bitnet_weights_final.safetensors")
else:
    torch.save(model.state_dict(), "bitnet_weights_final.pt")
    log("Final weights saved: bitnet_weights_final.pt")

_log_file.close()
