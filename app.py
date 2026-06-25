import time
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer
from model import BitNet158, get_device

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    NVML_OK = False


# ---------------------------------------------------------------------------
# Model loading — prefers SFT weights/tokenizer when available
# ---------------------------------------------------------------------------
import os
import re
import glob

device = str(get_device())

SFT_TOKENIZER_DIR = "sft_tokenizer"
SFT_WEIGHTS       = "bitnet_sft.pt"
BASE_WEIGHTS      = "bitnet_weights.pt"
SAFETENSORS_CANDIDATES = [
    "BitNet_UW_Final_Gold_1.04.safetensors",
    "BITNET_1.05_HERO_WEIGHTS.safetensors",
    "bitnet_weights_final.safetensors",
]

def _find_weights():
    """Return (path, is_safetensors) for the best available checkpoint."""
    if os.path.isdir(SFT_TOKENIZER_DIR) and os.path.isfile(SFT_WEIGHTS):
        return SFT_WEIGHTS, False
    if os.path.isfile(BASE_WEIGHTS):
        return BASE_WEIGHTS, False
    for name in SAFETENSORS_CANDIDATES:
        if os.path.isfile(name):
            return name, True
    ckpts = sorted(glob.glob("checkpoint_step*.safetensors"))
    if ckpts:
        return ckpts[-1], True
    return None, False

def _load_weights(model, path, is_safetensors, device):
    if is_safetensors:
        from safetensors.torch import load_file as st_load
        ckpt = st_load(path, device=device)
        ckpt = {re.sub(r'^(blocks\.\d+)\.block\.', r'\1.', k): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
    else:
        model.load_state_dict(torch.load(path, map_location=device))

if os.path.isdir(SFT_TOKENIZER_DIR):
    tokenizer   = AutoTokenizer.from_pretrained(SFT_TOKENIZER_DIR)
    model_label = "SFT · Ternary 1.58-bit · 12L · 512d"
    print(f"SFT tokenizer loaded from {SFT_TOKENIZER_DIR}/")
else:
    tokenizer   = AutoTokenizer.from_pretrained("gpt2")
    model_label = "Pre-trained · Ternary 1.58-bit · 12L · 512d"

tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)   # includes any added special tokens

model = BitNet158(vocab_size, embed_size=512, num_heads=8, num_layers=12, max_seq_len=256).to(device)

weights_path, is_st = _find_weights()
if weights_path is None:
    print("WARNING: no weights found — running with random weights.")
    model_label = "RANDOM INIT — no weights found"
else:
    try:
        _load_weights(model, weights_path, is_st, device)
        model.prepare_for_inference()
        print(f"Weights loaded from {weights_path}")
    except RuntimeError as e:
        raise RuntimeError(
            f"Architecture mismatch loading {weights_path} — "
            "check that num_layers/embed_size match the checkpoint.\n"
            f"Original error: {e}"
        )
model.eval()


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def apply_repetition_penalty(logits, input_ids, penalty):
    seen = input_ids[0].unique()
    score = logits[0, seen]
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[0, seen] = score
    return logits


def top_p_filter(logits, top_p):
    """Nucleus (Top-P) filter: zero out tokens outside the top-p probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right so the token that pushes over top_p is kept
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------
def generate_stream(prompt, max_new_tokens, temperature, top_p, repetition_penalty):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        logits, past_key_values = model(input_ids, use_cache=True)
        next_logits = logits[:, -1, :]

        for i in range(int(max_new_tokens)):
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

            elapsed = time.perf_counter() - t0
            tps     = (i + 1) / elapsed if elapsed > 0 else 0.0
            output  = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            yield output, tps


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def get_telemetry(tps=0.0):
    if NVML_OK:
        mem_info   = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        temp_c     = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        vram_used  = mem_info.used  / 1024**3
        vram_total = mem_info.total / 1024**3
    else:
        vram_used, vram_total, temp_c = 0.0, 8.0, 0

    vram_pct = vram_used / vram_total

    if temp_c <= 70:
        temp_str = f"{temp_c}°C  COOL"
    elif temp_c <= 80:
        temp_str = f"{temp_c}°C  STABLE"
    else:
        temp_str = f"{temp_c}°C  HOT"

    return (
        vram_pct,
        f"{vram_used:.2f} / {vram_total:.1f} GB  ({vram_pct*100:.0f}%)",
        temp_str,
        f"{tps:.1f} tok/s",
    )


# ---------------------------------------------------------------------------
# Terminal CSS
# ---------------------------------------------------------------------------
TERMINAL_CSS = """
/* ── Page background ─────────────────────────────────── */
body, .gradio-container {
    background-color: #0a0a0a !important;
    color: #00ff41 !important;
    font-family: 'Courier New', Courier, monospace !important;
}

/* ── All text labels and markdown ────────────────────── */
label, .label-wrap span, p, h1, h2, h3, h4, .markdown {
    color: #00cc33 !important;
    font-family: 'Courier New', Courier, monospace !important;
}

h1 { color: #00ff41 !important; text-shadow: 0 0 10px #00ff41; }

/* ── Textboxes (output + prompt) ─────────────────────── */
textarea, input[type="text"] {
    background-color: #0d0d0d !important;
    color: #00ff41 !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 13px !important;
    border: 1px solid #00aa22 !important;
    border-radius: 2px !important;
    caret-color: #00ff41;
}
textarea:focus, input[type="text"]:focus {
    border-color: #00ff41 !important;
    box-shadow: 0 0 6px #00ff4155 !important;
    outline: none !important;
}

/* ── Sliders ─────────────────────────────────────────── */
input[type="range"] {
    accent-color: #00ff41;
}
.wrap.svelte-h6n5h6, .range-slider {
    background: #111 !important;
}

/* ── Buttons ─────────────────────────────────────────── */
button {
    background-color: #001a00 !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
    border-radius: 2px !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.05em;
    transition: box-shadow 0.15s ease;
}
button:hover {
    box-shadow: 0 0 8px #00ff41 !important;
    background-color: #003300 !important;
}
button.primary {
    border-color: #00ff41 !important;
    box-shadow: 0 0 4px #00ff4166;
}

/* ── Sidebar stat boxes ──────────────────────────────── */
.stat-box {
    background: #0d0d0d;
    border: 1px solid #00aa22;
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 2px;
}

/* ── Panel / column borders ──────────────────────────── */
.block {
    background-color: #0a0a0a !important;
    border-color: #003300 !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="BitNet // Terminal", css=TERMINAL_CSS) as demo:
    tps_state = gr.State(0.0)

    gr.Markdown("# > BITNET_1.58b // INFERENCE TERMINAL")
    gr.Markdown("`12-layer ternary transformer — TinyStories corpus — T1000 GPU`")

    with gr.Row():

        # ── Left: output + controls ────────────────────────────────────────
        with gr.Column(scale=3):
            output_box = gr.Textbox(
                label="> OUTPUT",
                lines=18,
                interactive=False,
                placeholder="[ awaiting prompt... ]",
                elem_classes=["terminal-output"],
            )
            prompt_box = gr.Textbox(
                label="> PROMPT",
                placeholder="Once upon a time...",
                lines=2,
            )

            with gr.Row():
                temperature_slider = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.05,
                    label="Temperature  (creativity)",
                )
                top_p_slider = gr.Slider(
                    0.5, 1.0, value=0.92, step=0.01,
                    label="Top-P  (nucleus mass)",
                )

            with gr.Row():
                max_tokens_slider = gr.Slider(
                    10, 300, value=80, step=10,
                    label="Max New Tokens",
                )
                rep_penalty_slider = gr.Slider(
                    1.0, 2.0, value=1.3, step=0.05,
                    label="Repetition Penalty",
                )

            with gr.Row():
                generate_btn = gr.Button("> GENERATE", variant="primary")
                clear_btn    = gr.Button("CLEAR")

        # ── Right: telemetry sidebar ───────────────────────────────────────
        with gr.Column(scale=1, min_width=230):
            gr.Markdown("### // SYSTEM TELEMETRY")

            gr.Markdown("**VRAM**")
            vram_bar   = gr.Slider(minimum=0, maximum=1, value=0,
                                   interactive=False, label="Utilisation")
            vram_label = gr.Textbox(value="—", label="Used / Total",
                                    interactive=False, lines=1)

            gr.Markdown("**GPU TEMP**")
            temp_box = gr.Textbox(value="—", label="Temperature",
                                  interactive=False, lines=1)

            gr.Markdown("**THROUGHPUT**")
            tps_box = gr.Textbox(value="—", label="Tokens / sec",
                                 interactive=False, lines=1)

            gr.Markdown("**MODEL**")
            gr.Textbox(value=model_label,
                       label="Config", interactive=False, lines=1)

            refresh_btn = gr.Button("REFRESH TELEMETRY")

    # ── Wiring ─────────────────────────────────────────────────────────────
    def on_generate(prompt, max_tok, temp, top_p, rep_pen):
        """Streaming generator: yields (output_text, tps, vram_bar, vram_str, temp_str, tps_str)."""
        last_tps = 0.0
        for text, tps in generate_stream(prompt, max_tok, temp, top_p, rep_pen):
            last_tps = tps
            vram_val, vram_str, temp_str, tps_str = get_telemetry(tps)
            yield text, last_tps, vram_val, vram_str, temp_str, tps_str

    def on_refresh(tps):
        vram_val, vram_str, temp_str, tps_str = get_telemetry(tps)
        return vram_val, vram_str, temp_str, tps_str

    generate_btn.click(
        fn=on_generate,
        inputs=[prompt_box, max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider],
        outputs=[output_box, tps_state, vram_bar, vram_label, temp_box, tps_box],
    )
    clear_btn.click(
        fn=lambda: ("", 0.0),
        outputs=[output_box, tps_state],
    )
    refresh_btn.click(
        fn=on_refresh,
        inputs=[tps_state],
        outputs=[vram_bar, vram_label, temp_box, tps_box],
    )

if __name__ == "__main__":
    demo.launch()
