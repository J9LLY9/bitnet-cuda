from transformers import AutoTokenizer

# We use the GPT-2 tokenizer (it's the industry standard for learning)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The boy went to the store to buy a toy."
tokens = tokenizer.encode(text)

print(f"Original Text: {text}")
print(f"Tokens (Numbers): {tokens}")
print(f"Decoded back: {tokenizer.decode(tokens)}")

# Check your GPU memory usage while you're at it!
import torch
if torch.cuda.is_available():
    print(f"\nMemory allocated on GPU 0: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")