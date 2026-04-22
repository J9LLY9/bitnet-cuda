from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# 1. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # AI needs to know when a story ends

# 2. Download the data (Just 1% for now to keep it fast)
print("Downloading TinyStories...")
raw_data = load_dataset("roneneldan/TinyStories", split="train[:1%]")

# 3. Turn the words into numbers (Tokenization)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing data...")
tokenized_data = raw_data.map(tokenize_function, batched=True)

# 4. Convert to PyTorch format
tokenized_data.set_format(type="torch", columns=["input_ids"])
full_tensor = tokenized_data["input_ids"]

print(f"\nData Ready!")
print(f"Total Stories: {len(full_tensor)}")
print(f"Shape of the data: {full_tensor.shape}") # (Number of stories, 128 words each)