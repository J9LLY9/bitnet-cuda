import torch
from model import BitNet158

# 1. Setup
device = torch.device("cuda")
# Use small dimensions for quick verification
model = BitNet158(vocab_size=1000, embed_size=64, num_layers=1).to(device)
model.prepare_for_inference() # This forces the packing logic!
model.eval()

# 2. Dummy input
x = torch.randint(0, 1000, (1, 16)).to(device)

# 3. Execution
print("Running forward pass...")
with torch.no_grad():
    output = model(x)

print("Integration test complete.")
print("Output shape:", output.shape)
