import torch  # <--- THIS IS THE MISSING PIECE

# 1. Create a massive "dummy" tensor 
giant_block = torch.randn(10000, 10000)

# 2. Move it to the first GPU (T1000)
giant_block = giant_block.to("cuda:0")

# 3. Check the memory
print(f"Memory allocated on GPU 0: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")