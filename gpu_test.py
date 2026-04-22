import torch

print("--- GPU CAPABILITY REPORT ---")
print(f"Total GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {p.name}")
    print(f"  VRAM: {p.total_memory / 1024**3:.2f} GB")
    print(f"  Compute Capability: {p.major}.{p.minor}")
    
    # Simple speed test: Multiply two big matrices on the GPU
    x = torch.randn(2000, 2000).to(f'cuda:{i}')
    y = torch.randn(2000, 2000).to(f'cuda:{i}')
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    z = torch.matmul(x, y)
    end.record()
    
    torch.cuda.synchronize()
    print(f"  Matrix Multiplication Speed: {start.elapsed_time(end):.2f} ms")

print("\n------------------------------")