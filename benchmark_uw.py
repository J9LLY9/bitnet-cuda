import torch
import bitnet_cuda
import time

# --- HARDWARE CONSTANTS ---
# RTX 3050 Mobile/Desktop avg theoretical bandwidth (GB/s)
PEAK_BANDWIDTH_GBPS = 112.0  
LAYERS = 12 # From your architecture

def benchmark_function(name, func, *args, iters=100):
    # 1. Warmup (gets the GPU out of idle power state)
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()
    
    # 2. Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / iters
    return avg_time_ms

def run_benchmark(M, K, N, mode="Inference"):
    print(f"\n{'='*60}")
    print(f" BITNET-1.58B BENCHMARK: {mode.upper()} MODE")
    print(f" Matrix Shape: M={M} (Tokens), K={K} (In), N={N} (Out)")
    print(f"{'='*60}")

    # Generate dummy data
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
    
    # For custom kernel: Packed int8 weights (N, K/4)
    B_packed = torch.randint(-128, 127, (N, K // 4), dtype=torch.int8, device="cuda")

    # Define the functions
    def pytorch_baseline():
        return torch.nn.functional.linear(A, B_fp16)
    
    def custom_bitnet():
        return bitnet_cuda.bitnet_forward(A, B_packed, M, K, N)

    # --- 1. RUN TIMING ---
    time_pt = benchmark_function("PyTorch Baseline", pytorch_baseline)
    time_custom = benchmark_function("Custom BitNet", custom_bitnet)
    
    # --- 2. CALCULATE MEMORY TRAFFIC (Bytes moved) ---
    # PyTorch FP16: Read A, Read B, Write C (All float16 = 2 bytes)
    bytes_pt = (M * K * 2) + (K * N * 2) + (M * N * 2)
    
    # BitNet: Read A (2 bytes), Read B_packed (1 byte per 4 weights), Write C (2 bytes)
    bytes_custom = (M * K * 2) + ((K * N) / 4) + (M * N * 2)

    # --- 3. CALCULATE BANDWIDTH (GB/s) ---
    # GB/s = (Total Bytes / 1e9) / (Time in seconds)
    bw_pt = (bytes_pt / 1e9) / (time_pt / 1000)
    bw_custom = (bytes_custom / 1e9) / (time_custom / 1000)

    # --- 4. CALCULATE UTILIZATION (%) ---
    util_pt = (bw_pt / PEAK_BANDWIDTH_GBPS) * 100
    util_custom = (bw_custom / PEAK_BANDWIDTH_GBPS) * 100

    # --- 5. PROJECTED TOKENS PER SECOND (12-Layer Model) ---
    # For M=1 (Inference), how long does a full 12-layer pass take?
    if M == 1:
        tps_pt = 1000 / (time_pt * LAYERS)
        tps_custom = 1000 / (time_custom * LAYERS)
    else:
        tps_pt = 0
        tps_custom = 0

    # --- PRINT PROFESSIONAL REPORT ---
    print(f"{'Metric':<25} | {'PyTorch FP16':<15} | {'BitNet C++ Kernel':<15}")
    print("-" * 60)
    print(f"{'Execution Time (ms)':<25} | {time_pt:<15.3f} | {time_custom:<15.3f}")
    print(f"{'Speedup Factor':<25} | {'1.00x':<15} | {time_pt/time_custom:<15.2f}x")
    print("-" * 60)
    print(f"{'Memory Traffic (MB)':<25} | {bytes_pt/1e6:<15.2f} | {bytes_custom/1e6:<15.2f}")
    print(f"{'Achieved Bandwidth (GB/s)':<25} | {bw_pt:<15.2f} | {bw_custom:<15.2f}")
    print(f"{'Hardware Utilization (%)':<25} | {util_pt:<15.2f}% | {util_custom:<15.2f}%")
    
    if M == 1:
        print("-" * 60)
        print(f"{'Projected 12-Layer TPS':<25} | {tps_pt:<15.1f} | {tps_custom:<15.1f}")

if __name__ == "__main__":
    # Test 1: Inference Mode (Batch Size 1, generating tokens one by one)
    run_benchmark(M=1, K=4096, N=4096, mode="Inference (Generation)")
    
    # Test 2: Prefill/Training Mode (Batch Size 1024, reading a whole prompt)
    run_benchmark(M=1024, K=4096, N=4096, mode="Prefill (Context Processing)")