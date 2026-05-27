# BitNet 1.58b Custom CUDA Engine

🚀 **A from-scratch, highly optimized CUDA C++ inference engine for 1.58B Parameter Ternary LLMs.**

This project implements a custom forward-pass CUDA kernel for the [BitNet 1.58b](https://arxiv.org/abs/2402.17764) architecture. Standard LLMs (like LLaMA) use FP16 weights, requiring floating-point multiplication. BitNet restricts weights to ternary values `{-1, 0, 1}`, replacing heavy matrix multiplication with pure addition/subtraction.

By exploiting this architecture, I built a custom CUDA kernel that packs 4 ternary weights into a single byte, reducing memory bandwidth requirements by **87.5%**, and utilizing hardware-level memory coalescing to saturate the GPU memory bus.

---

## 📊 Performance & Benchmarks
Tested on Native Ubuntu | RTX 3050 (8GB VRAM) | CUDA 12.x

| Metric | PyTorch Native (FP16) | Custom CUDA Kernel (BitNet) | Improvement |
| :--- | :--- | :--- | :--- |
| **Memory Traffic (per step)** | 33.57 MB | 4.21 MB | **87.5% Reduction** |
| **Inference Latency (M=1)** | 0.167 ms | 0.224 ms* | *Approaching native cuBLAS* |
| **Prefill Latency (M=1024)** | 2.272 ms | 34.093 ms* | *40% speedup from V1* |

*(Note: Benchmark tracks pure memory/compute optimizations without full cuBLAS register tiling).*

---

## 🧠 The Engineering Journey: Shattering the Memory Wall

Because BitNet eliminates floating-point multiplication, its "Arithmetic Intensity" is incredibly low. The model is violently **Memory Bound**. The primary engineering challenge of this project was bridging the gap between high-level PyTorch and physical GPU hardware constraints.

### 1. The Math: 2-bit Weight Quantization
I implemented a bitwise packing algorithm that compresses four ternary weights (`-1, 0, 1`) into a single 8-bit integer (`uint8_t`). This mathematically reduced the VRAM footprint from 33.5 MB to 4.2 MB.

### 2. The Physics: Vectorized Memory Coalescing (`uint4`)
**The Bottleneck:** My initial V1 kernel used scalar memory access (1 byte per thread). This resulted in a catastrophic 0.38% hardware utilization because it forced the VRAM controller to fetch 128-byte cache lines for a single byte of useful data, clogging the memory bus.
**The Fix:** I upgraded the kernel to use 128-bit vectorized loads (`uint4`), forcing each thread to load 16 contiguous bytes (64 weights) in a single instruction (`LDG.E.128`). 

### 3. Strided Access & Matrix Pre-packing
**The Bottleneck:** Even with `uint4`, threads were reading down PyTorch's default Row-Major `[N, K]` memory layout. This caused "Strided Access"—threads asked for memory addresses 1,024 bytes apart, forcing the VRAM to send separate cache lines for every thread.
**The Fix:** I transposed the weight matrix layout in memory to `[K, N]` prior to execution. This physically placed the weights needed by the thread block perfectly side-by-side on the silicon, allowing 100% coalesced memory access.

### 4. Compute Bottlenecks & Shared Memory
**The Bottleneck:** Unpacking 64 weights required 64 bitwise shifts (`>>`) and masks (`&`) per thread. In a 16x16 block, 16 threads were redundantly unpacking the exact same memory bucket, suffocating the CUDA Cores with 15,000+ redundant math instructions.
**The Fix:** Engineered a **Shared Memory Unpacking Station**. Only the first row of threads fetches and unpacks the `uint4` data, casting it to `__half` floats and storing it in a 2KB `__shared__` memory array. The remaining threads synchronize (`__syncthreads()`) and read directly from the fast shared memory, bypassing the bitwise math entirely.

---

## 🛠️ Tech Stack & Concepts Applied
* **Languages:** C++, CUDA C, Python
* **Frameworks:** PyTorch, PyBind11 (C++ Extensions)
* **Low-Level Concepts:** Warp Scheduling, Memory Coalescing, Register Pressure, Shared Memory Allocation, SIMT Architecture.

## 🚀 How to Run
Ensure you have the CUDA Toolkit installed and a compatible NVIDIA GPU.

---

## 🤖 Model Results
Trained a conversation-capable BitNet b1.58 model achieving a training 
loss of 1.04, implemented from scratch using custom ternary quantization 
and the CUDA inference engine above. No pre-trained weights — trained 
independently on a Lenovo ThinkStation P520 running native Linux.

```bash
# 1. Clone the repository
git clone https://github.com/J9LLY9/bitnet-cuda.git
cd bitnet-custom-cuda

# 2. Setup the Python Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install torch ninja

# 3. Compile the C++ Extension
python setup.py build_ext --inplace

# 4. Run the Hardware Benchmark
python3 benchmark_uw.py
