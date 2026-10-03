# Chapter 1.4: Multi-Array Strided Streaming & Memory Bandwidth Bound Ops

In modern machine learning workloads, element-wise vector operations (such as residual addition `Residual + LayerOutput`, SwiGLU / GeLU activations, LayerNorm, and SAXPY `Y = alpha * X + Y`) stream multiple arrays simultaneously through the memory bus.

Understanding how multiple independent pointers interact with the cache hierarchy, memory controllers, and the GPU Roofline model is fundamental to optimizing bandwidth-bound CUDA kernels.

---

## 1. Multi-Stream Traversal & Pointer Pacing

When an operation consumes elements from two or more input buffers (`A` and `B`) and writes to an output buffer (`C`), the memory controller must service simultaneous memory access streams over the physical memory bus.

### The SAXPY Kernel Pattern (Single-Precision A*X + Y)
```cpp
void saxpy_strided(size_t n, float alpha, 
                   const float* __restrict__ x, size_t stride_x,
                   float* __restrict__ y, size_t stride_y) {
    for (size_t i = 0; i < n; ++i) {
        y[i * stride_y] = alpha * x[i * stride_x] + y[i * stride_y];
    }
}
```

### The Hardware Reality of `__restrict__` & The Read-Only Cache
In standard C++, the compiler must conservatively assume **pointer aliasing**: that `x` and `y` might point to overlapping regions in memory.
* **Without `__restrict__`**: Every write to `y[i]` forces the compiler to invalidate its register cache of `x`, assuming that writing to `y` might have overwritten upcoming values of `x`. It must reload `x[i+1]` from memory on every iteration!
* **With `__restrict__`**: You give a hardware contract that `x` and `y` never overlap.
* **GPU Hardware Consequence**: On NVIDIA GPUs, adding `const float* __restrict__` tells the `nvcc` compiler to emit `LDG.E` instructions. These route reads through the **dedicated Read-Only Data Cache** (formerly the texture cache), bypassing L1 write-traffic and freeing up cache bandwidth!

---

## 2. Arithmetic Intensity & The Roofline Model

Every computer system has two fundamental physical limits:
1. **Peak Compute Throughput**: The maximum number of floating-point operations the ALUs/Tensor Cores can execute per second (`TFLOPS`).
2. **Peak Memory Bandwidth**: The maximum rate at which the memory bus can move data from DRAM/HBM into the compute units (`TB/s`).

### Arithmetic Intensity Formula
Arithmetic Intensity ($I$) is defined as the ratio of work (FLOPs) to memory movement (Bytes):
```formula
Arithmetic_Intensity = FLOPs_Executed / Bytes_Transferred
```

### Case Study: Arithmetic Intensity of SAXPY
For each element in `y[i] = alpha * x[i] + y[i]`:
* **FLOPs**: 1 multiply + 1 addition = **2 FLOPs**.
* **Bytes**: Load `x[i]` (4 bytes) + Load `y[i]` (4 bytes) + Store `y[i]` (4 bytes) = **12 Bytes**.
* **Intensity**: `2 FLOPs / 12 Bytes = 0.167 FLOPs / Byte`.

```diagram:saxpy
{
  "title": "Dual-Stream Vector Traversal: SAXPY",
  "subtitle": "Concurrent streaming from read-only x[ ] and read-write y[ ] over the memory bus. 2 FLOPs per 12 bytes."
}
```

### The GPU Hardware Reality (NVIDIA H100 GPU):
* **H100 SXM Peak FP32 Compute**: ~67 TFLOPS (67,000 GFLOPS)
* **H100 SXM HBM3 Bandwidth**: ~3.35 TB/s (3,350 GB/s)
* **The Machine Balance Ridge Point**: `67,000 / 3,350 = 20.0 FLOPs / Byte`.

```text
         Attainable Performance (TFLOPS)
                     ^
       Peak Compute  |--------------------------------- (Compute-Bound Ceiling)
          (67 TF)    |                                /
                     |                              /
                     |                            /
                     |                          /
                     |                        /
                     |                      /
       SAXPY (0.167) | *                  /
                     |   (Memory-Bound) /
                     +---------------------------------------->
                                    20.0 (Ridge Point)   Arithmetic Intensity
```

* **Conclusion**: Any algorithm with Arithmetic Intensity $< 20.0$ is strictly **Memory Bandwidth Bound**.
* For SAXPY ($I = 0.167$), the maximum theoretical performance on an H100 is:
```formula
Attainable_GFLOPS = 3,350 GB/s * 0.167 FLOPs/Byte = 559 GFLOPS
```
* The GPU's massive 67 TFLOPS arithmetic ALUs are operating at less than **1% of their capacity**! They are starved for data.

---

## 3. Vectorized Memory Access: `float4` (128-Bit Loads)

When elements are contiguous (`stride = 1`), requesting data in 32-bit scalar chunks (`LDR.32`) chokes the instruction pipeline and issues too many discrete memory transactions.

NVIDIA GPUs have a 128-bit memory datapath per thread. A single thread can load or store **16 contiguous bytes (4 single-precision floats)** in a single clock cycle using the `LDG.E.128` assembly instruction.

### Production CUDA Vectorized Kernel Pattern:
```cpp
__global__ void saxpy_vectorized_kernel(int n, float alpha, 
                                        const float* __restrict__ x, 
                                        float* __restrict__ y) {
    // Each thread processes 4 elements simultaneously:
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Vectorized path (128-bit loads):
    if (idx + 3 < n) {
        // Cast raw pointers to 128-bit float4 vector pointers:
        float4 vx = *reinterpret_cast<const float4*>(&x[idx]);
        float4 vy = *reinterpret_cast<const float4*>(&y[idx]);

        // Fused Multiply-Add (FMA) executed across vector registers:
        vy.x = alpha * vx.x + vy.x;
        vy.y = alpha * vx.y + vy.y;
        vy.z = alpha * vx.z + vy.z;
        vy.w = alpha * vx.w + vy.w;

        // Single 128-bit memory store instruction:
        *reinterpret_cast<float4*>(&y[idx]) = vy;
    }
    // Scalar tail cleanup for arbitrary N not divisible by 4:
    else {
        for (int i = idx; i < n; ++i) {
            y[i] = alpha * x[i] + y[i];
        }
    }
}
```

### The Alignment Golden Rule for `float4`
To issue a `float4` (128-bit) load without crashing:
* The starting pointer address MUST be **16-byte aligned** (`addr % 16 == 0`).
* If an unaligned address is cast to `float4*`, modern GPUs trigger a hardware trap: `CUDA_ERROR_ILLEGAL_ADDRESS`.

---

## 4. The Ultimate Remedy: Operator Fusion

Because memory-bound kernels are throttled by memory bus roundtrips, the single most impactful optimization in modern deep learning (used extensively in PyTorch 2.0 Inductor, TensorRT, and FlashAttention) is **Operator Fusion**.

### The Naive Unfused Pattern:
Consider a transformer MLP block: `Y = GeLU(X + Residual)`.
In naive frameworks, this executes as two separate CUDA kernels:
1. **Kernel 1 (`Add`)**: Reads `X` (4B) + Reads `Residual` (4B) -> Writes `Temp` to DRAM (4B).
2. **Kernel 2 (`GeLU`)**: Reads `Temp` from DRAM (4B) -> Writes `Y` to DRAM (4B).
* **Total DRAM Traffic**: 16 bytes per element.
* **Latency**: 2 separate kernel launch overheads + 1 roundtrip to global memory.

### The Fused Pattern:
Fuse both operations into a single kernel:
```cpp
__global__ void fused_add_gelu_kernel(int n, const float* __restrict__ x, 
                                      const float* __restrict__ residual, 
                                      float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Read directly from inputs:
        float val = x[i] + residual[i];
        
        // Compute activation entirely within fast thread registers (NO DRAM write!):
        float gelu_val = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
        
        // Write only final result:
        y[i] = gelu_val;
    }
}
```
* **Total DRAM Traffic**: Read `x` (4B) + Read `residual` (4B) + Write `y` (4B) = **12 bytes per element**.
* **Eliminated**: Completely eliminates writing and re-reading the intermediate `Temp` buffer from DRAM!
* **Speedup**: Delivers an immediate **1.5x to 2x end-to-end speedup** for free.

---

## 5. Summary & Best Practices

1. **Roofline Awareness**: Element-wise kernels have low arithmetic intensity ($I < 1.0$) and are strictly bound by memory bandwidth.
2. **Mark Pointers `__restrict__`**: Informs the hardware that memory streams do not alias, enabling the compiler to route reads through the high-speed Read-Only Cache (`LDG.E`).
3. **128-Bit Vectorization (`float4`)**: Consolidates four 32-bit loads into a single 128-bit bus request, cutting instruction issuance overhead by 75%.
4. **Enforce 16-Byte Alignment**: Ensure base buffers passed to vectorized kernels are 16-byte aligned to avoid illegal address crashes.
5. **Fuse Operators Aggressively**: Keep intermediate results in GPU registers rather than writing them back to global DRAM.

