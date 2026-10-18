# Chapter 1.4: Multi-Array Strided Streaming & Memory Bandwidth Bound Ops

In modern machine learning workloads, element-wise vector operations (such as residual addition `Residual + LayerOutput`, SwiGLU / GeLU activations, RMSNorm / LayerNorm, and SAXPY `Y = alpha * X + Y`) stream multiple arrays simultaneously across the memory bus.

Understanding how multiple independent pointers interact with physical cache lines, memory controllers, pointer aliasing, and the GPU Roofline model is the foundation of high-performance systems engineering.

---

## 1. The Multi-Stream Memory Wall & The Tollbooth Bridge

### The Physical Mental Model: The Factory and the Bridge

Imagine a massive car manufacturing factory:
* **The Mechanics (ALUs / CUDA Cores)**: Inside the factory floor, you have 1,000 lightning-fast mechanics. Each mechanic can install an engine in 1 second.
* **The Bridge (The Memory Bus)**: Connecting the parts warehouse (DRAM / GDDR7 VRAM) to the factory floor is a narrow, single-lane bridge that can only move 1 car every minute.

What happens to your mechanics?
They spend 99% of their day sitting on the floor drinking coffee, waiting for the bridge to deliver parts. The factory is not limited by how fast the mechanics can wrench; it is physically **choked by the bridge**.

This is the **Memory Wall**.

```diagram:saxpy
{
  "title": "Dual-Stream Vector Traversal: SAXPY",
  "subtitle": "Concurrent streaming from read-only x[ ] and read-write y[ ] over the memory bus. 2 FLOPs per 12 bytes."
}
```

### The SAXPY Kernel Pattern: `Single-Precision A * X Plus Y`

In linear algebra (BLAS Level 1) and deep learning residual layers, **SAXPY** represents the fundamental element-wise memory streaming loop:

```formula
Y[i] = alpha * X[i] + Y[i]
```

In sequential C++, we write this multi-array traversal as:

```cpp
void saxpy_strided(size_t n, float alpha, 
                   const float* __restrict__ x, size_t stride_x,
                   float* __restrict__ y, size_t stride_y) {
    for (size_t i = 0; i < n; ++i) {
        y[i * stride_y] = alpha * x[i * stride_x] + y[i * stride_y];
    }
}
```

Let's trace the physical journey of a single element `i`:
1. **Read `X[i]`**: 4 bytes loaded across the bus from memory into a CPU/GPU register.
2. **Read `Y[i]`**: 4 bytes loaded across the bus from memory into a second register.
3. **Compute `alpha * X[i] + Y[i]`**: The ALU executes 1 multiplication and 1 addition (2 FLOPs).
4. **Write `Y[i]`**: 4 bytes streamed back across the bus to memory.

| Resource Stream | Direction | Data Type | Physical Wire Transfer |
| :--- | :--- | :--- | :--- |
| **Input Stream `X`** | Read (Memory -> Core) | `float` | 4 Bytes |
| **Input Stream `Y`** | Read (Memory -> Core) | `float` | 4 Bytes |
| **Output Stream `Y`** | Write (Core -> Memory) | `float` | 4 Bytes |
| **Total Bus Traffic** | Bidirectional | 3 floats | **12 Bytes per element** |

---

## 2. Arithmetic Intensity & The Roofline Model

Every computing machine on earth—from a Raspberry Pi to an NVIDIA RTX 5090 or H100 GPU—is constrained by two hard physical speed limits:

1. **Peak Compute Ceiling**: The absolute maximum number of math operations the arithmetic logic units (ALUs) or Tensor Cores can execute per second (`TFLOPS`).
2. **Peak Memory Bandwidth**: The absolute maximum rate at which the memory controllers and bus can pump bytes between DRAM/VRAM and the processing chip (`GB/s` or `TB/s`).

### Arithmetic Intensity Formula

**Arithmetic Intensity** measures how much computational work is performed per byte of memory moved:

```formula
Arithmetic_Intensity = FLOPs_Executed / Bytes_Transferred
```

For SAXPY:
* **Work Executed**: 1 multiply + 1 add = **2 FLOPs**.
* **Bytes Transferred**: Load `X[i]` (4B) + Load `Y[i]` (4B) + Store `Y[i]` (4B) = **12 Bytes**.

```formula
Arithmetic_Intensity(SAXPY) = 2 FLOPs / 12 Bytes = 0.167 FLOPs / Byte
```

This means for every **1 math operation**, the hardware must haul **6 full bytes** across the memory bus!

---

### Machine Balance: What Does the Hardware Demand?

The **Machine Balance** (also called the **Roofline Ridge Point**) represents the exact ratio of compute to memory speed built into the silicon. It defines how many FLOPs an algorithm must perform per byte to keep all arithmetic cores 100% saturated.

```formula
Machine_Balance = Peak_Compute_GFLOPS / Peak_Bandwidth_GBps
```

Let's calculate the Machine Balance for real modern hardware:

#### 1. NVIDIA GeForce RTX 5090 (Blackwell Flagship GPU)
* **Peak FP32 Compute**: ~108 TFLOPS = 108,000 GFLOPS
* **Peak GDDR7 Bandwidth**: ~1,800 GB/s (1.8 TB/s)

```formula
Machine_Balance(RTX 5090) = 108,000 GFLOPS / 1,800 GB/s = 60.0 FLOPs / Byte
```

Your RTX 5090 demands **60 math operations for every 1 byte** brought from VRAM to keep its 21,760 CUDA cores fully occupied!

Now compare what the 5090 demands vs what SAXPY delivers:
* **Hardware Demands**: 60.0 FLOPs / Byte
* **SAXPY Delivers**: 0.167 FLOPs / Byte

```formula
Compute_Utilization = 0.167 / 60.0 = 0.00278 (0.28%)
```

> [!WARNING]
> When executing SAXPY on an RTX 5090, over **99.7% of the CUDA cores are completely idle**, stalled waiting for the GDDR7 memory bus. Adding more compute units or overclocking the core clock will yield **zero speedup**. The only way to run faster is to saturate memory bandwidth!

#### 2. AMD Ryzen 9 9950X (Zen 5 CPU) with Dual-Channel DDR5-6000
* **Peak FP32 Compute**: ~3,500 GFLOPS (3.5 TFLOPS via dual 512-bit FMA units)
* **Peak DDR5 Bandwidth**: ~96 GB/s (Dual-channel 128-bit bus @ 6000 MT/s)

```formula
Machine_Balance(Ryzen 9950X) = 3,500 GFLOPS / 96 GB/s = 36.5 FLOPs / Byte
```

Even on a 16-core flagship CPU, SAXPY's intensity of `0.167` sits far below the machine balance of `36.5`. Both CPU and GPU are hard-locked in the **Memory Bandwidth Bound** regime.

---

### The Roofline Model Diagram

The Roofline chart visually maps every algorithm to its physical hardware bottleneck:

```diagram:roofline
{
  "title": "Hardware Roofline Model: Memory Bound vs Compute Bound",
  "subtitle": "Interactive plot mapping arithmetic intensity against physical hardware ceilings on RTX 5090 and Ryzen 9 9950X."
}
```

* **Left of the Ridge Point (`I < Machine_Balance`)**: **Memory Bandwidth Bound**. Performance scales strictly with memory bus throughput. Compute cores are starved.
* **Right of the Ridge Point (`I > Machine_Balance`)**: **Compute Bound**. Matrix multiplications (GEMM) with high arithmetic reuse live here, saturating the ALUs.

---

## 3. The Paranoid Compiler, Pointer Aliasing & `__restrict__`

Why does naive C++ code often run at only 30% to 50% of peak memory bandwidth? 
The number one silent performance killer in systems code is **Pointer Aliasing**.

### The Physical Analogy: The Paranoid Courier

Imagine you give a courier two clipboards:
* **Clipboard X** (read-only input sheet)
* **Clipboard Y** (output sheet to update)

You tell the courier: *"For each row `i`, read `X[i]`, multiply by 2, and write the result into `Y[i]`."*

In standard C++, the compiler is **extremely paranoid**. It reasons:
> *"Wait a moment... What if the caller passed pointers where `X` and `Y` overlap in memory? If I write to `Y[0]`, how do I know I didn't just overwrite `X[1]` or `X[2]`?"*

Because the compiler must obey standard C++ aliasing rules, it cannot assume that writing to `Y` will not alter upcoming values of `X`.

---

### In What Real Case Could `X` and `Y` Actually Overlap?

When you declare a function:
```cpp
void saxpy(int n, float a, const float* x, float* y);
```
The compiler does not know where `x` and `y` were allocated. It only sees two memory addresses.

Consider a caller who passes two shifted slices of the **exact same memory buffer**:

```cpp
float buffer[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

// Both pointers point into the SAME buffer, shifted by 1 float:
const float* x = &buffer[0]; // Covers indices 0, 1, 2...
float* y       = &buffer[1]; // Covers indices 1, 2, 3...

saxpy(3, 2.0f, x, y);
```

Let's look at the physical memory layout:

![Pointer Aliasing Hazard: Overlapping Buffer Memory](assets/buffer_aliasing.svg)

* `x[0]` is `buffer[0]` (`10.0`)
* `x[1]` is `buffer[1]` (`20.0`)
* But `y[0]` is ALSO `buffer[1]`!

Now trace the loop `y[i] = 2.0f * x[i] + y[i]`:

1. **At `i = 0`**:
   - Read `x[0]` (`10.0`).
   - Read `y[0]` (`20.0`).
   - Compute: `2.0 * 10.0 + 20.0 = 40.0`.
   - Write `40.0` into `y[0]`. This physically overwrites `buffer[1]` to `40.0`!

![Mutation Hazard: Buffer State After Iteration i = 0 Overwrite](assets/buffer_overwrite.svg)

2. **At `i = 1`**:
   - The loop now reads `x[1]`.
   - But `x[1]` is `buffer[1]`, which was just changed to `40.0` in the previous step!
   - It must compute: `2.0 * 40.0 + 30.0 = 110.0`.

If the compiler had tried to optimize by pre-loading `x[0], x[1], x[2]` into fast registers before the loop began:
- It would have kept the stale value `x[1] = 20.0`.
- The output would be completely wrong (`2.0 * 20.0 + 30.0 = 70.0` instead of `110.0`).

Because C and C++ legally permit programmers to pass overlapping slices (often used in audio echo filters, time-series shifts, and convolution stencils), the compiler is forced to assume the worst.

---

### Why Can't the Compiler "Just Read the Whole Program"?

A natural question is: *"Why can't the compiler look at the caller and figure out what the programmer meant?"*

There are two fundamental reasons:

#### 1. The "Different Days" Problem (Separate Compilation Units)
In C++, functions are compiled in isolated translation units:
* `saxpy` might be compiled into a static library or shared object (`.so`) on Monday.
* `main.cpp` might be written months later by a completely different engineer who links against that library.
* When compiling `saxpy`, the compiler has zero knowledge of who will call it in the future. It must generate machine code that runs 100% correctly for any legal caller.

#### 2. The Golden Law of Compilers: "Correctness Always Trumps Speed"
If a compiler guesses wrong and optimizes aggressively, it silently corrupts data. 
If there is even a 0.0001% chance that pointers overlap, Clang, GCC, and NVCC are strictly forbidden from caching `x` in registers across writes to `y`.

---

### The Cost of Paranoia: Register Invalidation

Without explicit guarantees:
1. The compiler loads `X[0]` into a register.
2. It writes `Y[0]` back to memory.
3. Because writing to `Y[0]` *might* have overwritten `X[1]`, the CPU pipeline is forced to **discard its cached prefetch** and issue a redundant, high-latency reload of `X[1]` from memory on the next iteration!
4. It cannot unroll the loop, cannot vectorize, and cannot schedule instructions concurrently.

---

### The Remedy: The `__restrict__` Legal Contract

When you declare a pointer with `__restrict__`, you sign a contract with the compiler:

```cpp
void saxpy_optimized(size_t n, float alpha, 
                     const float* __restrict__ x, 
                     float* __restrict__ y)
```

> **The `__restrict__` Contract**: *"I, the programmer, guarantee that for the lifetime of this function, the memory pointed to by `x` and `y` will never overlap."*

### Hardware Consequences of `__restrict__`:

1. **On Modern CPUs (x86-64 Clang / GCC)**:
   The compiler instantly vectorizes the loop into full AVX2 or AVX-512 instructions (`vfmadd213ps`). It unrolls the loop 4-way or 8-way, pipelining independent memory requests to hide DRAM latency.

2. **On NVIDIA GPUs (`nvcc`)**:
   Adding `const float* __restrict__ x` tells `nvcc` to emit `LDG.E` assembly instructions. These bypass the write-heavy L1 cache and read directly through the **dedicated Read-Only Data Cache** (formerly texture cache), freeing up L1 bandwidth for write-traffic and eliminating cache thrashing!


---

## 4. Strided Traversal & Cache Line Waste

In deep learning tensors, data is not always laid out contiguously. Strided accesses occur frequently:
* Extracting a single color channel from interleaved RGB image data (`stride = 3`).
* Slicing columns from a row-major matrix (`stride = Cols`).
* Batching sequences with non-unit strides.

### The Physical Analogy: The Box of Eggs

Memory controllers never move individual 4-byte floats across the physical bus. They move data in atomic **64-byte chunks (Cache Lines)** on CPUs, and **128-byte sectors** on NVIDIA GPUs.

Imagine going to the grocery store to buy eggs:
* The store **only sells 16-egg cartons** (a 64-byte cache line = 16 floats). You cannot buy a single egg.

```formula
Floats_Per_Cache_Line = 64 Bytes / 4 Bytes_per_Float = 16 Floats
```

```diagram:cache-line
{
  "title": "Hardware 64-Byte Cache Line Utilization across Strides",
  "subtitle": "Contiguous (stride=1) vs Strided (stride=4) memory bus efficiency"
}
```

Now look at what happens when your algorithm traverses memory with different strides:

### Case 1: Contiguous Traversal (`stride = 1`)
* Your loop requests `X[0]`.
* The memory controller fetches the entire 64-byte cache line containing `X[0]` through `X[15]`.
* On iterations `1` through `15`, the CPU/GPU reads directly from the ultra-fast L1 cache (**0.8 ns**).
* **Carton Utilization**: You ate all 16 eggs in the carton.
* **Bus Efficiency**: **100%**.

### Case 2: Strided Traversal (`stride = 2`)
* Your loop requests `X[0]`, then `X[2]`, `X[4]`, `X[6]`...
* The memory controller still had to haul all 64 bytes (16 floats) from DRAM.
* But your code only uses the 8 even floats and ignores the 8 odd floats.
* **Carton Utilization**: You ate 8 eggs and threw 8 in the trash.
* **Bus Efficiency**: **50%**.

### Case 3: Strided Traversal (`stride = 16`)
* Your loop requests `X[0]`, then `X[16]`, then `X[32]`...
* For every single float your code touches, the memory controller must fetch a brand new 64-byte cache line!
* Out of the 16 floats hauled across the bus, your code uses **only 1** and discards the other 15.
* **Carton Utilization**: You bought a 16-egg carton, ate 1 egg, and threw 15 eggs in the trash.
* **Bus Efficiency**: `4 / 64 = 6.25%`.

```table:stride_efficiency
| Stride | Elements Requested | Bytes Consumed | Hardware Bytes Hauled | Bus Efficiency | Memory Bandwidth Penalty |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`stride = 1`** | `X[0], X[1], X[2], ... X[15]` | 64 Bytes | 64 Bytes | **100.0%** | **1.0x (Peak Speed)** |
| **`stride = 2`** | `X[0], X[2], X[4], ... X[14]` | 32 Bytes | 64 Bytes | **50.0%** | **2.0x Slower** |
| **`stride = 4`** | `X[0], X[4], X[8], X[12]` | 16 Bytes | 64 Bytes | **25.0%** | **4.0x Slower** |
| **`stride = 8`** | `X[0], X[8]` | 8 Bytes | 64 Bytes | **12.5%** | **8.0x Slower** |
| **`stride = 16`** | `X[0]` | 4 Bytes | 64 Bytes | **6.25%** | **16.0x Slower** |
```

> [!CAUTION]
> If a strided traversal has `stride >= 16`, your code will run **16 times slower** on memory bandwidth alone, even if the total number of FLOPs executed is identical!

---

## 5. Vectorized Memory Access: `float4` & 128-Bit Loads

When elements are contiguous (`stride = 1`), requesting data in 32-bit scalar chunks (`LDR.32` or single-float loads) chokes the instruction pipeline with too many discrete memory commands.

Modern processors have wide load/store datapaths:
* **NVIDIA GPUs**: A single thread can load **16 contiguous bytes (4 single-precision floats)** in a single instruction (`LDG.E.128`) using the native vector type `float4`.
* **AMD Zen 5 (9950X)**: Native **512-bit AVX-512** instructions (`vmovups`) can load **64 contiguous bytes (16 floats)** in a single clock cycle.

### The `float4` Vectorized Kernel Pattern:

```cpp
void saxpy_vectorized_128(int n, float alpha, 
                          const float* __restrict__ x, 
                          float* __restrict__ y) {
    // Process 4 floats (16 bytes = 128 bits) per iteration:
    int n_vec = n / 4;
    
    const float4* vx = reinterpret_cast<const float4*>(x);
    float4* vy = reinterpret_cast<float4*>(y);
    
    for (int i = 0; i < n_vec; ++i) {
        float4 x_quad = vx[i]; // Single 128-bit memory load
        float4 y_quad = vy[i]; // Single 128-bit memory load
        
        y_quad.x = alpha * x_quad.x + y_quad.x;
        y_quad.y = alpha * x_quad.y + y_quad.y;
        y_quad.z = alpha * x_quad.z + y_quad.z;
        y_quad.w = alpha * x_quad.w + y_quad.w;
        
        vy[i] = y_quad;        // Single 128-bit memory store
    }
    
    // Scalar cleanup loop for remaining elements:
    for (int i = n_vec * 4; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}
```

### The Alignment Golden Rule for 128-Bit Loads

To issue a 128-bit vector load without crashing:
* The starting memory address MUST be **16-byte aligned**:

```formula
Address % 16 == 0
```

> [!IMPORTANT]
> If you cast an unaligned address (e.g. `0x1004`, where `0x1004 % 16 == 4`) to `float4*` and attempt a 128-bit load:
> * On CPUs, it incurs an unaligned split-load penalty.
> * On GPUs, it triggers an instant hardware trap: **`CUDA_ERROR_ILLEGAL_ADDRESS`**.
> Always allocate aligned memory using `posix_memalign`, `aligned_alloc`, or `cudaMalloc` (which naturally aligns to 256 bytes).

---

## 6. The Industry Remedy: Operator Fusion

Because element-wise operations have such low arithmetic intensity (intensity = 0.167), they spend almost all their time waiting on the memory bus.

The single most impactful optimization in modern deep learning (the driving principle behind **FlashAttention**, **PyTorch 2.0 TorchInductor**, and **TensorRT**) is **Operator Fusion**.

### The Naive Unfused Pipeline

Consider a standard Transformer block operation: `Y = GeLU(X + Residual)`.

In naive deep learning frameworks, this executes as two separate back-to-back GPU kernels, incurring multiple redundant DRAM round-trips. In an optimized systems pipeline, operator fusion executes both operations in a single kernel pass:

![Operator Fusion: Unfused DRAM Round-Trips vs Fused Register Reuse](assets/fused_pipeline.svg)

1. **Kernel 1 (`Add`) in Unfused Execution**:
   - Read `X[i]` (4B) from DRAM.
   - Read `Residual[i]` (4B) from DRAM.
   - Compute `Temp[i] = X[i] + Residual[i]`.
   - **Write `Temp[i]` (4B) back to DRAM**.
2. **Kernel 2 (`GeLU`) in Unfused Execution**:
   - **Read `Temp[i]` (4B) from DRAM**.
   - Compute `GeLU(Temp[i])`.
   - Write `Y[i]` (4B) to DRAM.

```formula
Unfused_DRAM_Traffic = 4B(X) + 4B(Res) + 4B(Temp Write) + 4B(Temp Read) + 4B(Y) = 20 Bytes
```

Notice the tragedy: `Temp` was written to DRAM only to be immediately read back and thrown away!

### The Fused Pipeline: Zero DRAM Roundtrips

In a fused kernel, the intermediate result `Temp` is kept strictly inside **fast on-chip registers**, bypassing DRAM completely:

```cpp
void fused_add_gelu(int n, 
                    const float* __restrict__ x, 
                    const float* __restrict__ residual, 
                    float* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        // Step 1: Read inputs from DRAM once:
        float val = x[i] + residual[i]; // 'val' lives strictly in a register!
        
        // Step 2: Compute GeLU entirely in register SRAM:
        float cube = val * val * val;
        float inner = 0.79788456f * (val + 0.044715f * cube);
        float gelu = 0.5f * val * (1.0f + tanhf(inner));
        
        // Step 3: Write only the final result to DRAM:
        y[i] = gelu;
    }
}
```

```formula
Fused_DRAM_Traffic = 4B(X) + 4B(Residual) + 4B(Y) = 12 Bytes
```

### The Production Impact:

| Metric | Naive Unfused | Fused Kernel | Hardware Savings |
| :--- | :--- | :--- | :--- |
| **DRAM Memory Traffic** | 20 Bytes / elem | **12 Bytes / elem** | **40% Memory Bandwidth Saved** |
| **Kernel Launches** | 2 launches | **1 launch** | **50% Driver Launch Overhead Saved** |
| **DRAM Allocations** | Requires `Temp` buffer | **Zero intermediate memory** | **Saves Gigabytes of VRAM** |
| **Execution Time** | Slow (Bus throttled) | **1.6x - 2.0x Faster** | Direct end-to-end acceleration |

---

## 7. Architectural Golden Rules for Systems Engineers

1. **Measure Arithmetic Intensity First**:
   Always compute `FLOPs / Bytes`. If your algorithm has `intensity < 10.0` on modern GPUs (like your RTX 5090), you are strictly memory-bound. Focus 100% of your effort on memory bandwidth optimization.


2. **Always Use `__restrict__` on Independent Pointers**:
   Sign the non-aliasing contract. It allows the compiler to keep variables in registers and use high-speed read-only caches (`LDG.E`).

3. **Keep Strides Contiguous (`stride = 1`)**:
   Strided traversals waste cache line bandwidth. A stride of 16 wastes 93.75% of your memory bus bandwidth.

4. **Vectorize with 128-Bit Loads (`float4` / AVX-512)**:
   Coalesce 4 floats into a single bus request. Always ensure starting addresses are 16-byte aligned.

5. **Fuse Element-Wise Operators Aggressively**:
   Never write intermediate tensors back to DRAM just to feed the next element-wise layer. Keep intermediates in registers and fuse!
