# Module 2.1: Lambdas, Closures & Function Objects in CUDA

In modern deep learning systems and GPU engines, modern C++ abstractions must achieve **zero-cost performance**. Abstractions should not cost a single extra clock cycle, register spill, or memory transaction compared to hand-written PTX assembly.

C++ lambdas and function objects (functors) provide the foundation for generic, composable CUDA kernels: from fused activation pipelines (SwiGLU, GeLU, RMSNorm scaling) to custom parallel reduction and scan primitives.

---

## 1. Physical Mental Model: What is a Lambda Closure in Silicon?

A lambda is not a function pointer. At compile time, the compiler synthesizes a unique, anonymous C++ struct (the **closure type**) where:
1. Every captured variable becomes a physical member variable inside the struct.
2. The lambda body becomes an inlined `operator()` member method.

```cpp
// This user lambda:
float scale = 0.5f;
int bias_offset = 4;
auto linear_act = [scale, bias_offset] __device__ (float x, const float* bias) {
    return x * scale + bias[bias_offset];
};

// Is lowered by nvcc into this physical layout:
struct __Lambda_64_bit_Layout {
    float scale;         // 4 bytes offset +0
    int bias_offset;     // 4 bytes offset +4
    // Total size = 8 bytes
    
    __device__ inline float operator()(float x, const float* bias) const {
        return x * scale + bias[bias_offset];
    }
};
```

### The Hardware Journey: Host Stack -> Kernel Parameter Buffer (Constant Memory)
When you pass a lambda into a `__global__` kernel launch:
```cpp
kernel<<<grid, block>>>(d_in, d_out, N, linear_act);
```
1. **Host-Side Packaging**: The CPU packs the 8-byte closure struct into the launch configuration payload.
2. **Driver Transfer**: The CUDA driver writes this struct into the GPU SM's dedicated **Constant Memory Parameter Bank (`c[0x0]`)**.
3. **Hardware Dispatch**: When SM warps begin executing, each thread reads the captured variables directly from constant cache (`c[0x0]`) or has them hoisted directly into high-speed **thread registers** ($R_0 \dots R_{255}$).
4. **Zero Overhead**: There is zero heap allocation, zero pointer chasing, and zero runtime indirection.

---

## 2. Capture Semantics & The Fatal `[this]` Trap

Understanding memory address spaces is critical when capturing variables across the CPU-GPU boundary:

| Capture Mode | Syntax | Silicon Reality | GPU Safety |
| :--- | :--- | :--- | :--- |
| **By Value** | `[=]` or `[x]` | Copies data bytes into the closure struct. | **100% Safe**. Copied to GPU Constant Memory. |
| **By Reference** | `[&]` or `[&x]` | Stores an 8-byte 64-bit pointer pointing to the caller's stack frame in host CPU RAM. | **FATAL CRASH**. GPU dereferences a CPU virtual address (`CUDA_ERROR_ILLEGAL_ADDRESS`). |
| **Init-Capture** | `[val = expr]` | Evaluates expression on CPU and stores the result by value in the closure. | **100% Safe & Idiomatic**. |
| **Implicit `this`** | `[=]` inside class | Silently captures `this` pointer (CPU heap/stack address)! | **SILENT BUG**. Kernel attempts to read CPU memory. |

### The `[this]` Pointer Bug in Production Kernels:
```cpp
class TransformerLayer {
    float dropout_prob = 0.1f;

    void launch(float* d_data, int N) {
        // BUG: [=] inside a member function silently captures `this` pointer!
        auto op = [=] __device__ (float x) { 
            return x * (1.0f - dropout_prob); // Translates to: this->dropout_prob!
        };
        transform_kernel<<<grid, block>>>(d_data, N, op); // CRASH on GPU!
    }
};
```

### The C++17 / C++20 Fix:
Always use explicit member capture or C++17 `[*this]` (capture class by value):
```cpp
// FIX 1: Explicit init-capture (cleanest & minimal bytes):
auto op = [p = this->dropout_prob] __device__ (float x) {
    return x * (1.0f - p);
};

// FIX 2: C++17 copy entire object by value:
auto op = [*this] __device__ (float x) {
    return x * (1.0f - dropout_prob);
};
```

---

## 3. CUDA Extended Lambdas (`--extended-lambda`) & Inlining Physics

Standard C++ lambdas are host-only. To use lambdas inside CUDA kernels, you must compile with the flag:
```bash
nvcc -O3 --extended-lambda ...
```
This enables the execution space qualifiers:
* `[] __device__ (...)`: Callable only on the GPU.
* `[] __host__ __device__ (...)`: Dual-target closure compiled for both CPU and GPU execution.

### Why Lambdas Crush Function Pointers: The PTX Inlining Proof
Consider passing an activation function to a transform kernel.

#### Approach A: Function Pointer (`float (*func)(float)`)
```cpp
__global__ void transform_fp(float* out, const float* in, int n, float (*func)(float)) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = func(in[i]);
}
```
* **Hardware Assembly (PTX)**: Compiles to an indirect branch instruction (`call.uni %rd1;`).
* **The Penalty**:
  1. The GPU warp execution pipeline must resolve a variable target address.
  2. Prevents the compiler from optimizing across instruction boundaries.
  3. Registers cannot be shared; the compiler is forced to spill parameters into slow **Thread-Local DRAM memory (Local Memory)**.

#### Approach B: Templated Lambda Functor (`template <typename Op>`)
```cpp
template <typename Op>
__global__ void transform_lambda(float* out, const float* in, int n, Op op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}
```
* **Hardware Assembly (PTX)**: Because `Op` is a unique compile-time type, `nvcc` completely inlines `op.operator()` into the body of the kernel.
* The instruction is emitted directly as a hardware Fused Multiply-Add:
```assembly
// Hand-optimized PTX output:
ld.global.f32    %f1, [%rd1];     // Load in[i]
fma.rn.f32       %f2, %f1, %f3, %f4; // FMA directly in registers!
st.global.f32    [%rd2], %f2;     // Store out[i]
```
* **Zero function call overhead. Zero local memory spills. Maximum arithmetic throughput.**

---

## 4. Production Application: Composable SwiGLU Activation Pipeline

In state-of-the-art Large Language Models (LLaMA 3, Mistral, DeepSeek), the feed-forward network (FFN) uses the **SwiGLU** activation:
```formula
SwiGLU(gate, up) = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
```

Using extended lambdas with higher-order kernel templates, we can write a single generic binary transform kernel and instantiate any fused activation without repeating low-level CUDA boilerplate:

```cpp
// Generic 2-Input Fused Streaming Kernel:
template <typename BinaryOp>
__global__ void fused_elementwise_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         int n,
                                         BinaryOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

// Host invocation with inlined device lambda:
void launch_swiglu(const float* d_gate, const float* d_up, float* d_out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Zero-overhead lambda closure defining SwiGLU:
    auto swiglu_op = [] __device__ (float gate, float up) -> float {
        // Fast hardware reciprocal and exponential intrinsics:
        float silu = gate / (1.0f + __expf(-gate));
        return silu * up;
    };

    fused_elementwise_kernel<<<blocks, threads, 0, stream>>>(
        d_gate, d_up, d_out, n, swiglu_op
    );
}
```

---

## 5. Summary & Best Practices

1. **Physical Reality**: Lambdas are unique structs with member variables representing captured values. When passed to kernels, they are placed in GPU Constant Memory (`c[0x0]`).
2. **Never Capture by Reference (`[&]`)**: Reference captures point to host CPU stack memory and crash the GPU. Always capture by value `[=]` or explicit init-capture `[var = expr]`.
3. **Beware Implicit `[this]`**: Inside class methods, `[=]` implicitly captures the host `this` pointer. Extract members explicitly before launching device closures.
4. **Template Kernel on `Op`**: Always declare kernel operators with `template <typename Op>` rather than function pointers to guarantee compiler inlining and avoid local memory spills.
5. **Use Fast Math Intrinsics**: Inside device lambdas, use CUDA math intrinsics (`__expf`, `fmaxf`, `__fmaf_rn`) to utilize dedicated hardware Special Function Units (SFUs).

