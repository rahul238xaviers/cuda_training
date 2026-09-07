# Module 2.1 — Lambdas, Closures & Function Objects in CUDA

## 1. The Anatomy of Modern C++ Lambdas

In modern C++ (C++14/17/20), lambdas provide concise, inlined syntax for creating anonymous function objects (functors):
```cpp
[capture-clause](parameters) specifiers -> return-type {
    // function body
}
```

Under the hood, the compiler translates every lambda expression into a unique, unnamed struct with an overloaded `operator()`:
```cpp
// This lambda:
auto add_bias = [bias = 0.5f](float x) { return x + bias; };

// Is equivalent to this functor:
struct __Lambda_123 {
    float bias;
    __Lambda_123(float b) : bias(b) {}
    float operator()(float x) const { return x + bias; }
};
```

---

## 2. Capture Semantics & Gotchas

| Capture Mode | Syntax | Lifetime & Memory Behavior |
| :--- | :--- | :--- |
| **By Value** | `[=]` or `[x]` | Copies variables into the closure struct. Safe across thread boundaries. |
| **By Reference** | `[&]` or `[&x]` | Stores reference/pointer to variable. **Dangerous** if caller frame exits or if sent to GPU! |
| **Init-Capture** | `[ptr = std::move(p)]` | Moves non-copyable objects (e.g. `std::unique_ptr`) into the closure. |
| **Mutable** | `[x]() mutable { x++; }` | Allows modifying value-captured members (normally `operator()` is `const`). |

> [!CAUTION]
> **Never capture host references into device closures!**
> When passing a closure from host code to a GPU kernel, any reference capture `[&]` points to host stack memory, causing a fatal page fault (illegal memory access) on the GPU. **All device lambdas must capture by value `[=]`**.

---

## 3. CUDA Extended Lambdas (`--extended-lambda`)

By default, standard C++ compilers only generate host CPU code for lambdas. NVIDIA CUDA enables **Extended Lambdas** with the `--extended-lambda` compiler flag.
This allows decorating closures with CUDA execution qualifiers:

```cpp
// Device-only lambda (callable inside GPU kernels)
auto relu = [] __device__ (float x) { return fmaxf(0.0f, x); };

// Dual host/device lambda
auto square = [] __host__ __device__ (float x) { return x * x; };
```

### Passing Lambdas Directly to `__global__` Kernels
A high-performance pattern in modern GPU frameworks is the **Higher-Order Transform Kernel**:
```cuda
template <typename Op>
__global__ void transform_kernel(const float* in, float* out, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(in[idx]);
    }
}
```

### Why Lambdas Beat Function Pointers
1. **Zero Function Call Overhead**: Function pointers cannot be inlined by the compiler and incur dynamic indirect branch latency.
2. **Aggressive Inlining**: Because each lambda generates a distinct C++ type `Op`, the CUDA compiler inlines the operator directly into the inner loop, producing instruction-level parallelism, FMA fusion, and register reuse identical to hand-written kernels!

---

## 4. Machine Learning Applications: Composable Activation Pipelines

In LLM architectures (like LLaMA and Mistral), activation functions are frequently modified:
- **GELU**: $x \cdot \Phi(x) \approx 0.5x \cdot (1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$
- **SiLU / Swish**: $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
- **SwiGLU**: $\text{SwiGLU}(G, U) = \text{SiLU}(G) \cdot U$

Using higher-order templated transform kernels with extended lambdas, an engine can instantiate any custom activation or fused post-processing step at compile-time with zero runtime overhead or code duplication.
