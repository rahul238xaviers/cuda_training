# Module 2.2 — Templates, SFINAE & Precision Type Traits

## 1. The Role of Generic Programming in GPU ML Kernels

Modern machine learning libraries (such as PyTorch ATen, FlashAttention, and CUTLASS) do not write separate, duplicate GPU kernels for `float`, `double`, `half`, and `__nv_bfloat16`.
Instead, they write **generic templated kernels** parameterized by tensor data types:
```cuda
template <typename T>
__global__ void generic_vector_add(const T* a, const T* b, T* c, int N);
```

Using templates allows the compiler to generate specialized, instruction-optimal machine code for every type while maintaining a single, clean codebase.

---

## 2. Compile-Time Dispatch: SFINAE & `if constexpr`

### SFINAE (`std::enable_if_t`)
SFINAE (Substitution Failure Is Not An Error) enables or disables function overloads based on compile-time type properties:
```cpp
// Only enabled for floating-point types (float, double)
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
__device__ T safe_inv(T x) {
    return 1.0f / x;
}
```

### Modern Compile-Time Branching (`if constexpr`)
In C++17, `if constexpr` allows conditional compilation within a single function body, eliminating the need for verbose SFINAE boilerplate:
```cpp
template <typename T>
__device__ float to_float(T val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}
```

---

## 3. ML Precision Traits: Decoupling Storage from Accumulation

In Large Language Model training:
- **Storage Precision**: BF16 or FP16 (16-bit) to minimize global DRAM bandwidth and memory footprint.
- **Compute / Accumulation Precision**: FP32 (32-bit) to prevent numerical instability, catastrophic cancellation, underflow, and overflow during reductions (RMSNorm, Softmax, GEMM dot-products).

To enforce this architecture-wide, we define **Precision Type Traits**:
```cpp
template <typename StorageT>
struct AccumulatorTraits {
    using ComputeT = float; // Default fallback
};

template <>
struct AccumulatorTraits<double> {
    using ComputeT = double;
};

// Trait alias helper
template <typename T>
using ComputeType = typename AccumulatorTraits<T>::ComputeT;
```

### Numerical Epsilon & Stability Bounds
Different data types have vastly different machine epsilons and exponent ranges:
- FP32: Epsilon $\approx 1.19 \times 10^{-7}$, min positive $\approx 1.18 \times 10^{-38}$
- FP16: Epsilon $\approx 9.77 \times 10^{-4}$, min normal $\approx 6.10 \times 10^{-5}$
- BF16: Epsilon $\approx 7.81 \times 10^{-3}$, min normal $\approx 1.18 \times 10^{-38}$ (same range as FP32!)

Using a traits struct allows passing safe clamping bounds and normalization epsilons directly into GPU kernels at compile time:
```cpp
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<float> {
    static __device__ __host__ constexpr float eps() { return 1e-6f; }
};

template <>
struct NumericLimits<half> {
    static __device__ __host__ constexpr float eps() { return 1e-3f; }
};
```
