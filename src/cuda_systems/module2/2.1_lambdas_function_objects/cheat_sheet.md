# Topic 2.1: Lambdas, Closures & Function Objects in CUDA — Cheat Sheet

A concise reference for using modern C++ lambdas and function objects with CUDA kernels.

---

### 1. Host vs. Device Lambda Syntax
- **Device Qualifier**: Use `__device__` inside the lambda declaration (requires `--extended-lambda`).
- **Dual Qualifier**: `__host__ __device__` allows the same functor to execute on both CPU and GPU.
- **Example**:
  ```cpp
  auto relu = [] __device__ (float x) -> float {
      return (x > 0.0f) ? x : 0.0f;
  };
  ```

---

### 2. Lambda Capture Rules (Physical Safety)
- **Rule 1 (By Value Only)**: Always capture by value `[=]` or `[alpha, beta]` when passing closures to GPU kernels.
- **Rule 2 (No Reference Captures)**: Never capture host stack references `[&]` for device execution. It creates dangling pointers and GPU illegal memory traps.
- **Rule 3 (Mutable Closures)**:
  ```cpp
  auto counter = [c = 0]() mutable { return c++; };
  ```

---

### 3. Higher-Order Transform Kernel Pattern
- Pass arbitrary elementwise operations into a templated kernel:
  ```cpp
  template <typename Op>
  __global__ void transform_kernel(const float* in, float* out, int N, Op op) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N) {
          out[idx] = op(in[idx]);
      }
  }
  ```
- **Hardware Advantage**: Because each lambda produces a unique C++ struct type `Op`, the CUDA compiler completely inlines `operator()` into the inner loop with zero function call overhead.

---

### 4. LLM Systems Context
- **Activation Pipelines**: Instantiating custom activations (GELU, SiLU, SwiGLU, QuickGELU) with compile-time zero-cost inlining.
- **Fused Scale-Bias**: Fusing activation functions with layer scaling without writing new separate kernels.

---

### 5. Common Pitfalls & Anti-Patterns
- Forgetting `--extended-lambda` in compiler flags (`nvcc -std=c++17 --extended-lambda`).
- Passing capturing lambdas to `cudaMemcpy` (capturing lambdas are not trivially copyable if they hold complex objects).
