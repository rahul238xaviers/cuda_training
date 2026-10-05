# Topic 2.2: Templates, SFINAE & Precision Type Traits — Cheat Sheet

A concise reference for compile-time kernel dispatch and precision abstractions in CUDA.

---

### 1. Compile-Time Precision Dispatch
- **Template Kernel Pattern**:
  ```cpp
  template <typename T>
  __global__ void vector_add(const T* a, const T* b, T* c, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
          c[idx] = a[idx] + b[idx];
      }
  }
  ```
- **Explicit Instantiation**: Ensures the PTX is compiled ahead-of-time for target types:
  ```cpp
  template __global__ void vector_add<float>(const float*, const float*, float*, int);
  template __global__ void vector_add<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
  ```

---

### 2. Type Traits for Low-Precision Math
- Use type traits (`std::is_same_v`, `sizeof(T)`) to select specialized hardware instructions:
  - If `sizeof(T) == 2`: Use packed 16-bit half/bfloat2 intrinsics.
  - If `sizeof(T) == 4`: Use single-precision FMA (`fmaf`).

---

### 3. LLM Systems Context
- **Mixed-Precision Training**: Running weights and activations in FP16/BF16 while computing reductions and optimizer states in FP32.
- **Quantization Support**: Uniform kernel wrappers supporting FP32, FP16, BF16, and INT8 through template specializations.

---

### 4. Common Pitfalls
- Host-only type traits like `std::string` inside `__device__` code (only header-only, constexpr type traits are legal on device).
- Precision truncation: Always accumulate dot-products in FP32 registers even when inputs are BF16/FP16.
