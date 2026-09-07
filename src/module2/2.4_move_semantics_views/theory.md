# Module 2.4 — Move Semantics, Perfect Forwarding & Tensor Views

## 1. Move Semantics & The Rule of Five for GPU Buffers

Heavy GPU resources (such as multi-gigabyte device tensors) must never be copied by accident.
In C++11 and beyond, **Move Semantics** allow transferring ownership of underlying device pointers without copying memory:

```cpp
template <typename T>
class DeviceTensor {
private:
    T* data_;
    size_t count_;

public:
    // Move Constructor
    DeviceTensor(DeviceTensor&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr; // Nullify source to prevent double-free
        other.count_ = 0;
    }

    // Move Assignment Operator
    DeviceTensor& operator=(DeviceTensor&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Prohibit accidental deep copies
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
};
```

---

## 2. Perfect Forwarding in Kernel Launch Wrappers

In template metaprogramming, a universal reference (`T&&` in a deduced template parameter context) can bind to both lvalues and rvalues.
`std::forward<T>` preserves the original value category when passing arguments into kernel launchers:

```cpp
template <typename Kernel, typename... Args>
void dispatch_kernel(Kernel k, dim3 grid, dim3 block, Args&&... args) {
    k<<<grid, block>>>(std::forward<Args>(args)...);
}
```

---

## 3. Owning Tensors vs Non-Owning Tensor Views

A fundamental design pattern in deep learning engines (such as PyTorch ATen, NumPy, and CUTLASS) is the separation between:
1. **Owning Storage (`DeviceTensor<T>`)**: Owns the contiguous memory allocation in GPU DRAM.
2. **Non-Owning View (`TensorView<T>`)**: Lightweight descriptor referencing a memory pointer alongside metadata (`shape`, `strides`, `offset`).

### Zero-Copy Operations
Because a view only stores metadata:
- **Reshaping**: Changes the `shape` array without touching DRAM.
- **Transposing**: Swaps `shape[0], shape[1]` and `strides[0], strides[1]` in $O(1)$ time without copying memory.
- **Slicing**: Adjusts the base pointer and updates dimensions in $O(1)$ time.

### The Strided Multi-Dimensional Indexing Formula
For a tensor with dimensions $D$ and coordinates $(i_0, i_1, \dots, i_{D-1})$:
$$\text{Linear Offset} = \sum_{d=0}^{D-1} i_d \cdot \text{strides}[d]$$

In row-major order:
$$\text{strides}[D-1] = 1, \quad \text{strides}[d] = \text{strides}[d+1] \cdot \text{shape}[d+1]$$
In column-major order:
$$\text{strides}[0] = 1, \quad \text{strides}[d] = \text{strides}[d-1] \cdot \text{shape}[d-1]$$
