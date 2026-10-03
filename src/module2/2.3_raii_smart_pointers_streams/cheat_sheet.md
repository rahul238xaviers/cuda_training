# Topic 2.3: RAII, Smart Pointers & CUDA Streams — Cheat Sheet

A concise reference for deterministic GPU resource management and multi-stream concurrency.

---

### 1. RAII GPU Device Pointer Wrapper
- Automate `cudaFree` on scope exit:
  ```cpp
  template <typename T>
  struct CudaDeviceBuffer {
      T* ptr = nullptr;
      size_t count = 0;

      explicit CudaDeviceBuffer(size_t n) : count(n) {
          cudaMalloc(&ptr, n * sizeof(T));
      }
      ~CudaDeviceBuffer() {
          if (ptr) cudaFree(ptr);
      }
      // Non-copyable, movable
      CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
      CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
      CudaDeviceBuffer(CudaDeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
          o.ptr = nullptr; o.count = 0;
      }
  };
  ```

---

### 2. CUDA Stream Lifecycle & Priority
- **Stream Creation & Destruction**:
  ```cpp
  cudaStream_t stream;
  cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1); // High priority
  cudaStreamDestroy(stream);
  ```
- **Asynchronous Launch**:
  ```cpp
  kernel<<<grid, block, 0, stream>>>(args...);
  cudaMemcpyAsync(d_out, h_in, bytes, cudaMemcpyHostToDevice, stream);
  ```

---

### 3. LLM Systems Context
- **Pipeline Parallelism**: Overlapping next-layer weight transfer over PCIe with current-layer matrix multiplication on GPU.
- **Continuous Batching**: Allocating individual streams for concurrent decode requests in inference servers (vLLM / TensorRT-LLM).

---

### 4. Common Pitfalls
- Using non-pinned host memory with `cudaMemcpyAsync` (asynchronous transfer falls back to synchronous if host memory is pageable!).
- Stream races: Not synchronizing dependent streams with `cudaStreamWaitEvent`.
