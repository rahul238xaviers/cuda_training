# Module 2.3 — Smart Pointers, RAII & Resource Wrappers

## 1. The RAII Principle in GPU Programming

The **RAII (Resource Acquisition Is Initialization)** idiom binds the lifecycle of a resource to the lifetime of an object.
When programming in raw CUDA C:
```cpp
float* d_ptr;
cudaMalloc(&d_ptr, N * sizeof(float));
cudaStream_t stream;
cudaStreamCreate(&stream);

// If an error occurs here, or the function returns early:
// d_ptr and stream are LEAKED!
cudaStreamDestroy(stream);
cudaFree(d_ptr);
```

In production deep learning engines, manual memory management leads to GPU VRAM exhaustion. RAII encapsulates raw handles within scope-bound C++ classes whose destructors automatically release resources.

---

## 2. Smart Pointers with Custom Deleters

`std::unique_ptr` supports custom deleter functors:
```cpp
struct CudaDeviceDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, CudaDeviceDeleter>;

template <typename T>
unique_device_ptr<T> make_unique_device(size_t count) {
    T* raw = nullptr;
    cudaMalloc(&raw, count * sizeof(T));
    return unique_device_ptr<T>(raw);
}
```

### Pinned Host Memory (`cudaMallocHost` / `cudaFreeHost`)
Host memory for asynchronous DMA transfers must be page-locked (pinned):
```cpp
struct CudaHostDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

template <typename T>
using unique_host_pinned_ptr = std::unique_ptr<T, CudaHostDeleter>;
```

---

## 3. Scoped Stream Management (`StreamGuard`)

When executing multi-stream CUDA pipelines, managing stream creation, synchronization, and destruction across nested functions can become error-prone.
An RAII `StreamGuard` automatically synchronizes and destroys the stream upon exiting scope:
```cpp
class StreamGuard {
private:
    cudaStream_t stream_;
    bool own_;

public:
    StreamGuard() : own_(true) {
        cudaStreamCreate(&stream_);
    }

    explicit StreamGuard(cudaStream_t s) : stream_(s), own_(false) {}

    ~StreamGuard() {
        if (own_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t get() const { return stream_; }
};
```

---

## 4. High-Precision Microsecond Profiling (`EventTimer`)

CUDA kernel launches are asynchronous: host code returns immediately while the GPU continues executing instructions.
Measuring GPU execution time with CPU clocks (`std::chrono`) requires calling `cudaDeviceSynchronize()`, which drains the pipeline and skews performance benchmarks.

Instead, production profiling uses **CUDA Events**:
```cpp
class EventTimer {
private:
    cudaEvent_t start_, stop_;

public:
    EventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed_millis() {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};
```
