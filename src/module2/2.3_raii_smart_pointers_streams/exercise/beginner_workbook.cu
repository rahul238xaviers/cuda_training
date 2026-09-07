// ==============================================================================
// Module 2.3: Smart Pointers & RAII — Beginner Workbook
// ==============================================================================
// In this workbook, you will implement foundational RAII wrappers to prevent
// memory leaks and safely manage CUDA execution resources:
// 1. Custom std::unique_ptr Deleter for cudaFree
// 2. make_device_unique<T> Factory Helper
// 3. Pinned Host Memory Custom Deleter for cudaFreeHost
// 4. StreamRAII Scope-Bound CUDA Stream Manager
// 5. Basic EventTimer RAII Helper for GPU Execution Timing
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ==============================================================================
// Exercise 1: Custom std::unique_ptr Deleter for cudaFree
// Implement CudaDeleter whose operator()(void* ptr) calls cudaFree if ptr != nullptr.
// ==============================================================================
struct CudaDeleter {
    static bool called;
    void operator()(void* ptr) const {
        // TODO:
        // If ptr is not null:
        // cudaFree(ptr);
        // called = true;
    }
};
bool CudaDeleter::called = false;

template <typename T>
using DevicePtr = std::unique_ptr<T, CudaDeleter>;

// ==============================================================================
// Exercise 2: make_device_unique<T> Factory Helper
// Allocate device memory using cudaMalloc and return DevicePtr<T>.
// ==============================================================================
template <typename T>
DevicePtr<T> make_device_unique(size_t count) {
    // TODO:
    // 1. T* raw = nullptr;
    // 2. cudaMalloc(&raw, count * sizeof(T));
    // 3. return DevicePtr<T>(raw);
    return DevicePtr<T>(nullptr);
}

// ==============================================================================
// Exercise 3: Pinned Host Memory Custom Deleter
// Pinned (page-locked) host memory allocated via cudaMallocHost must be freed
// via cudaFreeHost. Implement CudaHostDeleter.
// ==============================================================================
struct CudaHostDeleter {
    void operator()(void* ptr) const {
        // TODO:
        // If ptr is not null, call cudaFreeHost(ptr);
    }
};

template <typename T>
using HostPinnedPtr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
HostPinnedPtr<T> make_pinned_host(size_t count) {
    // TODO:
    // Allocate with cudaMallocHost and return HostPinnedPtr<T>.
    return HostPinnedPtr<T>(nullptr);
}

// ==============================================================================
// Exercise 4: StreamRAII Scope-Bound CUDA Stream Manager
// Constructor creates cudaStream_t with cudaStreamCreate(&stream_).
// Destructor synchronizes with cudaStreamSynchronize(stream_) and destroys with
// cudaStreamDestroy(stream_).
// ==============================================================================
class StreamRAII {
private:
    cudaStream_t stream_;

public:
    StreamRAII() {
        // TODO: Create stream
        stream_ = nullptr;
    }

    ~StreamRAII() {
        // TODO: Synchronize and destroy stream
    }

    cudaStream_t get() const { return stream_; }

    // Disable copy
    StreamRAII(const StreamRAII&) = delete;
    StreamRAII& operator=(const StreamRAII&) = delete;
};

// ==============================================================================
// Exercise 5: Basic EventTimer RAII Helper
// Manages two cudaEvent_t objects: start_ and stop_.
// Constructor creates events; destructor destroys them.
// ==============================================================================
class EventTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;

public:
    EventTimer() {
        // TODO: cudaEventCreate(&start_); cudaEventCreate(&stop_);
    }

    ~EventTimer() {
        // TODO: cudaEventDestroy(start_); cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        // TODO: cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        // TODO: cudaEventRecord(stop_, stream);
    }

    float elapsed_ms() {
        // TODO:
        // cudaEventSynchronize(stop_);
        // float ms = 0.0f;
        // cudaEventElapsedTime(&ms, start_, stop_);
        // return ms;
        return 0.0f;
    }
};

__global__ void dummy_kernel(float* d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) d[idx] = d[idx] * 2.0f;
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: DevicePtr Deleter
    // --------------------------------------------------------------------------
    {
        float* raw = nullptr;
        CHECK_CUDA(cudaMalloc(&raw, 16 * sizeof(float)));
        CudaDeleter::called = false;
        {
            DevicePtr<float> ptr(raw);
        } // Destructor runs CudaDeleter
        if (CudaDeleter::called) {
            std::cout << "[Test 1: DevicePtr Deleter] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: DevicePtr Deleter] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: make_device_unique
    // --------------------------------------------------------------------------
    {
        auto d_buf = make_device_unique<int>(32);
        if (d_buf != nullptr) {
            std::vector<int> h_data(32, 42);
            CHECK_CUDA(cudaMemcpy(d_buf.get(), h_data.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
            std::vector<int> h_out(32, 0);
            CHECK_CUDA(cudaMemcpy(h_out.data(), d_buf.get(), 32 * sizeof(int), cudaMemcpyDeviceToHost));
            if (h_out[0] == 42 && h_out[31] == 42) {
                std::cout << "[Test 2: make_device_unique] PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "[Test 2: make_device_unique] FAILED" << std::endl;
            }
        } else {
            std::cout << "[Test 2: make_device_unique] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: Pinned Host Memory
    // --------------------------------------------------------------------------
    {
        auto pinned = make_pinned_host<float>(16);
        if (pinned != nullptr) {
            pinned.get()[0] = 100.0f;
            pinned.get()[15] = 200.0f;
            if (pinned.get()[0] == 100.0f && pinned.get()[15] == 200.0f) {
                std::cout << "[Test 3: Pinned Host Memory] PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "[Test 3: Pinned Host Memory] FAILED" << std::endl;
            }
        } else {
            std::cout << "[Test 3: Pinned Host Memory] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: StreamRAII
    // --------------------------------------------------------------------------
    {
        bool ok = false;
        {
            StreamRAII stream_guard;
            if (stream_guard.get() != nullptr) {
                auto d_mem = make_device_unique<float>(16);
                CHECK_CUDA(cudaMemsetAsync(d_mem.get(), 0, 16 * sizeof(float), stream_guard.get()));
                ok = true;
            }
        } // Stream synchronized and destroyed
        if (ok) {
            std::cout << "[Test 4: StreamRAII] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: StreamRAII] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 5: EventTimer
    // --------------------------------------------------------------------------
    {
        EventTimer timer;
        auto d_buf = make_device_unique<float>(1024);
        timer.start();
        dummy_kernel<<<4, 256>>>(d_buf.get(), 1024);
        timer.stop();
        float ms = timer.elapsed_ms();
        if (ms > 0.0f) {
            std::cout << "[Test 5: EventTimer] PASSED (elapsed: " << ms << " ms)" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: EventTimer] FAILED" << std::endl;
        }
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
