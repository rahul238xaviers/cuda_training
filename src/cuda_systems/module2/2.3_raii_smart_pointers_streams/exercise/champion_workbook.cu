// ==============================================================================
// Module 2.3: Smart Pointers & RAII — Champion Workbook
// ==============================================================================
// In this champion workbook, you will master production-grade GPU resource
// architecture used in high-throughput LLM serving systems (e.g. vLLM / TensorRT-LLM):
// 1. Thread-Safe Device Memory Arena / Slab Allocator
// 2. Scoped CUDA Stream Context Switcher
// 3. Asynchronous Double-Buffering Execution Pipeline
// 4. Statistical RAII Kernel Profiler (Mean, Min, Max Latencies)
// 5. CUDA Graph RAII Lifecycle Manager (cudaGraph_t / cudaGraphExec_t)
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
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
// Exercise 1: Device Memory Arena Allocator
// Allocates a large VRAM slab of `capacity_bytes`.
// allocate(size_t bytes) hands out aligned sub-slices by advancing an offset.
// reset() resets the offset back to 0 without calling cudaFree.
// Destructor calls cudaFree on the master slab.
// ==============================================================================
class DeviceArena {
private:
    char* slab_;
    size_t capacity_;
    size_t offset_;

public:
    explicit DeviceArena(size_t capacity)
        : slab_(nullptr), capacity_(capacity), offset_(0) {
        // TODO: Allocate slab_ with cudaMalloc
    }

    ~DeviceArena() {
        // TODO: Free slab_ with cudaFree
    }

    void* allocate(size_t bytes) {
        // Align to 256 bytes
        size_t aligned_bytes = (bytes + 255) & ~255;
        // TODO:
        // If offset_ + aligned_bytes <= capacity_:
        // return pointer at slab_ + offset, and advance offset_.
        // else return nullptr.
        return nullptr;
    }

    void reset() {
        offset_ = 0;
    }

    size_t allocated_bytes() const { return offset_; }
};

// ==============================================================================
// Exercise 2: Scoped CUDA Stream Context Switcher
// Binds an active stream to the current thread scope and provides it to callers.
// ==============================================================================
class ScopedStreamContext {
private:
    cudaStream_t stream_;

public:
    ScopedStreamContext() {
        // TODO: cudaStreamCreate(&stream_);
        stream_ = nullptr;
    }

    ~ScopedStreamContext() {
        // TODO: if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
    }

    cudaStream_t stream() const { return stream_; }
};

// ==============================================================================
// Exercise 3: Asynchronous Double-Buffering Execution Pipeline
// While buffer[0] is being computed on the GPU, buffer[1] is receiving
// the next chunk of data from host memory!
// ==============================================================================
__global__ void double_buffer_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 100.0f;
}

class DoubleBufferPipeline {
public:
    cudaStream_t streams[2];
    float* d_buffers[2];
    int N;

    DoubleBufferPipeline(int n) : N(n) {
        // TODO:
        // Create 2 streams and allocate 2 device buffers of size N * sizeof(float)
        streams[0] = nullptr; streams[1] = nullptr;
        d_buffers[0] = nullptr; d_buffers[1] = nullptr;
    }

    ~DoubleBufferPipeline() {
        // TODO: Synchronize streams, destroy streams, and free buffers
    }

    void execute_step(const float* h_chunk0, const float* h_chunk1,
                     float* h_out0, float* h_out1) {
        // TODO:
        // Async H2D on streams[0] and streams[1]
        // Launch double_buffer_kernel on both
        // Async D2H on both
        // Synchronize both streams
    }
};

// ==============================================================================
// Exercise 4: Statistical RAII Kernel Profiler
// Measures kernel execution across multiple iterations using cudaEvent_t.
// Computes mean, min, and max latency in milliseconds.
// ==============================================================================
struct ProfileStats {
    float mean_ms;
    float min_ms;
    float max_ms;
};

template <typename Func>
ProfileStats profile_kernel(Func kernel_launcher, int warmups, int runs) {
    ProfileStats stats = {0.0f, 0.0f, 0.0f};

    // Warmups
    for (int w = 0; w < warmups; ++w) {
        kernel_launcher();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measured runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times(runs);
    for (int r = 0; r < runs; ++r) {
        CHECK_CUDA(cudaEventRecord(start));
        kernel_launcher();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times[r] = ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // TODO:
    // Compute mean, min, and max of `times` and return in `stats`.
    return stats;
}

// ==============================================================================
// Exercise 5: CUDA Graph RAII Lifecycle Manager
// Encapsulates cudaGraph_t and cudaGraphExec_t lifecycle.
// Captures a stream into a graph, instantiates it, launches it, and cleans up.
// ==============================================================================
class CudaGraphRAII {
private:
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;

public:
    CudaGraphRAII() : graph_(nullptr), instance_(nullptr) {}

    ~CudaGraphRAII() {
        // TODO:
        // if (instance_) cudaGraphExecDestroy(instance_);
        // if (graph_) cudaGraphDestroy(graph_);
    }

    void begin_capture(cudaStream_t stream) {
        // TODO: cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }

    void end_capture(cudaStream_t stream) {
        // TODO:
        // cudaStreamEndCapture(stream, &graph_);
        // cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    }

    void launch(cudaStream_t stream) {
        // TODO:
        // if (instance_) cudaGraphLaunch(instance_, stream);
    }

    bool is_instantiated() const { return instance_ != nullptr; }
};

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Device Memory Arena Allocator
    // --------------------------------------------------------------------------
    {
        DeviceArena arena(1024 * 1024); // 1 MB
        void* p1 = arena.allocate(128);
        void* p2 = arena.allocate(256);

        if (p1 != nullptr && p2 != nullptr && p1 != p2 && arena.allocated_bytes() >= 384) {
            std::cout << "[Test 1: DeviceArena] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: DeviceArena] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: ScopedStreamContext
    // --------------------------------------------------------------------------
    {
        ScopedStreamContext ctx;
        if (ctx.stream() != nullptr) {
            std::cout << "[Test 2: ScopedStreamContext] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: ScopedStreamContext] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: DoubleBufferPipeline
    // --------------------------------------------------------------------------
    {
        int N = 64;
        DoubleBufferPipeline pipeline(N);
        std::vector<float> h0(N, 1.0f), h1(N, 2.0f);
        std::vector<float> out0(N, 0.0f), out1(N, 0.0f);

        pipeline.execute_step(h0.data(), h1.data(), out0.data(), out1.data());

        if (out0[0] == 101.0f && out1[0] == 102.0f) {
            std::cout << "[Test 3: DoubleBufferPipeline] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: DoubleBufferPipeline] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: Statistical RAII Kernel Profiler
    // --------------------------------------------------------------------------
    {
        float* d_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, 1024 * sizeof(float)));

        auto launcher = [&]() {
            double_buffer_kernel<<<4, 256>>>(d_buf, 1024);
        };

        ProfileStats stats = profile_kernel(launcher, 3, 5);
        if (stats.mean_ms > 0.0f && stats.min_ms > 0.0f && stats.max_ms >= stats.min_ms) {
            std::cout << "[Test 4: Statistical Profiler] PASSED (mean: "
                      << stats.mean_ms << " ms)" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Statistical Profiler] FAILED" << std::endl;
        }

        cudaFree(d_buf);
    }

    // --------------------------------------------------------------------------
    // Test 5: CUDA Graph RAII Lifecycle Manager
    // --------------------------------------------------------------------------
    {
        float* d_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, 64 * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_buf, 0, 64 * sizeof(float)));

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        CudaGraphRAII graph;
        graph.begin_capture(stream);
        double_buffer_kernel<<<1, 64, 0, stream>>>(d_buf, 64);
        graph.end_capture(stream);

        if (graph.is_instantiated()) {
            graph.launch(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));

            std::vector<float> h_out(64);
            CHECK_CUDA(cudaMemcpy(h_out.data(), d_buf, 64 * sizeof(float), cudaMemcpyDeviceToHost));
            if (h_out[0] == 100.0f) {
                std::cout << "[Test 5: CudaGraphRAII] PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "[Test 5: CudaGraphRAII] FAILED" << std::endl;
            }
        } else {
            std::cout << "[Test 5: CudaGraphRAII] FAILED" << std::endl;
        }

        cudaStreamDestroy(stream);
        cudaFree(d_buf);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
