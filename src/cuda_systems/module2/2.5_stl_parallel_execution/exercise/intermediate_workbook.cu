// ==============================================================================
// Module 2.5: Host Parallel Execution and Asynchronous Pipelines
// Level: Intermediate Workbook
//
// Focus:
// 1. Multi-Threaded Host RMSNorm implementation (CPU baseline).
// 2. GPU RMSNorm CUDA Kernel with block reduction.
// 3. Pinned Host Memory RAII Buffer (cudaMallocHost / cudaFreeHost).
// 4. Asynchronous Two-Stream Host-to-Device Memory & Compute Overlap.
// 5. High-Precision GPU Event Timer (cudaEvent_t).
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <cmath>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// ==============================================================================
// Exercise 1: Multi-Threaded CPU RMSNorm
// y[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
// Parallelizes the scaling step across `num_threads`.
// ==============================================================================
void cpu_rmsnorm_parallel(const float* x, const float* weight, float* out,
                          size_t n, float eps, size_t num_threads) {
    // TODO:
    // 1. Compute sum of squares: sum(x[i]^2) across all n elements
    // 2. rms = sqrt(sum_sq / n + eps)
    // 3. In parallel across num_threads, compute out[i] = (x[i] / rms) * weight[i]
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 2: GPU Block-Cooperative RMSNorm Kernel
// A single thread-block (e.g. 256 threads) normalizes a vector of size n <= 1024.
// Shared memory accumulates the sum of squares.
// ==============================================================================
__global__ void gpu_rmsnorm_kernel(const float* x, const float* weight, float* out,
                                   size_t n, float eps) {
    extern __shared__ float sdata[];
    // TODO:
    // 1. Each thread computes local sum of squares across strided elements
    // 2. Perform block-level tree reduction in shared memory sdata
    // 3. Thread 0 calculates inv_rms = rsqrtf(sdata[0] / n + eps)
    // 4. __syncthreads()
    // 5. Each thread writes out[i] = x[i] * inv_rms * weight[i]
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 3: Pinned Host Memory RAII Buffer
// Allocates page-locked host memory using cudaMallocHost and frees with cudaFreeHost.
// ==============================================================================
template <typename T>
class PinnedBuffer {
private:
    T* data_{nullptr};
    size_t size_{0};

public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(size_t n) : size_(n) {
        if (size_ > 0) {
            // TODO: Exercise 3a: Allocate pinned host memory
            // CHECK_CUDA(cudaMallocHost(&data_, size_ * sizeof(T)));
            // [YOUR CODE HERE]
        }
    }

    ~PinnedBuffer() {
        if (data_) {
            // TODO: Exercise 3b: Free pinned host memory
            // cudaFreeHost(data_);
            // [YOUR CODE HERE]
            data_ = nullptr;
        }
    }

    // Disable copy
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Move semantics
    PinnedBuffer(PinnedBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFreeHost(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    T& operator[](size_t i) { return data_[i]; }
};

// ==============================================================================
// Exercise 4: Asynchronous Two-Stream Host-to-Device Memory & Compute Overlap
// Copies chunk B to device on stream_copy while kernel operates on chunk A on stream_compute.
// ==============================================================================
__global__ void dummy_compute_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale;
    }
}

void execute_overlapped_pipeline(const float* pinned_chunk_b, float* d_chunk_b,
                                float* d_chunk_a, size_t chunk_size,
                                cudaStream_t stream_compute, cudaStream_t stream_copy) {
    // TODO:
    // 1. Launch dummy_compute_kernel on stream_compute processing d_chunk_a
    // 2. Simultaneously launch cudaMemcpyAsync of pinned_chunk_b into d_chunk_b on stream_copy
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 5: High-Precision GPU Event Timer
// Measures elapsed GPU time in milliseconds using cudaEvent_t.
// ==============================================================================
class GpuTimer {
private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};

public:
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        // TODO: Record start_ event on stream
        // [YOUR CODE HERE]
    }

    void stop(cudaStream_t stream = 0) {
        // TODO: Record stop_ event on stream
        // [YOUR CODE HERE]
    }

    float elapsed_ms() {
        // TODO: Synchronize stop_ event and compute elapsed milliseconds
        // [YOUR CODE HERE]
        return 0.0f;
    }
};

// ==============================================================================
// Verification Test Harness
// ==============================================================================
bool test_cpu_rmsnorm() {
    const size_t N = 1024;
    std::vector<float> x(N, 2.0f);
    std::vector<float> w(N, 1.0f);
    std::vector<float> out(N, 0.0f);

    cpu_rmsnorm_parallel(x.data(), w.data(), out.data(), N, 1e-5f, 4);

    // mean(x^2) = 4.0. sqrt(4.0) = 2.0. x / rms = 2.0 / 2.0 = 1.0.
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(out[i] - 1.0f) > 1e-3f) {
            return false;
        }
    }
    return true;
}

bool test_gpu_rmsnorm() {
    const size_t N = 512;
    std::vector<float> h_x(N, 3.0f);
    std::vector<float> h_w(N, 2.0f);
    std::vector<float> h_out(N, 0.0f);

    float *d_x, *d_w, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    gpu_rmsnorm_kernel<<<1, 256, 256 * sizeof(float)>>>(d_x, d_w, d_out, N, 1e-5f);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);

    // mean(x^2) = 9.0. rms = 3.0. out[i] = (3.0 / 3.0) * 2.0 = 2.0.
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_out[i] - 2.0f) > 1e-3f) {
            return false;
        }
    }
    return true;
}

bool test_pinned_buffer() {
    PinnedBuffer<float> buf(1024);
    if (buf.data() == nullptr || buf.size() != 1024) return false;

    // Check pointer attributes to verify it is indeed pinned (cudaMemoryTypeHost)
    cudaPointerAttributes attr{};
    CHECK_CUDA(cudaPointerGetAttributes(&attr, buf.data()));
    
    // In CUDA 12: type == cudaMemoryTypeHost or cudaMemoryTypeUnregistered
    // A pinned buffer allocated via cudaMallocHost has type == cudaMemoryTypeHost
    if (attr.type != cudaMemoryTypeHost) return false;

    buf[0] = 42.0f;
    buf[1023] = 99.0f;
    return (buf[0] == 42.0f && buf[1023] == 99.0f);
}

bool test_overlapped_pipeline() {
    cudaStream_t s_comp, s_copy;
    CHECK_CUDA(cudaStreamCreate(&s_comp));
    CHECK_CUDA(cudaStreamCreate(&s_copy));

    const size_t N = 2048;
    PinnedBuffer<float> host_b(N);
    if (host_b.data() == nullptr) {
        cudaStreamDestroy(s_comp);
        cudaStreamDestroy(s_copy);
        return false;
    }
    for (size_t i = 0; i < N; ++i) host_b[i] = 10.0f;

    float *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));

    std::vector<float> init_a(N, 5.0f);
    CHECK_CUDA(cudaMemcpy(d_a, init_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_b, 0, N * sizeof(float)));

    execute_overlapped_pipeline(host_b.data(), d_b, d_a, N, s_comp, s_copy);

    CHECK_CUDA(cudaStreamSynchronize(s_comp));
    CHECK_CUDA(cudaStreamSynchronize(s_copy));

    std::vector<float> res_a(N), res_b(N);
    CHECK_CUDA(cudaMemcpy(res_a.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(res_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaStreamDestroy(s_comp);
    cudaStreamDestroy(s_copy);

    if (res_a[0] != 15.0f) return false; // 5.0 * 3.0
    if (res_b[0] != 10.0f) return false;

    return true;
}

bool test_gpu_timer() {
    GpuTimer timer;
    float* d_buf;
    const size_t N = 1000000;
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(float)));

    timer.start();
    dummy_compute_kernel<<<256, 256>>>(d_buf, N, 2.0f);
    timer.stop();

    float ms = timer.elapsed_ms();
    cudaFree(d_buf);

    return (ms > 0.0001f && ms < 1000.0f);
}

int main() {
    int passed = 0;

    if (test_cpu_rmsnorm()) {
        std::cout << "[Test 1: Multi-Threaded CPU RMSNorm] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 1: Multi-Threaded CPU RMSNorm] FAILED\n";
    }

    if (test_gpu_rmsnorm()) {
        std::cout << "[Test 2: GPU Cooperative RMSNorm Kernel] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 2: GPU Cooperative RMSNorm Kernel] FAILED\n";
    }

    if (test_pinned_buffer()) {
        std::cout << "[Test 3: Pinned Host Memory Buffer] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 3: Pinned Host Memory Buffer] FAILED\n";
    }

    if (test_overlapped_pipeline()) {
        std::cout << "[Test 4: Asynchronous Multi-Stream Overlap] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 4: Asynchronous Multi-Stream Overlap] FAILED\n";
    }

    if (test_gpu_timer()) {
        std::cout << "[Test 5: High-Precision GPU Event Timer] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 5: High-Precision GPU Event Timer] FAILED\n";
    }

    std::cout << "Passed: " << passed << " / 5 tests.\n";
    return (passed == 5) ? 0 : 1;
}
