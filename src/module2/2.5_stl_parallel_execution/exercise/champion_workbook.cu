// ==============================================================================
// Module 2.5: Host Parallel Execution and Asynchronous Pipelines
// Level: Champion Workbook
//
// Focus:
// 1. Concurrent Multi-Stream Transformer Stage Dispatch.
// 2. Stream-to-Stream Event Synchronization (cudaStreamWaitEvent) without Host Stalls.
// 3. High-Throughput Asynchronous Host-to-Device Pinned DMA Bandwidth Profiler.
// 4. Asynchronous Host Pre-processing Pipeline overlapping CPU and GPU execution.
// 5. Analytical CPU Parallel Ground-Truth vs GPU Kernel Verification (Fused Linear+GELU).
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <cmath>
#include <chrono>

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
// Exercise 1: Multi-Stream Independent Kernel Dispatch
// Given two independent streams and two distinct device buffers, dispatches
// `scale_kernel` concurrently to both streams so SMs execute both simultaneously.
// ==============================================================================
__global__ void scale_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

void launch_dual_stream_kernels(float* d_buf1, float* d_buf2, size_t n,
                                cudaStream_t stream1, cudaStream_t stream2,
                                float scale1, float scale2) {
    // TODO:
    // 1. Launch scale_kernel on stream1 for d_buf1
    // 2. Launch scale_kernel on stream2 for d_buf2
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 2: Inter-Stream Event Synchronization
// Stream 2 must wait for Stream 1 to finish an initialization kernel before
// proceeding, WITHOUT stalling the host CPU (do not call cudaStreamSynchronize).
// Uses cudaEventRecord and cudaStreamWaitEvent.
// Returns cudaSuccess upon success.
// ==============================================================================
cudaError_t record_and_wait_dependency(cudaStream_t stream_producer,
                                       cudaStream_t stream_consumer,
                                       cudaEvent_t event) {
    // TODO:
    // 1. Record event in stream_producer via cudaEventRecord
    // 2. Make stream_consumer wait for event via cudaStreamWaitEvent
    // 3. Return cudaSuccess
    // [YOUR CODE HERE]
    return cudaErrorNotReady;
}

// ==============================================================================
// Exercise 3: High-Throughput Pinned DMA Bandwidth Profiler
// Measures effective Host-to-Device transfer bandwidth in Gigabytes per second (GB/s).
// Formula: Bandwidth (GB/s) = (bytes / 1e9) / (elapsed_ms / 1000.0)
// ==============================================================================
float profile_h2d_bandwidth(const float* h_pinned, float* d_out, size_t num_elements,
                            cudaStream_t stream) {
    // TODO:
    // 1. Create start and stop cudaEvent_t
    // 2. Record start on stream
    // 3. Issue cudaMemcpyAsync from h_pinned to d_out on stream
    // 4. Record stop on stream
    // 5. Synchronize stop event, compute elapsed_ms, and return bandwidth in GB/s
    // 6. Clean up events
    // [YOUR CODE HERE]
    return 0.0f;
}

// ==============================================================================
// Exercise 4: Asynchronous CPU-GPU Pipeline Worker
// While the GPU is computing batch K on stream_gpu, an asynchronous CPU thread
// runs a CPU tokenization/pre-processing function on batch K+1.
// Returns a std::future containing the pre-processed CPU results.
// ==============================================================================
template <typename PreprocessFunc>
std::future<std::vector<float>> async_cpu_stage(PreprocessFunc&& fn,
                                               const std::vector<float>& raw_input) {
    // TODO:
    // Launch fn asynchronously on a background thread via std::async(std::launch::async, ...)
    // Return the resulting std::future
    // [YOUR CODE HERE]
    return std::async(std::launch::deferred, []() { return std::vector<float>(); });
}

// ==============================================================================
// Exercise 5: Fused Linear + GELU Verification (CPU Analytical vs GPU Kernel)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Fused op: y = GELU(x * weight + bias)
// ==============================================================================
__host__ __device__ inline float gelu_activation(float x) {
    const float k0 = 0.7978845608f; // sqrt(2 / pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gpu_fused_linear_gelu(const float* x, float weight, float bias,
                                     float* out, size_t n) {
    // TODO:
    // Compute thread idx.
    // If idx < n:
    //   float val = x[idx] * weight + bias;
    //   out[idx] = gelu_activation(val);
    // [YOUR CODE HERE]
}

void cpu_fused_linear_gelu_parallel(const float* x, float weight, float bias,
                                    float* out, size_t n, size_t num_threads) {
    // TODO:
    // Multi-threaded CPU evaluation of the identical fused operation
    // [YOUR CODE HERE]
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
bool test_dual_stream_launch() {
    cudaStream_t s1, s2;
    CHECK_CUDA(cudaStreamCreate(&s1));
    CHECK_CUDA(cudaStreamCreate(&s2));

    const size_t N = 1024;
    float *d1, *d2;
    CHECK_CUDA(cudaMalloc(&d1, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d2, N * sizeof(float)));

    std::vector<float> h_init(N, 2.0f);
    CHECK_CUDA(cudaMemcpy(d1, h_init.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d2, h_init.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch_dual_stream_kernels(d1, d2, N, s1, s2, 3.0f, 5.0f);

    CHECK_CUDA(cudaStreamSynchronize(s1));
    CHECK_CUDA(cudaStreamSynchronize(s2));

    std::vector<float> res1(N), res2(N);
    CHECK_CUDA(cudaMemcpy(res1.data(), d1, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(res2.data(), d2, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d1);
    cudaFree(d2);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(res1[i] - 6.0f) > 1e-4f || std::abs(res2[i] - 10.0f) > 1e-4f) {
            return false;
        }
    }
    return true;
}

bool test_inter_stream_sync() {
    cudaStream_t s_prod, s_cons;
    cudaEvent_t evt;
    CHECK_CUDA(cudaStreamCreate(&s_prod));
    CHECK_CUDA(cudaStreamCreate(&s_cons));
    CHECK_CUDA(cudaEventCreate(&evt));

    const size_t N = 512;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    std::vector<float> init(N, 1.0f);
    CHECK_CUDA(cudaMemcpyAsync(d_data, init.data(), N * sizeof(float), cudaMemcpyHostToDevice, s_prod));

    // Producer multiplies by 2
    int block = 256;
    int grid = (N + block - 1) / block;
    scale_kernel<<<grid, block, 0, s_prod>>>(d_data, N, 2.0f);

    // Sync dependency
    cudaError_t sync_err = record_and_wait_dependency(s_prod, s_cons, evt);
    if (sync_err != cudaSuccess) {
        cudaFree(d_data);
        cudaEventDestroy(evt);
        cudaStreamDestroy(s_prod);
        cudaStreamDestroy(s_cons);
        return false;
    }

    // Consumer multiplies by 3
    scale_kernel<<<grid, block, 0, s_cons>>>(d_data, N, 3.0f);

    CHECK_CUDA(cudaStreamSynchronize(s_cons));

    std::vector<float> res(N);
    CHECK_CUDA(cudaMemcpy(res.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaEventDestroy(evt);
    cudaStreamDestroy(s_prod);
    cudaStreamDestroy(s_cons);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(res[i] - 6.0f) > 1e-4f) return false;
    }
    return true;
}

bool test_bandwidth_profiler() {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const size_t N = 10000000; // ~40 MB
    float* h_pinned;
    CHECK_CUDA(cudaMallocHost(&h_pinned, N * sizeof(float)));
    for (size_t i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    float bw = profile_h2d_bandwidth(h_pinned, d_out, N, stream);

    cudaFreeHost(h_pinned);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    // PCIe 3.0/4.0/5.0 bandwidth typically between 1.0 GB/s and 100.0 GB/s
    return (bw > 0.5f && bw < 150.0f);
}

bool test_async_pipeline() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto future_res = async_cpu_stage([](const std::vector<float>& raw) {
        std::vector<float> out(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            out[i] = raw[i] * 10.0f;
        }
        return out;
    }, input);

    std::vector<float> result = future_res.get();
    if (result.size() != 5) return false;
    return (result[0] == 10.0f && result[4] == 50.0f);
}

bool test_fused_linear_gelu() {
    const size_t N = 2048;
    std::vector<float> h_x(N);
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = -2.0f + 4.0f * (static_cast<float>(i) / N); // range [-2, 2]
    }
    float weight = 1.5f;
    float bias = -0.25f;

    // CPU analytical reference
    std::vector<float> cpu_out(N, 0.0f);
    cpu_fused_linear_gelu_parallel(h_x.data(), weight, bias, cpu_out.data(), N, 4);

    // GPU evaluation
    float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;
    gpu_fused_linear_gelu<<<grid, block>>>(d_x, weight, bias, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_out(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_out);

    // Ensure non-trivial results were actually computed
    if (cpu_out[N - 1] < 1.0f || gpu_out[N - 1] < 1.0f) {
        return false;
    }

    // Validate that CPU and GPU results match within 1e-4 tolerance
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(cpu_out[i] - gpu_out[i]) > 1e-4f || std::isnan(gpu_out[i])) {
            return false;
        }
    }
    return true;
}

int main() {
    int passed = 0;

    if (test_dual_stream_launch()) {
        std::cout << "[Test 1: Dual-Stream Kernel Dispatch] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 1: Dual-Stream Kernel Dispatch] FAILED\n";
    }

    if (test_inter_stream_sync()) {
        std::cout << "[Test 2: Inter-Stream Event Synchronization] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 2: Inter-Stream Event Synchronization] FAILED\n";
    }

    if (test_bandwidth_profiler()) {
        std::cout << "[Test 3: Pinned DMA Bandwidth Profiler] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 3: Pinned DMA Bandwidth Profiler] FAILED\n";
    }

    if (test_async_pipeline()) {
        std::cout << "[Test 4: Asynchronous CPU-GPU Pipeline Worker] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 4: Asynchronous CPU-GPU Pipeline Worker] FAILED\n";
    }

    if (test_fused_linear_gelu()) {
        std::cout << "[Test 5: Fused Linear + GELU Verification] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 5: Fused Linear + GELU Verification] FAILED\n";
    }

    std::cout << "Passed: " << passed << " / 5 tests.\n";
    return (passed == 5) ? 0 : 1;
}
