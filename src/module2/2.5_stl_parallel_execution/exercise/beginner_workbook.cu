// ==============================================================================
// Module 2.5: Host Parallel Execution and Asynchronous Pipelines
// Level: Beginner Workbook
//
// Focus:
// 1. Multi-threaded range partitioning (`parallel_for`) using std::thread.
// 2. Parallel reduction (`parallel_sum`) across host CPU threads.
// 3. Parallel map-reduce (`parallel_l2_norm_sq`) computing sum of squares.
// 4. CUDA Vector Addition Kernel baseline.
// 5. High-resolution CPU Timer using std::chrono.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <chrono>
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
// Exercise 1: Multi-Threaded parallel_for
// Splits range [0, n) across `num_threads` worker threads.
// Each thread invokes `fn(i)` for each element in its assigned range.
// ==============================================================================
template <typename Func>
void parallel_for(size_t n, size_t num_threads, Func&& fn) {
    // TODO:
    // 1. Calculate chunk size = (n + num_threads - 1) / num_threads
    // 2. Launch workers using std::thread
    // 3. Join all threads before returning
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 2: Multi-Threaded parallel_sum
// Divides `data` across `num_threads`. Each thread computes local sum of its chunk.
// Aggregates local sums and returns total sum.
// ==============================================================================
float parallel_sum(const std::vector<float>& data, size_t num_threads) {
    // TODO:
    // 1. Divide data into chunks
    // 2. Use std::async or std::thread to compute local sums
    // 3. Accumulate and return total
    // [YOUR CODE HERE]
    return 0.0f;
}

// ==============================================================================
// Exercise 3: Multi-Threaded parallel_l2_norm_sq
// Computes sum of squares: sum(x_i * x_i) across all elements in parallel.
// ==============================================================================
float parallel_l2_norm_sq(const std::vector<float>& data, size_t num_threads) {
    // TODO:
    // Compute sum of squares in parallel across num_threads
    // [YOUR CODE HERE]
    return 0.0f;
}

// ==============================================================================
// Exercise 4: GPU Vector Add Kernel
// Elementwise addition: out[idx] = a[idx] + b[idx]
// ==============================================================================
__global__ void vector_add_kernel(const float* a, const float* b, float* out, size_t n) {
    // TODO:
    // Compute 1D thread index idx.
    // If idx < n: out[idx] = a[idx] + b[idx]
    // [YOUR CODE HERE]
}

// ==============================================================================
// Exercise 5: High-Resolution CPU Timer
// Measures elapsed microseconds between start() and stop().
// ==============================================================================
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
    bool stopped_{false};

public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        stopped_ = false;
    }

    void stop() {
        stop_time_ = std::chrono::high_resolution_clock::now();
        stopped_ = true;
    }

    // Returns elapsed time in microseconds
    double elapsed_microseconds() const {
        // TODO:
        // Return duration in microseconds as double
        // [YOUR CODE HERE]
        return 0.0;
    }
};

// ==============================================================================
// Verification Test Harness
// ==============================================================================
bool test_parallel_for() {
    const size_t N = 10000;
    std::vector<int> arr(N, 0);

    parallel_for(N, 4, [&](size_t i) {
        arr[i] = static_cast<int>(i * 2);
    });

    for (size_t i = 0; i < N; ++i) {
        if (arr[i] != static_cast<int>(i * 2)) {
            return false;
        }
    }
    return true;
}

bool test_parallel_sum() {
    const size_t N = 100000;
    std::vector<float> arr(N, 1.5f);

    float sum = parallel_sum(arr, 4);
    float expected = N * 1.5f;

    return std::abs(sum - expected) < 1e-1f;
}

bool test_parallel_l2_norm_sq() {
    const size_t N = 50000;
    std::vector<float> arr(N, 2.0f); // 2^2 = 4

    float l2 = parallel_l2_norm_sq(arr, 4);
    float expected = N * 4.0f;

    return std::abs(l2 - expected) < 1e-1f;
}

bool test_vector_add_kernel() {
    const size_t N = 1024;
    std::vector<float> h_a(N, 1.25f);
    std::vector<float> h_b(N, 2.75f);
    std::vector<float> h_out(N, 0.0f);

    float *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

    int block = 256;
    int grid = (N + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_out[i] - 4.0f) > 1e-4f) {
            return false;
        }
    }
    return true;
}

bool test_cpu_timer() {
    CpuTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();

    double us = timer.elapsed_microseconds();
    // 10 ms = 10,000 us. Check reasonable window (8,000 to 50,000 us)
    return (us >= 8000.0 && us <= 50000.0);
}

int main() {
    int passed = 0;

    if (test_parallel_for()) {
        std::cout << "[Test 1: Multi-Threaded parallel_for] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 1: Multi-Threaded parallel_for] FAILED\n";
    }

    if (test_parallel_sum()) {
        std::cout << "[Test 2: Multi-Threaded parallel_sum] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 2: Multi-Threaded parallel_sum] FAILED\n";
    }

    if (test_parallel_l2_norm_sq()) {
        std::cout << "[Test 3: Multi-Threaded parallel_l2_norm_sq] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 3: Multi-Threaded parallel_l2_norm_sq] FAILED\n";
    }

    if (test_vector_add_kernel()) {
        std::cout << "[Test 4: GPU Vector Add Kernel] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 4: GPU Vector Add Kernel] FAILED\n";
    }

    if (test_cpu_timer()) {
        std::cout << "[Test 5: High-Resolution CPU Timer] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 5: High-Resolution CPU Timer] FAILED\n";
    }

    std::cout << "Passed: " << passed << " / 5 tests.\n";
    return (passed == 5) ? 0 : 1;
}
