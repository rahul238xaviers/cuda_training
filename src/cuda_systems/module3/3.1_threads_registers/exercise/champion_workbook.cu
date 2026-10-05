#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

void reportStatus(const std::string &name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 1: Vectorized 128-bit SiLU Activation (float4 with fast __expf)
// SiLU(x) = x / (1.0f + exp(-x))
// -----------------------------------------------------------------------------
__device__ inline float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void kernel_silu_float4(const float *in, float *out, int n) {
    // TODO: Process elements in chunks of 4 using float4
    // int n4 = n / 4;
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gid < n4) {
    //     float4 v = reinterpret_cast<const float4*>(in)[gid];
    //     v.x = fast_silu(v.x);
    //     v.y = fast_silu(v.y);
    //     v.z = fast_silu(v.z);
    //     v.w = fast_silu(v.w);
    //     reinterpret_cast<float4*>(out)[gid] = v;
    // }
    // TODO: Handle tail/epilogue elements
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Register-Shift Window (3-tap Moving Average in Registers)
// out[i] = (in[i-1] + in[i] + in[i+1]) / 3.0f
// -----------------------------------------------------------------------------
__global__ void kernel_register_window_filter(const float *in, float *out, int n) {
    // Each thread processes a sequence of elements using 3 sliding registers: r_prev, r_curr, r_next
    // avoiding redundant global loads.
    // TODO: Implement sliding window logic per thread
}

// -----------------------------------------------------------------------------
// PROBLEM 3: 2D Grid-Stride Matrix Scaling with Register Reuse
// -----------------------------------------------------------------------------
__global__ void kernel_2d_grid_stride(const float *in, float *out, float scale, int rows, int cols) {
    // Write a 2D grid-stride loop that can handle matrices of ANY size (e.g. 10000x10000)
    // regardless of launched grid/block dimensions.
    // TODO: for (int r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += gridDim.y * blockDim.y)
    // TODO:     for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
    // TODO:         out[r * cols + c] = in[r * cols + c] * scale;
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Thread-Private Register Histogram (4 Quadrant Bins)
// Count occurrences of values in ranges: [0, 25), [25, 50), [50, 75), [75, 100)
// -----------------------------------------------------------------------------
__global__ void kernel_register_histogram_bins(const float *in, int *out_bins, int n) {
    // Each thread maintains 4 register counters (b0, b1, b2, b3)
    // Processes a chunk of inputs, then writes to thread's dedicated output slot (thread_id * 4 + bin)
    // TODO: Initialize register bins to 0
    // TODO: Loop through assigned elements and increment register bins
    // TODO: Write out_bins[tid * 4 + 0..3]
}

// -----------------------------------------------------------------------------
// PROBLEM 5: In-Register 4-Way Extremum (Fused Min & Max Tracker)
// -----------------------------------------------------------------------------
__global__ void kernel_register_min_max_4way(const float *in, float *out_min, float *out_max, int n) {
    // Thread reads 4 elements into registers using float4, computes local min and max
    // entirely using fminf / fmaxf in registers, and stores to out_min and out_max.
    // TODO: Implement vectorized load and in-register min/max tracking
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.1 Threads & Registers (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: SiLU float4
    {
        int n = 2049; // Non-multiple of 4
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = -3.0f + (float)i * (6.0f / (float)n);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = ((n + 3) / 4 + blockSize - 1) / blockSize;
        kernel_silu_float4<<<numBlocks, blockSize>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            float exp_val = h_in[i] / (1.0f + std::exp(-h_in[i]));
            if (std::fabs(h_out[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 1: Vectorized 128-bit SiLU Activation (float4)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Register-Shift Moving Average
    {
        int n = 128;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)(i * 3);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_register_window_filter<<<1, 1>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 1; i < n - 1; ++i) {
            float expected = (h_in[i - 1] + h_in[i] + h_in[i + 1]) / 3.0f;
            if (std::fabs(h_out[i] - expected) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: Register-Shift Window (3-tap Moving Average)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: 2D Grid-Stride Loop
    {
        int rows = 128, cols = 256;
        int n = rows * cols;
        std::vector<float> h_in(n, 1.5f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(4, 4); // Intentionally smaller than matrix to force stride
        kernel_2d_grid_stride<<<grid, block>>>(d_in, d_out, 4.0f, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 6.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: 2D Grid-Stride Matrix Scaling with Register Reuse", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Register Histogram Bins
    {
        int n = 400;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 100);
        int threads = 4;
        std::vector<int> h_bins(threads * 4, 0);
        float *d_in;
        int *d_bins;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_bins, threads * 4 * sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_register_histogram_bins<<<1, threads>>>(d_in, d_bins, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_bins.data(), d_bins, threads * 4 * sizeof(int), cudaMemcpyDeviceToHost);
        int total_b0 = 0, total_b1 = 0, total_b2 = 0, total_b3 = 0;
        for (int t = 0; t < threads; ++t) {
            total_b0 += h_bins[t * 4 + 0];
            total_b1 += h_bins[t * 4 + 1];
            total_b2 += h_bins[t * 4 + 2];
            total_b3 += h_bins[t * 4 + 3];
        }
        bool ok = (total_b0 == 100 && total_b1 == 100 && total_b2 == 100 && total_b3 == 100);
        reportStatus("Problem 4: Thread-Private Register Histogram (4 Bins)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_bins);
    }

    // Test 5: Register Min/Max 4-Way
    {
        int n = 256;
        int num_chunks = n / 4;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 17) - 8.0f;
        std::vector<float> h_min(num_chunks, 0.0f), h_max(num_chunks, 0.0f);

        float *d_in, *d_min, *d_max;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_min, num_chunks * sizeof(float));
        cudaMalloc(&d_max, num_chunks * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_register_min_max_4way<<<1, num_chunks>>>(d_in, d_min, d_max, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_min.data(), d_min, num_chunks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max.data(), d_max, num_chunks * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = true;
        for (int i = 0; i < num_chunks; ++i) {
            float exp_min = std::min({h_in[4*i], h_in[4*i+1], h_in[4*i+2], h_in[4*i+3]});
            float exp_max = std::max({h_in[4*i], h_in[4*i+1], h_in[4*i+2], h_in[4*i+3]});
            if (std::fabs(h_min[i] - exp_min) > 1e-4f || std::fabs(h_max[i] - exp_max) > 1e-4f) {
                ok = false; break;
            }
        }
        reportStatus("Problem 5: In-Register 4-Way Extremum (Min/Max Tracker)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_min); cudaFree(d_max);
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
