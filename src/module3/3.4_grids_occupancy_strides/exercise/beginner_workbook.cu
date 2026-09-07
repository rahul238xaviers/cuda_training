#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void reportStatus(const std::string &name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 1: Canonical 1D Grid-Stride Loop
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_1d(const float *in, float *out, float scale, int n) {
    // TODO: Write a 1D grid-stride loop:
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    //     out[i] = in[i] * scale;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Grid Stride Step Identification
// -----------------------------------------------------------------------------
__global__ void kernel_store_grid_stride(int *out_stride) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // TODO: Calculate the total number of threads in the entire grid:
        // *out_stride = gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: 2D Grid-Stride Loop for Image / Matrix Transform
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_2d(const float *in, float *out, int rows, int cols) {
    // TODO: Nested grid-stride loop in 2D
    // for (int r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += gridDim.y * blockDim.y) {
    //     for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) {
    //         out[r * cols + c] = in[r * cols + c] + 1.0f;
    //     }
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Vectorized float4 Grid-Stride Loop
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_float4(const float *in, float *out, int n) {
    // int n4 = n / 4;
    // const float4 *in4 = reinterpret_cast<const float4*>(in);
    // float4 *out4 = reinterpret_cast<float4*>(out);
    // TODO: Grid-stride loop over 4-element vectors
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    //     float4 v = in4[i];
    //     v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
    //     out4[i] = v;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Strided Batch Accumulator
// -----------------------------------------------------------------------------
__global__ void kernel_strided_batch_add(const float *batch_a, const float *batch_b, float *batch_c, int total_elements) {
    // TODO: Grid-stride loop adding batch_a and batch_b into batch_c
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
    //     batch_c[i] = batch_a[i] + batch_b[i];
    // }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.4 Grids & Grid-Stride Loops (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: 1D Grid Stride (Few blocks handling large array)
    {
        int n = 10000;
        std::vector<float> h_in(n, 3.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        // Only 4 blocks of 64 threads = 256 threads handling 10000 elements!
        kernel_grid_stride_1d<<<4, 64>>>(d_in, d_out, 5.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 15.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 1: Canonical 1D Grid-Stride Loop (Few Blocks -> Large Array)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Stride Step
    {
        int *d_stride;
        cudaMalloc(&d_stride, sizeof(int));
        kernel_store_grid_stride<<<8, 128>>>(d_stride);
        cudaDeviceSynchronize();

        int h_stride = 0;
        cudaMemcpy(&h_stride, d_stride, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_stride == (8 * 128));
        reportStatus("Problem 2: Grid Stride Step Identification", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_stride);
    }

    // Test 3: 2D Grid Stride
    {
        int rows = 100, cols = 150, n = rows * cols;
        std::vector<float> h_in(n, 2.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(2, 2); // Only 4 blocks handling 15,000 matrix elements
        kernel_grid_stride_2d<<<grid, block>>>(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 3.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: 2D Grid-Stride Loop for Arbitrary Matrices", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: float4 Grid Stride
    {
        int n = 4096;
        std::vector<float> h_in(n, 4.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_grid_stride_float4<<<2, 128>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 8.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 4: Vectorized float4 Grid-Stride Loop", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Strided Batch Add
    {
        int n = 20000;
        std::vector<float> h_a(n, 1.5f), h_b(n, 2.5f), h_c(n, 0.0f);
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_strided_batch_add<<<4, 256>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_c[i] - 4.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 5: Strided Batch Array Addition", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
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
