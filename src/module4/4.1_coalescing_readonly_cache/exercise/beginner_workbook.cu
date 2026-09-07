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
// PROBLEM 1: Coalesced 1D Vector Scale
// -----------------------------------------------------------------------------
__global__ void kernel_coalesced_scale(const float *in, float *out, float scale, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Coalesced memory read and write with bounds check
    // if (gid < n) { out[gid] = in[gid] * scale; }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Fixing Strided Uncoalesced Access into Coalesced Access
// In row-major matrix [rows, cols], reading col-by-col is uncoalesced.
// Fix this so threads in a warp access contiguous columns (row-by-row).
// -----------------------------------------------------------------------------
__global__ void kernel_coalesced_matrix_copy(const float *in, float *out, int rows, int cols) {
    // Map threadIdx.x to column dimension to ensure adjacent threads read adjacent floats!
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO: if (row < rows && col < cols) {
    //     int idx = row * cols + col;
    //     out[idx] = in[idx] + 5.0f;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Explicit Read-Only Cache Load via __ldg()
// -----------------------------------------------------------------------------
__global__ void kernel_ldg_load(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO: Use float val = __ldg(&in[gid]);
        // out[gid] = val * 3.0f;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Constant Memory Broadcasting (__constant__)
// -----------------------------------------------------------------------------
__constant__ float c_weights[4];

__global__ void kernel_constant_broadcast(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // All threads read identical weights from constant cache
        // TODO: Compute out[gid] = in[gid] * (c_weights[0] + c_weights[1] + c_weights[2] + c_weights[3]);
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: 128-bit Aligned Vectorized Memory Read (float4)
// -----------------------------------------------------------------------------
__global__ void kernel_aligned_float4(const float *in, float *out, int n4) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Load using float4 from reinterpret_cast<const float4*>(in)[gid]
    // Add 1.0f to each component and write to reinterpret_cast<float4*>(out)[gid]
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.1 Coalescing & Read-Only Cache (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Coalesced Scale
    {
        int n = 1024;
        std::vector<float> h_in(n, 2.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_coalesced_scale<<<(n + 255) / 256, 256>>>(d_in, d_out, 3.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 6.0f) { ok = false; break; }
        reportStatus("Problem 1: Coalesced 1D Vector Scale", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Coalesced Matrix Copy
    {
        int rows = 32, cols = 64, n = rows * cols;
        std::vector<float> h_in(n, 10.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 8);
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        kernel_coalesced_matrix_copy<<<grid, block>>>(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 15.0f) { ok = false; break; }
        reportStatus("Problem 2: Fixing Strided Access into Coalesced Matrix Memory", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: __ldg Load
    {
        int n = 512;
        std::vector<float> h_in(n, 4.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_ldg_load<<<2, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 12.0f) { ok = false; break; }
        reportStatus("Problem 3: Explicit Read-Only Cache Load via __ldg()", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Constant Memory
    {
        int n = 256;
        float host_w[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // sum = 10
        cudaMemcpyToSymbol(c_weights, host_w, 4 * sizeof(float));

        std::vector<float> h_in(n, 5.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_constant_broadcast<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 50.0f) { ok = false; break; }
        reportStatus("Problem 4: Constant Memory Broadcasting (__constant__)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Aligned float4
    {
        int n4 = 256;
        int n = n4 * 4;
        std::vector<float> h_in(n, 7.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_aligned_float4<<<1, n4>>>(d_in, d_out, n4);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 8.0f) { ok = false; break; }
        reportStatus("Problem 5: 128-bit Aligned Vectorized Memory Read (float4)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
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
