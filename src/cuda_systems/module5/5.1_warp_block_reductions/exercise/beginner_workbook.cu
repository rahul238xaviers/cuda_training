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

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// -----------------------------------------------------------------------------
// PROBLEM 1: 256-Thread Block Sum Reduction
// -----------------------------------------------------------------------------
__global__ void kernel_block_sum_reduce(const float *in, float *out, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? in[gid] : 0.0f;
    // TODO: 1. Reduce in warp: float w_sum = warp_reduce_sum(val);
    // TODO: 2. if (lane == 0) s_scratch[warp_id] = w_sum;
    // TODO: 3. __syncthreads();
    // TODO: 4. if (warp_id == 0) final warp reduction of s_scratch
    // TODO: 5. if (lane == 0) out[blockIdx.x] = block_sum;
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 256-Thread Block Max Reduction
// -----------------------------------------------------------------------------
__global__ void kernel_block_max_reduce(const float *in, float *out, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? in[gid] : -1e30f;
    // TODO: Two-pass block reduction finding maximum value
}

// -----------------------------------------------------------------------------
// PROBLEM 3: 256-Thread Block Min Reduction
// -----------------------------------------------------------------------------
__global__ void kernel_block_min_reduce(const float *in, float *out, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? in[gid] : 1e30f;
    // TODO: Two-pass block reduction finding minimum value
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Grid Reduction with Global Atomic Accumulation
// -----------------------------------------------------------------------------
__global__ void kernel_grid_atomic_reduce(const float *in, float *global_total, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float local_sum = 0.0f;
    // TODO: 1. Grid-stride loop accumulation into local_sum
    // TODO: 2. Two-pass reduction across block
    // TODO: 3. Thread 0 executes: atomicAdd(global_total, block_sum);
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Block Sum of Squares (for RMSNorm)
// -----------------------------------------------------------------------------
__global__ void kernel_block_sum_sq_reduce(const float *in, float *out, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? in[gid] : 0.0f;
    float sq = val * val;
    // TODO: Two-pass block reduction of sq into out[blockIdx.x]
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.1 Two-Pass Block & Warp Reductions (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Block Sum
    {
        int n = 256;
        std::vector<float> h_in(n, 3.0f);
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_sum_reduce<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 768.0f) < 1e-3f);
        reportStatus("Problem 1: 256-Thread Block Sum Reduction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Block Max
    {
        int n = 256;
        std::vector<float> h_in(n, 10.0f);
        h_in[127] = 999.0f;
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_max_reduce<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (h_out == 999.0f);
        reportStatus("Problem 2: 256-Thread Block Max Reduction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Block Min
    {
        int n = 256;
        std::vector<float> h_in(n, 50.0f);
        h_in[42] = -123.0f;
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_min_reduce<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (h_out == -123.0f);
        reportStatus("Problem 3: 256-Thread Block Min Reduction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Grid Atomic Reduce
    {
        int n = 10000;
        std::vector<float> h_in(n, 2.0f);
        float h_total = 0.0f;
        float *d_in, *d_total;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_total, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_total, 0, sizeof(float));

        kernel_grid_atomic_reduce<<<4, 256>>>(d_in, d_total, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_total - 20000.0f) < 1.0f);
        reportStatus("Problem 4: Grid Reduction with Global Atomic Accumulation", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_total);
    }

    // Test 5: Block Sum of Squares
    {
        int n = 256;
        std::vector<float> h_in(n, 4.0f); // 4^2 = 16, 16 * 256 = 4096
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_sum_sq_reduce<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 4096.0f) < 1e-3f);
        reportStatus("Problem 5: Block Sum of Squares (for RMSNorm)", ok);
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
