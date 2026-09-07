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

// -----------------------------------------------------------------------------
// PROBLEM 1: Dual-Statistic Reduction (Simultaneous Sum & Sum of Squares for LayerNorm)
// -----------------------------------------------------------------------------
__device__ inline float2 warp_reduce_dual(float2 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

__global__ void kernel_dual_stats_reduce(const float *in, float *out_mean, float *out_var, int n) {
    __shared__ float2 s_scratch[8]; // 256 threads = 8 warps
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float val = (tid < n) ? in[tid] : 0.0f;
    float2 local_pair = make_float2(val, val * val);

    // TODO: 1. Reduce local_pair across warp using warp_reduce_dual
    // TODO: 2. Lane 0 stores in s_scratch[warp_id]
    // TODO: 3. __syncthreads();
    // TODO: 4. Warp 0 performs final reduction of s_scratch
    // TODO: 5. Lane 0 computes mean = sum / n; variance = (sum_sq / n) - (mean * mean);
    //          *out_mean = mean; *out_var = variance;
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Row-Wise RMSNorm Forward Kernel (1 Block per Row)
// -----------------------------------------------------------------------------
__global__ void kernel_rmsnorm_row_reduce(
    const float *in,
    const float *gamma,
    float *out,
    float eps,
    int cols
) {
    __shared__ float s_scratch[8];
    __shared__ float s_rms;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: 1. Grid-stride or direct row load of in[row * cols + col]
    // TODO: 2. Compute local sum of squares: local_sq += x * x
    // TODO: 3. Two-pass reduction to compute total row sum of squares
    // TODO: 4. If threadIdx.x == 0, s_rms = rsqrtf(total_sq / (float)cols + eps);
    // TODO: 5. __syncthreads();
    // TODO: 6. Normalize and scale: out[row * cols + col] = in[row * cols + col] * s_rms * gamma[col];
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Block ArgMax Reduction (Tracking Value AND Index)
// -----------------------------------------------------------------------------
struct ValIndex {
    float val;
    int idx;
};

__device__ inline ValIndex warp_reduce_argmax(ValIndex v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        if (other_val > v.val) {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__global__ void kernel_block_argmax(const float *in, float *out_max, int *out_idx, int n) {
    __shared__ ValIndex s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    ValIndex local_v;
    local_v.val = (tid < n) ? in[tid] : -1e30f;
    local_v.idx = tid;

    // TODO: Two-pass reduction using warp_reduce_argmax
    // Thread 0 writes out_max and out_idx
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Multi-Row Batched Sum Reduction (Output Vector of Row Sums)
// -----------------------------------------------------------------------------
__global__ void kernel_batched_row_sums(const float *in, float *out_row_sums, int rows, int cols) {
    // 1 block per row, blockDim.x = 256
    __shared__ float s_scratch[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Compute sum of row 'row' and store into out_row_sums[row]
}

// -----------------------------------------------------------------------------
// PROBLEM 5: 1024-Thread Block Reduction (32 Warps -> 32 Shared Scratchpad)
// -----------------------------------------------------------------------------
__global__ void kernel_1024_thread_reduce(const float *in, float *out, int n) {
    __shared__ float s_scratch[32]; // 32 warps for 1024 threads
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float val = (tid < n) ? in[tid] : 0.0f;
    // TODO: Intra-warp reduce, write to s_scratch[warp_id], __syncthreads(),
    // warp 0 reduces all 32 scratchpad elements!
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.1 Two-Pass Block & Warp Reductions (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Dual Stats (Mean & Variance)
    {
        int n = 256;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i; // Mean is 127.5
        float h_mean = 0.0f, h_var = 0.0f;

        float *d_in, *d_mean, *d_var;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_mean, sizeof(float));
        cudaMalloc(&d_var, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_dual_stats_reduce<<<1, 256>>>(d_in, d_mean, d_var, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_var, d_var, sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = (std::fabs(h_mean - 127.5f) < 1e-2f);
        reportStatus("Problem 1: Dual-Statistic Reduction (Mean & Variance for LayerNorm)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_mean); cudaFree(d_var);
    }

    // Test 2: Row-Wise RMSNorm
    {
        int rows = 4, cols = 256;
        int n = rows * cols;
        std::vector<float> h_in(n, 2.0f);
        std::vector<float> h_gamma(cols, 1.5f);
        std::vector<float> h_out(n, 0.0f);

        float *d_in, *d_gamma, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_gamma, cols * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

        kernel_rmsnorm_row_reduce<<<rows, cols>>>(d_in, d_gamma, d_out, 1e-5f, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // In this uniform array, RMS is exactly 2.0. So 2.0 / 2.0 * 1.5 = 1.5!
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 1.5f) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 2: Row-Wise RMSNorm Forward Kernel (1 Block per Row)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_gamma); cudaFree(d_out);
    }

    // Test 3: Block ArgMax
    {
        int n = 256;
        std::vector<float> h_in(n, 1.0f);
        h_in[173] = 888.0f; // Spike at index 173
        float h_max = 0.0f;
        int h_idx = -1;

        float *d_in, *d_max;
        int *d_idx;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_idx, sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_argmax<<<1, 256>>>(d_in, d_max, d_idx, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);

        bool ok = (h_max == 888.0f && h_idx == 173);
        reportStatus("Problem 3: Block ArgMax Reduction (Tracking Value AND Index)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_max); cudaFree(d_idx);
    }

    // Test 4: Batched Row Sums
    {
        int rows = 8, cols = 256;
        int n = rows * cols;
        std::vector<float> h_in(n, 2.5f);
        std::vector<float> h_row_sums(rows, 0.0f);

        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, rows * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_batched_row_sums<<<rows, 256>>>(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_row_sums.data(), d_out, rows * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            if (std::fabs(h_row_sums[r] - (cols * 2.5f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 4: Multi-Row Batched Sum Reduction (Output Vector)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: 1024-Thread Reduction
    {
        int n = 1024;
        std::vector<float> h_in(n, 1.0f);
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_1024_thread_reduce<<<1, 1024>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 1024.0f) < 1e-3f);
        reportStatus("Problem 5: 1024-Thread Block Reduction (32 Warps -> 1)", ok);
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
