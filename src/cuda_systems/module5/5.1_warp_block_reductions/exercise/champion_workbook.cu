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
// PROBLEM 1: RMSNorm Backward Pass (Dual Reductions: sum(dY*x*gamma) and sum(dY*x))
// -----------------------------------------------------------------------------
__global__ void kernel_rmsnorm_backward(
    const float *dY,
    const float *x,
    const float *gamma,
    float *dX,
    float eps,
    int cols
) {
    // Row-wise backward pass.
    // 1. Compute rms = rsqrtf(mean(x^2) + eps)
    // 2. Reduce sum(dY * x * gamma) across row
    // 3. Compute dX[i] = rms * (dY[i] * gamma[i] - (x[i] / (cols * rms^2)) * sum)
    __shared__ float s_scratch[8];
    __shared__ float s_sum_sq;
    __shared__ float s_sum_dy_x_gamma;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Implement RMSNorm backward pass with hierarchical reductions
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Fused Residual Add + RMSNorm Forward in Single Pass
// -----------------------------------------------------------------------------
__global__ void kernel_fused_add_rmsnorm(
    float *x, // in-place residual target: x = x + residual
    const float *residual,
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

    // TODO: 1. In-place add residual to x: float updated = x[row * cols + tid] + residual[row * cols + tid];
    //          x[row * cols + tid] = updated;
    // TODO: 2. Compute RMS reduction over updated values
    // TODO: 3. Normalize: out[row * cols + tid] = updated * s_rms * gamma[tid];
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Hierarchical Row-Wise Softmax (Max -> ExpSum -> Normalize)
// -----------------------------------------------------------------------------
__global__ void kernel_hierarchical_row_softmax(const float *in, float *out, int cols) {
    __shared__ float s_scratch[8];
    __shared__ float s_row_max;
    __shared__ float s_row_sum_exp;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Phase 1: Block Max Reduction
    // TODO: Phase 2: Compute exp(x - s_row_max), Block Sum Reduction
    // TODO: Phase 3: Normalize out[row * cols + tid] = exp_val / s_row_sum_exp
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Multi-Block Non-Atomic Global Reduction Tree
// -----------------------------------------------------------------------------
__global__ void kernel_block_partial_sum(const float *in, float *block_sums, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    // TODO: Grid-stride reduction of in into block_sums[blockIdx.x] without any atomics
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Block-Level Cosine Similarity Accumulator (Dot Product / Norms)
// -----------------------------------------------------------------------------
__global__ void kernel_block_cosine_similarity(const float *a, const float *b, float *out_sim, int n) {
    // Simultaneously reduce: dot = sum(a*b), norm_a = sum(a*a), norm_b = sum(b*b)
    // Result = dot / (sqrt(norm_a) * sqrt(norm_b))
    // TODO: Triple accumulation in registers + hierarchical block reduction
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.1 Two-Pass Block & Warp Reductions (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: RMSNorm Backward
    {
        int rows = 4, cols = 256;
        int n = rows * cols;
        std::vector<float> h_dY(n, 1.0f), h_x(n, 2.0f), h_gamma(cols, 1.0f), h_dX(n, 0.0f);

        float *d_dY, *d_x, *d_gamma, *d_dX;
        cudaMalloc(&d_dY, n * sizeof(float));
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_gamma, cols * sizeof(float));
        cudaMalloc(&d_dX, n * sizeof(float));

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                h_x[r * cols + c] = (float)(c + 1) * 0.1f;
                h_dY[r * cols + c] = 1.0f;
            }
        }
        cudaMemcpy(d_dY, h_dY.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

        kernel_rmsnorm_backward<<<rows, cols>>>(d_dY, d_x, d_gamma, d_dX, 1e-5f, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_dX.data(), d_dX, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // Verify non-zero gradient computed across row
        float row0_sum = 0.0f;
        for (int c = 0; c < cols; ++c) row0_sum += std::fabs(h_dX[c]);
        if (row0_sum < 1e-3f) ok = false;
        reportStatus("Problem 1: RMSNorm Backward Pass (Dual Reductions)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_dY); cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_dX);
    }

    // Test 2: Fused Residual Add + RMSNorm
    {
        int rows = 4, cols = 256;
        int n = rows * cols;
        std::vector<float> h_x(n, 1.0f), h_res(n, 1.0f), h_gamma(cols, 2.0f), h_out(n, 0.0f);

        float *d_x, *d_res, *d_gamma, *d_out;
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_res, n * sizeof(float));
        cudaMalloc(&d_gamma, cols * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));

        cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_res, h_res.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

        kernel_fused_add_rmsnorm<<<rows, cols>>>(d_x, d_res, d_gamma, d_out, 1e-5f, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = true;
        // x was updated to 2.0 (1+1). RMS of 2.0 is 2.0. Normalized = (2.0/2.0) * 2.0 = 2.0.
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 2.0f) > 1e-3f || std::fabs(h_x[i] - 2.0f) > 1e-3f) {
                ok = false; break;
            }
        }
        reportStatus("Problem 2: Fused Residual Add + RMSNorm Forward in Single Pass", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_gamma); cudaFree(d_out);
    }

    // Test 3: Hierarchical Row-Wise Softmax
    {
        int rows = 4, cols = 256;
        int n = rows * cols;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) h_in[r * cols + c] = (float)(c % 32) * 0.1f;
        }

        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_hierarchical_row_softmax<<<rows, cols>>>(d_in, d_out, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) sum += h_out[r * cols + c];
            if (std::fabs(sum - 1.0f) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 3: Hierarchical Row-Wise Softmax (Max -> ExpSum -> Norm)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Multi-Block Reduction Tree
    {
        int n = 10000;
        int num_blocks = 4;
        std::vector<float> h_in(n, 1.0f);
        std::vector<float> h_block_sums(num_blocks, 0.0f);

        float *d_in, *d_block_sums;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_partial_sum<<<num_blocks, 256>>>(d_in, d_block_sums, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        float total_sum = 0.0f;
        for (float s : h_block_sums) total_sum += s;

        bool ok = (std::fabs(total_sum - 10000.0f) < 1.0f);
        reportStatus("Problem 4: Multi-Block Non-Atomic Global Reduction Tree", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_block_sums);
    }

    // Test 5: Cosine Similarity
    {
        int n = 256;
        std::vector<float> h_a(n, 3.0f), h_b(n, 3.0f);
        float h_sim = 0.0f;

        float *d_a, *d_b, *d_sim;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_sim, sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_cosine_similarity<<<1, 256>>>(d_a, d_b, d_sim, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_sim, d_sim, sizeof(float), cudaMemcpyDeviceToHost);
        // Cosine similarity of identical vectors is exactly 1.0
        bool ok = (std::fabs(h_sim - 1.0f) < 1e-3f);
        reportStatus("Problem 5: Block-Level Cosine Similarity Accumulator", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_sim);
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
