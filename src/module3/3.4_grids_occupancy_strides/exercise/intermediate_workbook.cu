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
// PROBLEM 1: Grid-Stride Local Reduction into Registers
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kernel_grid_stride_reduction(const float *in, float *out_block_sums, int n) {
    __shared__ float s_scratch[32]; // for 32 warps max
    float local_sum = 0.0f;

    // TODO: 1. Accumulate multiple elements per thread using grid-stride loop:
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    //     local_sum += in[i];
    // }
    // TODO: 2. Intra-warp reduce local_sum
    // float warp_sum = warp_reduce_sum(local_sum);
    // int lane = threadIdx.x % 32;
    // int warp_id = threadIdx.x / 32;
    // if (lane == 0) s_scratch[warp_id] = warp_sum;
    // __syncthreads();
    // TODO: 3. Final reduction by warp 0
    // if (warp_id == 0) {
    //     float final_val = (lane < (blockDim.x / 32)) ? s_scratch[lane] : 0.0f;
    //     float block_sum = warp_reduce_sum(final_val);
    //     if (lane == 0) out_block_sums[blockIdx.x] = block_sum;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 3D Multi-Channel Grid-Stride Loop
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_3d(const float *in, float *out, int B, int C, int N) {
    int total = B * C * N;
    // TODO: Process flattened 3D tensor with grid stride, extracting b, c, n
    // for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    //     int n_idx = idx % N;
    //     int c_idx = (idx / N) % C;
    //     int b_idx = idx / (C * N);
    //     out[idx] = in[idx] + (float)b_idx + (float)c_idx;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Grid-Stride Conditional Filter & Thresholding
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_threshold(const float *in, float *out, float threshold, int n) {
    // TODO: Grid stride clamp: if in[i] > threshold keep, else set to 0.0f
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    //     out[i] = (in[i] > threshold) ? in[i] : 0.0f;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Vectorized Grid-Stride Scaling with Non-Aligned Tail
// -----------------------------------------------------------------------------
__global__ void kernel_vectorized_tail_stride(const float *in, float *out, float scale, int n) {
    int n4 = n / 4;
    const float4 *in4 = reinterpret_cast<const float4*>(in);
    float4 *out4 = reinterpret_cast<float4*>(out);

    // TODO: 1. Vectorized grid stride loop over n4
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    //     float4 v = in4[i];
    //     v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    //     out4[i] = v;
    // }
    // TODO: 2. Grid-stride loop over tail elements (n4 * 4 to n)
    // for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    //     out[i] = in[i] * scale;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Host-Side Max Potential Block Size Query via CUDA Runtime
// -----------------------------------------------------------------------------
// This tests using cudaOccupancyMaxPotentialBlockSize to compute launch parameters
void host_calculate_launch_dims(int *out_min_grid_size, int *out_block_size) {
    // TODO: Use cudaOccupancyMaxPotentialBlockSize(out_min_grid_size, out_block_size, kernel_grid_stride_threshold);
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.4 Grids & Grid-Stride Loops (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Grid-Stride Reduction
    {
        int n = 100000;
        std::vector<float> h_in(n, 1.0f);
        int num_blocks = 4;
        std::vector<float> h_block_sums(num_blocks, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, num_blocks * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_grid_stride_reduction<<<num_blocks, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_block_sums.data(), d_out, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        float total_sum = 0.0f;
        for (float s : h_block_sums) total_sum += s;

        bool ok = (std::fabs(total_sum - 100000.0f) < 1.0f);
        reportStatus("Problem 1: Grid-Stride Local Reduction into Registers", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: 3D Multi-Channel Grid Stride
    {
        int B = 2, C = 3, N = 100;
        int total_len = B * C * N;
        std::vector<float> h_in(total_len, 10.0f), h_out(total_len, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, total_len * sizeof(float));
        cudaMalloc(&d_out, total_len * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), total_len * sizeof(float), cudaMemcpyHostToDevice);

        kernel_grid_stride_3d<<<2, 128>>>(d_in, d_out, B, C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, total_len * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int n_i = 0; n_i < N; ++n_i) {
                    int idx = b * (C * N) + c * N + n_i;
                    float expected = 10.0f + (float)b + (float)c;
                    if (std::fabs(h_out[idx] - expected) > 1e-4f) { ok = false; break; }
                }
            }
        }
        reportStatus("Problem 2: 3D Multi-Channel Grid-Stride Loop", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Grid Stride Threshold
    {
        int n = 5000;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i - 2500.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_grid_stride_threshold<<<4, 128>>>(d_in, d_out, 0.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            float exp_val = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
            if (std::fabs(h_out[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: Grid-Stride Conditional Filter & Thresholding", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Vectorized Tail Stride
    {
        int n = 5007; // Not multiple of 4
        std::vector<float> h_in(n, 2.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_vectorized_tail_stride<<<4, 64>>>(d_in, d_out, 3.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 6.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 4: Vectorized Grid-Stride Scaling with Non-Aligned Tail", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Host-Side Max Potential Block Size Query
    {
        int min_grid = 0, block_size = 0;
        host_calculate_launch_dims(&min_grid, &block_size);
        bool ok = (min_grid > 0 && block_size > 0);
        reportStatus("Problem 5: Host-Side Max Potential Block Size Query", ok);
        if (ok) passed++;
        total++;
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
