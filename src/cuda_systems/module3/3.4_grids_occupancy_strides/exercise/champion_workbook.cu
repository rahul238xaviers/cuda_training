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
// PROBLEM 1: Grid-Stride Single-Pass Global Sum with Atomic Block Counter
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kernel_single_pass_global_sum(
    const float *in,
    float *out_total,
    float *block_scratch,
    int *atomic_counter,
    int n
) {
    __shared__ float s_scratch[32];
    float local_sum = 0.0f;

    // TODO: 1. Grid-stride loop accumulation
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    //     local_sum += in[i];
    // }
    // float warp_sum = warp_reduce_sum(local_sum);
    // int lane = threadIdx.x % 32;
    // int warp_id = threadIdx.x / 32;
    // if (lane == 0) s_scratch[warp_id] = warp_sum;
    // __syncthreads();
    // if (warp_id == 0) {
    //     float final_val = (lane < (blockDim.x / 32)) ? s_scratch[lane] : 0.0f;
    //     float block_sum = warp_reduce_sum(final_val);
    //     if (lane == 0) block_scratch[blockIdx.x] = block_sum;
    // }
    // TODO: 2. Last block detection using atomicAdd(atomic_counter, 1)
    // __threadfence();
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     int finished = atomicAdd(atomic_counter, 1);
    //     if (finished == gridDim.x - 1) {
    //         float grand_total = 0.0f;
    //         for (int b = 0; b < gridDim.x; ++b) grand_total += block_scratch[b];
    //         *out_total = grand_total;
    //     }
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Grid-Stride 2D Tiled Transpose (Arbitrary Giant Matrices)
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_tiled_transpose(const float *in, float *out, int rows, int cols) {
    __shared__ float s_tile[16][17]; // +1 padding
    // TODO: Grid-stride in 2D over tiles of size 16x16
    // for (int by = blockIdx.y; by * 16 < rows; by += gridDim.y) {
    //     for (int bx = blockIdx.x; bx * 16 < cols; bx += gridDim.x) {
    //         int gx = bx * 16 + threadIdx.x;
    //         int gy = by * 16 + threadIdx.y;
    //         if (gx < cols && gy < rows) s_tile[threadIdx.y][threadIdx.x] = in[gy * cols + gx];
    //         __syncthreads();
    //         int out_gx = by * 16 + threadIdx.x;
    //         int out_gy = bx * 16 + threadIdx.y;
    //         if (out_gx < rows && out_gy < cols) out[out_gy * rows + out_gx] = s_tile[threadIdx.x][threadIdx.y];
    //         __syncthreads();
    //     }
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Persistent Work-Queue Consumer Kernel
// -----------------------------------------------------------------------------
__global__ void kernel_persistent_queue_worker(
    const float *input_tasks,
    float *output_results,
    int *global_task_counter,
    int total_tasks
) {
    // Threads cooperatively consume task indices using atomicAdd on global_task_counter
    // TODO: While loop grabbing next task until all tasks complete
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Grid-Stride Multi-Head Softmax Accumulator
// -----------------------------------------------------------------------------
__global__ void kernel_grid_stride_mha_scale(const float *in, float *out, float scale, int B, int H, int S, int D) {
    int total_elements = B * H * S * D;
    // TODO: Grid-stride scale by 1.0f / sqrt(D)
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
    //     out[i] = in[i] * scale;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Dual-Grid Split Pipeline (Even vs Odd Rows)
// -----------------------------------------------------------------------------
__global__ void kernel_split_row_stride(const float *in, float *out_even, float *out_odd, int rows, int cols) {
    // TODO: Distribute even rows to out_even and odd rows to out_odd with grid stride
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.4 Grids & Grid-Stride Loops (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Single-Pass Global Sum with Atomic Block Counter
    {
        int n = 200000;
        std::vector<float> h_in(n, 1.5f);
        float h_total = 0.0f;
        int num_blocks = 8;

        float *d_in, *d_total, *d_scratch;
        int *d_counter;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_total, sizeof(float));
        cudaMalloc(&d_scratch, num_blocks * sizeof(float));
        cudaMalloc(&d_counter, sizeof(int));

        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_counter, 0, sizeof(int));
        cudaMemset(d_total, 0, sizeof(float));

        kernel_single_pass_global_sum<<<num_blocks, 256>>>(d_in, d_total, d_scratch, d_counter, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_total - 300000.0f) < 1.0f);
        reportStatus("Problem 1: Single-Pass Global Sum with Atomic Block Counter", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_total); cudaFree(d_scratch); cudaFree(d_counter);
    }

    // Test 2: Grid-Stride Tiled Transpose
    {
        int rows = 128, cols = 64;
        int n = rows * cols;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) h_in[r * cols + c] = (float)(r * 100 + c);
        }
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(2, 2); // 4 blocks handling entire 128x64 matrix
        kernel_grid_stride_tiled_transpose<<<grid, block>>>(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (h_out[c * rows + r] != h_in[r * cols + c]) { ok = false; break; }
            }
        }
        reportStatus("Problem 2: Grid-Stride 2D Tiled Transpose (Arbitrary Giant Matrices)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Persistent Work-Queue Consumer
    {
        int total_tasks = 1000;
        std::vector<float> h_tasks(total_tasks, 5.0f), h_res(total_tasks, 0.0f);
        float *d_tasks, *d_res;
        int *d_counter;
        cudaMalloc(&d_tasks, total_tasks * sizeof(float));
        cudaMalloc(&d_res, total_tasks * sizeof(float));
        cudaMalloc(&d_counter, sizeof(int));
        cudaMemcpy(d_tasks, h_tasks.data(), total_tasks * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_counter, 0, sizeof(int));

        kernel_persistent_queue_worker<<<4, 64>>>(d_tasks, d_res, d_counter, total_tasks);
        cudaDeviceSynchronize();

        cudaMemcpy(h_res.data(), d_res, total_tasks * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < total_tasks; ++i) {
            if (h_res[i] != 10.0f) { ok = false; break; } // task multiplies by 2
        }
        reportStatus("Problem 3: Persistent Work-Queue Consumer Kernel", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_tasks); cudaFree(d_res); cudaFree(d_counter);
    }

    // Test 4: MHA Scale Grid Stride
    {
        int B = 2, H = 4, S = 16, D = 64;
        int total_elements = B * H * S * D;
        std::vector<float> h_in(total_elements, 8.0f), h_out(total_elements, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, total_elements * sizeof(float));
        cudaMalloc(&d_out, total_elements * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

        kernel_grid_stride_mha_scale<<<4, 128>>>(d_in, d_out, 0.125f, B, H, S, D);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < total_elements; ++i) {
            if (std::fabs(h_out[i] - 1.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 4: Grid-Stride Multi-Head Softmax Pre-Scaler", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Split Row Stride
    {
        int rows = 20, cols = 10;
        int n = rows * cols;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        std::vector<float> h_even((rows / 2) * cols, 0.0f), h_odd((rows / 2) * cols, 0.0f);

        float *d_in, *d_even, *d_odd;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_even, (rows / 2) * cols * sizeof(float));
        cudaMalloc(&d_odd, (rows / 2) * cols * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_split_row_stride<<<2, 32>>>(d_in, d_even, d_odd, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_even.data(), d_even, (rows / 2) * cols * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_odd.data(), d_odd, (rows / 2) * cols * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (r % 2 == 0) {
                    if (h_even[(r / 2) * cols + c] != h_in[r * cols + c]) ok = false;
                } else {
                    if (h_odd[(r / 2) * cols + c] != h_in[r * cols + c]) ok = false;
                }
            }
        }
        reportStatus("Problem 5: Dual-Grid Split Pipeline (Even vs Odd Rows)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_even); cudaFree(d_odd);
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
