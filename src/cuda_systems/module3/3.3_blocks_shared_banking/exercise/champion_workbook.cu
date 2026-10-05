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
// PROBLEM 1: Two-Pass Block Reduction (Intra-Warp Shuffle + Shared Warp Scratchpad)
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kernel_two_pass_block_reduce(const float *in, float *out_block_sums, int n) {
    // 256 threads = 8 warps. Shared scratchpad needs only 8 floats!
    __shared__ float s_warp_sums[8];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? in[gid] : 0.0f;

    // TODO: 1. Intra-warp reduction
    // float sum = warp_reduce_sum(val);
    // TODO: 2. Lane 0 writes warp sum into shared scratchpad
    // if (lane == 0) s_warp_sums[warp_id] = sum;
    // TODO: 3. Synchronize block: __syncthreads();
    // TODO: 4. Warp 0 reads shared scratchpad and performs final warp reduction
    // if (warp_id == 0) {
    //     float final_val = (lane < 8) ? s_warp_sums[lane] : 0.0f;
    //     float block_total = warp_reduce_sum(final_val);
    //     if (lane == 0) out_block_sums[blockIdx.x] = block_total;
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Double-Buffered Shared Memory Streaming (Ping-Pong Latency Hiding)
// -----------------------------------------------------------------------------
__global__ void kernel_double_buffered_stream(const float *in, float *out, int chunk_size, int num_chunks) {
    // Two shared memory buffers: s_buf[0] and s_buf[1]
    __shared__ float s_buf[2][128];
    int tid = threadIdx.x;

    // TODO: 1. Prefetch chunk 0 into s_buf[0][tid]
    // TODO: 2. Loop through remaining chunks:
    //          compute on s_buf[curr_buf], prefetch into s_buf[next_buf]
    // TODO: 3. Synchronize between stages
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Row-Wise Block Softmax using Shared Memory Scratchpad
// -----------------------------------------------------------------------------
__global__ void kernel_block_row_softmax(const float *in, float *out, int cols) {
    // 1 block per row, blockDim.x = 256
    __shared__ float s_scratch[8];
    __shared__ float s_row_max;
    __shared__ float s_row_sum_exp;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int row = blockIdx.x;

    // TODO: 1. Find row max across 256 threads using warp reduction + shared scratchpad
    // TODO: 2. Broadcast s_row_max to all threads
    // TODO: 3. Compute exp(x - s_row_max) and sum across 256 threads
    // TODO: 4. Broadcast s_row_sum_exp
    // TODO: 5. Normalize and write out[row * cols + tid]
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Conflict-Free 8-Way Sub-Matrix Tile Transpose
// -----------------------------------------------------------------------------
__global__ void kernel_conflict_free_submatrix(const float *in, float *out, int n) {
    // Transpose 64x64 using padded 64x65 shared tile
    __shared__ float s_tile[64][65];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // TODO: Load in-place, sync, transpose to out without bank conflicts
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Shared Memory Fast Histogram with Warp Serialization Prevention
// -----------------------------------------------------------------------------
__global__ void kernel_shared_fast_histogram(const int *in, int *out_hist, int n, int num_bins) {
    // Replicate bins across warps to prevent atomic contention in shared memory
    __shared__ int s_warp_bins[8][16]; // 8 warps, 16 bins
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // TODO: 1. Initialize s_warp_bins to 0
    // TODO: 2. __syncthreads();
    // TODO: 3. Each thread atomics only into its private warp bin: atomicAdd(&s_warp_bins[warp_id][bin], 1)
    // TODO: 4. __syncthreads();
    // TODO: 5. Reduce across warps and write to out_hist
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.3 Thread Blocks & Shared Memory (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Two-Pass Block Reduce
    {
        int n = 256;
        std::vector<float> h_in(n, 2.0f);
        float h_out = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_two_pass_block_reduce<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 512.0f) < 1e-4f);
        reportStatus("Problem 1: Two-Pass Block Reduction (Intra-Warp + Scratchpad)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Double-Buffered Shared Memory Streaming
    {
        int chunk_size = 128;
        int num_chunks = 4;
        int n = chunk_size * num_chunks;
        std::vector<float> h_in(n, 1.5f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_double_buffered_stream<<<1, 128>>>(d_in, d_out, chunk_size, num_chunks);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (h_in[i] * 2.0f)) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: Double-Buffered Shared Memory Streaming", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Row-Wise Softmax
    {
        int rows = 4, cols = 256;
        int n = rows * cols;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                h_in[r * cols + c] = (float)(c % 32) * 0.1f;
            }
        }
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_row_softmax<<<rows, cols>>>(d_in, d_out, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) sum += h_out[r * cols + c];
            // Sum of probabilities for a row must equal 1.0
            if (std::fabs(sum - 1.0f) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 3: Row-Wise Block Softmax using Shared Scratchpad", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Conflict-Free 64x64 Transpose
    {
        int n = 64;
        std::vector<float> h_in(n * n), h_out(n * n, 0.0f);
        for (int i = 0; i < n * n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * n * sizeof(float));
        cudaMalloc(&d_out, n * n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 8); // 256 threads iterating over 64x64
        kernel_conflict_free_submatrix<<<1, block>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (h_out[c * n + r] != h_in[r * n + c]) { ok = false; break; }
            }
        }
        reportStatus("Problem 4: Conflict-Free 8-Way Sub-Matrix Tile Transpose", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Shared Fast Histogram
    {
        int n = 1024, num_bins = 16;
        std::vector<int> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = i % num_bins;
        std::vector<int> h_hist(num_bins, 0);

        int *d_in, *d_hist;
        cudaMalloc(&d_in, n * sizeof(int));
        cudaMalloc(&d_hist, num_bins * sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        kernel_shared_fast_histogram<<<1, 256>>>(d_in, d_hist, n, num_bins);
        cudaDeviceSynchronize();

        cudaMemcpy(h_hist.data(), d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        int expected_per_bin = n / num_bins;
        for (int b = 0; b < num_bins; ++b) {
            if (h_hist[b] != expected_per_bin) { ok = false; break; }
        }
        reportStatus("Problem 5: Shared Memory Fast Histogram (Replicated Bins)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_hist);
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
