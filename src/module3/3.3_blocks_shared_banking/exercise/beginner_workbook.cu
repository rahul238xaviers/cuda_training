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
// PROBLEM 1: Static Shared Memory Load & Store
// -----------------------------------------------------------------------------
__global__ void kernel_shared_load_store(const float *in, float *out, int n) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 1. If gid < n, load in[gid] into s_data[tid], else load 0.0f
    // TODO: 2. Synchronize block threads: __syncthreads();
    // TODO: 3. If gid < n, write s_data[tid] to out[gid]
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Block-Level Array Reversal in Shared Memory
// -----------------------------------------------------------------------------
__global__ void kernel_block_reverse(const float *in, float *out, int block_size) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;

    // TODO: 1. Load in[tid] into s_data[tid]
    // TODO: 2. Synchronize threads: __syncthreads();
    // TODO: 3. Write reversed element s_data[block_size - 1 - tid] to out[tid]
}

// -----------------------------------------------------------------------------
// PROBLEM 3: 2D 16x16 Tile Staging in Shared Memory
// -----------------------------------------------------------------------------
__global__ void kernel_tile_staging(const float *in, float *out, int width, int height) {
    __shared__ float s_tile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // TODO: 1. If inside bounds (gx < width && gy < height), load in[gy * width + gx] into s_tile[ty][tx]
    // TODO: 2. Synchronize: __syncthreads();
    // TODO: 3. Multiply value in s_tile by 2.0f and write back to out[gy * width + gx]
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Dynamic Shared Memory Allocation
// -----------------------------------------------------------------------------
extern __shared__ float s_dyn[];

__global__ void kernel_dynamic_shared(const float *in, float *out, float bias, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 1. Load in[gid] into s_dyn[tid]
    // TODO: 2. __syncthreads();
    // TODO: 3. Write s_dyn[tid] + bias to out[gid]
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Stride-1 Bank Mapping Verification
// -----------------------------------------------------------------------------
__global__ void kernel_stride1_bank_check(int *out_banks) {
    // Each of the 32 threads in warp 0 writes its bank ID: (threadIdx.x * 4 / 4) % 32 = threadIdx.x
    __shared__ int s_words[32];
    int tid = threadIdx.x;
    if (tid < 32) {
        // TODO: Store tid in s_words[tid] (stride-1 access to 32 banks)
        // TODO: __syncthreads();
        // TODO: out_banks[tid] = s_words[tid];
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.3 Thread Blocks & Shared Memory (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Static Shared Memory Load & Store
    {
        int n = 256;
        std::vector<float> h_in(n, 7.5f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_shared_load_store<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 7.5f) { ok = false; break; }
        reportStatus("Problem 1: Static Shared Memory Load & Store", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Block Reverse
    {
        int n = 256;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_reverse<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != (float)(n - 1 - i)) { ok = false; break; }
        reportStatus("Problem 2: Block-Level Array Reversal in Shared Memory", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: 2D Tile Staging
    {
        int width = 16, height = 16, n = width * height;
        std::vector<float> h_in(n, 4.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        kernel_tile_staging<<<1, block>>>(d_in, d_out, width, height);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 8.0f) { ok = false; break; }
        reportStatus("Problem 3: 2D 16x16 Tile Staging in Shared Memory", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Dynamic Shared Memory
    {
        int n = 128;
        std::vector<float> h_in(n, 10.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_dynamic_shared<<<1, n, n * sizeof(float)>>>(d_in, d_out, 5.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) if (h_out[i] != 15.0f) { ok = false; break; }
        reportStatus("Problem 4: Dynamic Shared Memory Allocation", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Stride-1 Bank Mapping
    {
        int *d_banks;
        cudaMalloc(&d_banks, 32 * sizeof(int));
        kernel_stride1_bank_check<<<1, 32>>>(d_banks);
        cudaDeviceSynchronize();

        std::vector<int> h_banks(32);
        cudaMemcpy(h_banks.data(), d_banks, 32 * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 32; ++i) if (h_banks[i] != i) { ok = false; break; }
        reportStatus("Problem 5: Stride-1 Bank Mapping Verification", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_banks);
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
