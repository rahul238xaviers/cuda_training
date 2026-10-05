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
// PROBLEM 1: 32x32 Matrix Transpose with Padded Shared Memory (Zero Bank Conflicts)
// -----------------------------------------------------------------------------
__global__ void kernel_transpose_padded(const float *in, float *out, int n) {
    // Shared tile with +1 column padding to prevent bank conflicts on column reads
    __shared__ float s_tile[32][33];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * 32 + tx;
    int gy = blockIdx.y * 32 + ty;

    // TODO: 1. Coalesced read from in[gy * n + gx] into s_tile[ty][tx]
    // TODO: 2. __syncthreads();
    // TODO: 3. Transposed write from s_tile[tx][ty] to out[gx * n + gy] (coalesced write)
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Block-Wide Reduction in Shared Memory (1024 -> 1)
// -----------------------------------------------------------------------------
__global__ void kernel_block_tree_reduce(const float *in, float *out_sum, int n) {
    __shared__ float s_data[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 1. Load in[gid] into s_data[tid]
    // TODO: 2. __syncthreads();
    // TODO: 3. Perform tree reduction in shared memory:
    //          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //              if (tid < s) { s_data[tid] += s_data[tid + s]; }
    //              __syncthreads();
    //          }
    // TODO: 4. If tid == 0, out_sum[blockIdx.x] = s_data[0];
}

// -----------------------------------------------------------------------------
// PROBLEM 3: 1D 3-Point Stencil with Shared Memory Halo (Radius = 1)
// -----------------------------------------------------------------------------
__global__ void kernel_stencil_1d_shared(const float *in, float *out, int n) {
    // 256 threads per block, loading 256 + 2 halo elements = 258 elements
    __shared__ float s_halo[258];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 1. Load interior element: s_halo[tid + 1] = (gid < n) ? in[gid] : 0.0f;
    // TODO: 2. Load left halo (tid == 0): s_halo[0] = (gid > 0) ? in[gid - 1] : 0.0f;
    // TODO: 3. Load right halo (tid == blockDim.x - 1): s_halo[tid + 2] = (gid + 1 < n) ? in[gid + 1] : 0.0f;
    // TODO: 4. __syncthreads();
    // TODO: 5. Apply stencil: out[gid] = 0.25f * s_halo[tid] + 0.5f * s_halo[tid + 1] + 0.25f * s_halo[tid + 2];
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Shared Memory Multi-Row Circular Shift
// -----------------------------------------------------------------------------
__global__ void kernel_shared_circular_shift(const float *in, float *out, int shift, int n) {
    __shared__ float s_buf[256];
    int tid = threadIdx.x;
    if (tid < n) {
        // TODO: 1. s_buf[tid] = in[tid];
        // TODO: 2. __syncthreads();
        // TODO: 3. out[tid] = s_buf[(tid + shift) % n];
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Parallel Prefix Sum in Shared Memory (Blelloch Scan Work-Efficient)
// -----------------------------------------------------------------------------
__global__ void kernel_shared_blelloch_scan(const int *in, int *out, int n) {
    __shared__ int temp[256];
    int tid = threadIdx.x;

    // Up-sweep (reduce) & Down-sweep
    // TODO: 1. Load in[tid] into temp[tid];
    // TODO: 2. __syncthreads();
    // TODO: 3. Up-sweep:
    //          int offset = 1;
    //          for (int d = n >> 1; d > 0; d >>= 1) {
    //              __syncthreads();
    //              if (tid < d) {
    //                  int ai = offset * (2 * tid + 1) - 1;
    //                  int bi = offset * (2 * tid + 2) - 1;
    //                  temp[bi] += temp[ai];
    //              }
    //              offset *= 2;
    //          }
    // TODO: 4. if (tid == 0) temp[n - 1] = 0; // Clear root
    // TODO: 5. Down-sweep:
    //          for (int d = 1; d < n; d *= 2) {
    //              offset >>= 1;
    //              __syncthreads();
    //              if (tid < d) {
    //                  int ai = offset * (2 * tid + 1) - 1;
    //                  int bi = offset * (2 * tid + 2) - 1;
    //                  int t = temp[ai];
    //                  temp[ai] = temp[bi];
    //                  temp[bi] += t;
    //              }
    //          }
    // TODO: 6. __syncthreads(); out[tid] = temp[tid];
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.3 Thread Blocks & Shared Memory (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Padded Matrix Transpose
    {
        int n = 32;
        std::vector<float> h_in(n * n), h_out(n * n, 0.0f);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                h_in[r * n + c] = (float)(r * 100 + c);
            }
        }
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * n * sizeof(float));
        cudaMalloc(&d_out, n * n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 32);
        dim3 grid(1, 1);
        kernel_transpose_padded<<<grid, block>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (h_out[c * n + r] != h_in[r * n + c]) { ok = false; break; }
            }
        }
        reportStatus("Problem 1: 32x32 Matrix Transpose with Padded Shared Memory", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Block-Wide Reduction
    {
        int n = 1024;
        std::vector<float> h_in(n, 1.0f);
        float h_sum = 0.0f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_block_tree_reduce<<<1, 1024>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_sum - 1024.0f) < 1e-4f);
        reportStatus("Problem 2: Block-Wide Reduction in Shared Memory (1024 -> 1)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: 1D Stencil with Halo
    {
        int n = 256;
        std::vector<float> h_in(n, 2.0f), h_out(n, 0.0f);
        h_in[10] = 10.0f; // Spike
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_stencil_1d_shared<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // Verify index 10: 0.25*2 + 0.5*10 + 0.25*2 = 6.0
        if (std::fabs(h_out[10] - 6.0f) > 1e-4f) ok = false;
        // Verify index 9: 0.25*2 + 0.5*2 + 0.25*10 = 4.0
        if (std::fabs(h_out[9] - 4.0f) > 1e-4f) ok = false;
        reportStatus("Problem 3: 1D 3-Point Stencil with Shared Memory Halo", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Circular Shift
    {
        int n = 256;
        int shift = 5;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_shared_circular_shift<<<1, 256>>>(d_in, d_out, shift, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != (float)((i + shift) % n)) { ok = false; break; }
        }
        reportStatus("Problem 4: Shared Memory Multi-Row Circular Shift", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Blelloch Scan
    {
        int n = 256;
        std::vector<int> h_in(n, 1), h_out(n, 0);
        int *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(int));
        cudaMalloc(&d_out, n * sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        kernel_shared_blelloch_scan<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != i) { ok = false; break; } // Exclusive scan of all 1s is 0, 1, 2, ..., n-1
        }
        reportStatus("Problem 5: Parallel Exclusive Prefix Scan in Shared Memory", ok);
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
