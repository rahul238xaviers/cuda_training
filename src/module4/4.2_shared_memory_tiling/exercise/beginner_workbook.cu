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
// PROBLEM 1: 1D Tiled Dot Product using Shared Memory
// -----------------------------------------------------------------------------
__global__ void kernel_tiled_dot_1d(const float *a, const float *b, float *out_sum, int n) {
    __shared__ float s_tile[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 1. Load elementwise product a[gid] * b[gid] into s_tile[tid]
    // TODO: 2. __syncthreads();
    // TODO: 3. Tree reduction inside shared memory
    // TODO: 4. Lane 0 writes to atomicAdd(out_sum, s_tile[0])
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 2D Block-Tiled GEMM (Square 16x16 Tile)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_tiled_16x16(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    const int TILE = 16;
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // TODO: Loop over tiles along the K dimension:
    // for (int t = 0; t < N / TILE; ++t) {
    //     s_A[ty][tx] = A[row * N + (t * TILE + tx)];
    //     s_B[ty][tx] = B[(t * TILE + ty) * N + col];
    //     __syncthreads();
    //     for (int k = 0; k < TILE; ++k) {
    //         acc += s_A[ty][k] * s_B[k][tx];
    //     }
    //     __syncthreads();
    // }
    // C[row * N + col] = acc;
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Tiled GEMM with Boundary Checks (Arbitrary M, N, K)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_tiled_boundary(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
) {
    const int TILE = 16;
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    // TODO: Implement boundary-safe loading: if index >= dimension, load 0.0f
    // Accumulate dot product and if (row < M && col < N) write C[row * N + col] = acc;
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Padded Shared Tile GEMM (16x17 Tile to avoid bank conflicts)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_padded_tile(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    const int TILE = 16;
    __shared__ float s_A[TILE][TILE + 1]; // +1 padding
    __shared__ float s_B[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    // TODO: Same tiled GEMM logic utilizing s_A[ty][tx] and s_B[ty][tx]
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Tiled Vector-Matrix Multiplication (GEMV: y = x * A)
// -----------------------------------------------------------------------------
__global__ void kernel_tiled_gemv(
    const float *x,
    const float *A,
    float *y,
    int M, int N
) {
    // x is 1xM, A is MxN, y is 1xN
    // Block computes a tile of N columns
    // TODO: Tiled dot product of vector x with columns of A
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.2 Shared Memory Matrix Tiling (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Tiled Dot
    {
        int n = 256;
        std::vector<float> h_a(n, 2.0f), h_b(n, 3.0f);
        float h_sum = 0.0f;
        float *d_a, *d_b, *d_sum;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_sum, sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sum, 0, sizeof(float));

        kernel_tiled_dot_1d<<<1, 256>>>(d_a, d_b, d_sum, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_sum - (256 * 6.0f)) < 1e-3f);
        reportStatus("Problem 1: 1D Tiled Dot Product using Shared Memory", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_sum);
    }

    // Test 2: 16x16 Tiled GEMM
    {
        int N = 32;
        std::vector<float> h_A(N * N, 1.0f), h_B(N * N, 2.0f), h_C(N * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(N / 16, N / 16);
        kernel_gemm_tiled_16x16<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // Result of multiplying two matrices of 1.0 and 2.0 with K=32 is 32 * 2 = 64.0
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - 64.0f) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 2: 2D Block-Tiled GEMM (Square 16x16 Tile)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 3: Tiled GEMM with Boundary
    {
        int M = 20, N = 25, K = 30;
        std::vector<float> h_A(M * K, 1.5f), h_B(K * N, 2.0f), h_C(M * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        kernel_gemm_tiled_boundary<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float expected = K * (1.5f * 2.0f);
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - expected) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 3: Tiled GEMM with Boundary Checks (Arbitrary M, N, K)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 4: Padded GEMM
    {
        int N = 32;
        std::vector<float> h_A(N * N, 2.0f), h_B(N * N, 3.0f), h_C(N * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(N / 16, N / 16);
        kernel_gemm_padded_tile<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - (N * 6.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 4: Padded Shared Tile GEMM (16x17 Tile)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 5: GEMV
    {
        int M = 32, N = 32;
        std::vector<float> h_x(M, 1.0f), h_A(M * N, 2.0f), h_y(N, 0.0f);
        float *d_x, *d_A, *d_y;
        cudaMalloc(&d_x, M * sizeof(float));
        cudaMalloc(&d_A, M * N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaMemcpy(d_x, h_x.data(), M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

        kernel_tiled_gemv<<<1, 32>>>(d_x, d_A, d_y, M, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_y[i] - (M * 2.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 5: Tiled Vector-Matrix Multiplication (GEMV: y = x * A)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_x); cudaFree(d_A); cudaFree(d_y);
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
