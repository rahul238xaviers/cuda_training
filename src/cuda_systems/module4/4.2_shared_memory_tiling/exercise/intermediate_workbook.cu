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
// PROBLEM 1: Rectangular Shared Memory Tiling (BM=32, BN=32, BK=8)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_rect_tile(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
) {
    const int BM = 32, BN = 32, BK = 8;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;
    // TODO: Loop over K in chunks of BK = 8, load collaborative tiles, accumulate C
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 1D Thread Tiling (TM=2, TN=1, 2 Outputs per Thread)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_thread_tiling_1d(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    // Block size is 16x16 = 256 threads, but computes 32x16 output (each thread computes 2 elements: row0 and row1)
    const int BM = 32, BN = 16, BK = 16;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // TODO: Loop over K, load s_A and s_B
    // For each k: acc0 += s_A[ty * 2][k] * s_B[k][tx];
    //             acc1 += s_A[ty * 2 + 1][k] * s_B[k][tx];
    // Write out to C
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Transpose-B Tiled GEMM (C = A * B^T)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_trans_b_tiled(
    const float *A,
    const float *B, // B is stored as K x N in transpose layout (so row i is column i of original)
    float *C,
    int M, int N, int K
) {
    const int TILE = 16;
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    // TODO: Load s_A and s_B from row-major A and row-major B where B's rows are dot-product vectors!
    // Accumulate C[row * N + col]
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Fused Bias Addition in Tiled GEMM (C = A * B + bias)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_fused_bias(
    const float *A,
    const float *B,
    const float *bias, // 1 x N
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

    // TODO: Compute tiled matrix multiply, then add bias[col] in registers before writing C
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Batched Tiled GEMM
// -----------------------------------------------------------------------------
__global__ void kernel_batched_tiled_gemm(
    const float *A,
    const float *B,
    float *C,
    int batch_stride_A,
    int batch_stride_B,
    int batch_stride_C,
    int N
) {
    int b = blockIdx.z; // Batch index
    const float *cur_A = A + b * batch_stride_A;
    const float *cur_B = B + b * batch_stride_B;
    float *cur_C = C + b * batch_stride_C;

    // TODO: Perform tiled GEMM on cur_A and cur_B into cur_C
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.2 Shared Memory Matrix Tiling (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Rectangular Tiling
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 2.0f), h_C(M * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        kernel_gemm_rect_tile<<<1, 256>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - (K * 2.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 1: Rectangular Shared Memory Tiling (BM=32, BN=32, BK=8)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 2: 1D Thread Tiling
    {
        int N = 32;
        std::vector<float> h_A(N * N, 1.5f), h_B(N * N, 2.0f), h_C(N * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(N / 16, N / 32);
        kernel_gemm_thread_tiling_1d<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - (N * 3.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 2: 1D Thread Tiling (TM=2, TN=1, 2 Outputs/Thread)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 3: Transpose-B GEMM
    {
        int M = 16, N = 16, K = 32;
        std::vector<float> h_A(M * K, 2.0f), h_B(N * K, 3.0f), h_C(M * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, N * K * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        kernel_gemm_trans_b_tiled<<<1, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - (K * 6.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 3: Transpose-B Tiled GEMM (C = A * B^T)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 4: Fused Bias Add
    {
        int N = 32;
        std::vector<float> h_A(N * N, 1.0f), h_B(N * N, 2.0f), h_bias(N, 10.0f), h_C(N * N, 0.0f);
        float *d_A, *d_B, *d_bias, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_bias, N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(N / 16, N / 16);
        kernel_gemm_fused_bias<<<grid, block>>>(d_A, d_B, d_bias, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - (N * 2.0f + 10.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 4: Fused Bias Addition in Tiled GEMM (C = A * B + bias)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    }

    // Test 5: Batched GEMM
    {
        int batch = 4, N = 16;
        int matrix_size = N * N;
        std::vector<float> h_A(batch * matrix_size, 1.0f);
        std::vector<float> h_B(batch * matrix_size, 2.0f);
        std::vector<float> h_C(batch * matrix_size, 0.0f);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, h_A.size() * sizeof(float));
        cudaMalloc(&d_B, h_B.size() * sizeof(float));
        cudaMalloc(&d_C, h_C.size() * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(1, 1, batch);
        kernel_batched_tiled_gemm<<<grid, block>>>(d_A, d_B, d_C, matrix_size, matrix_size, matrix_size, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t i = 0; i < h_C.size(); ++i) {
            if (std::fabs(h_C[i] - (N * 2.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 5: Batched Tiled GEMM across Independent Matrices", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
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
