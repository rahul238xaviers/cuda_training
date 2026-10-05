// ==============================================================================
// Module 7.5: GEMM & Linear Projection Suite — Beginner Workbook
// ==============================================================================
// In this workbook, you will implement foundational General Matrix Multiplication
// (GEMM) kernels used in modern LLM training pipelines:
// 1. Naive row-major GEMM: C = A * B
// 2. 2D Block-Tiled GEMM with Shared Memory (16x16 tile)
// 3. Transposed-B Projection GEMM: C = A * B^T (B stored as [N x K])
// 4. Batched GEMM: C[b] = A[b] * B[b] for multi-head projections
// 5. Linear Projection with Bias Addition: C = A * B + bias
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ==============================================================================
// Exercise 1: Naive Row-Major GEMM
// Computes C [M x N] = A [M x K] * B [K x N]
// Each thread computes one element C[row, col].
// Grid: (ceil(N / 16), ceil(M / 16)), Block: (16, 16)
// ==============================================================================
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // TODO: Calculate global row and col.
    // Boundary check: row < M && col < N
    // Accumulate sum over k from 0 to K-1: sum += A[row * K + k] * B[k * N + col]
    // Write out to C[row * N + col]
}

// ==============================================================================
// Exercise 2: 2D Block-Tiled Shared Memory GEMM (16x16 Tile)
// TILE_DIM = 16. Each block computes a [16 x 16] tile of C.
// Threads load sub-tiles of A [16 x 16] and B [16 x 16] into shared memory,
// synchronize, accumulate partial dot-products, and iterate across K.
// ==============================================================================
const int TILE_DIM = 16;

__global__ void tiled_gemm_16x16_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // TODO:
    // 1. Identify row = blockIdx.y * TILE_DIM + threadIdx.y
    //             col = blockIdx.x * TILE_DIM + threadIdx.x
    // 2. Initialize float sum = 0.0f
    // 3. Loop over tiles: for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    //    - Load tile of A into s_A[threadIdx.y][threadIdx.x] with boundary check (row < M && t * TILE_DIM + threadIdx.x < K)
    //    - Load tile of B into s_B[threadIdx.y][threadIdx.x] with boundary check (t * TILE_DIM + threadIdx.y < K && col < N)
    //    - __syncthreads()
    //    - Inner loop: for (int k = 0; k < TILE_DIM; ++k) sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    //    - __syncthreads()
    // 4. If row < M && col < N, write C[row * N + col] = sum;
}

// ==============================================================================
// Exercise 3: Transposed-B Projection GEMM
// In linear layers, weights are typically stored row-major as [N x K] (out_features x in_features).
// Computes C [M x N] = A [M x K] * B^T [K x N], where B is stored in memory as [N x K].
// Therefore, B^T[k, j] = B[j, k] = B[col * K + k].
// ==============================================================================
__global__ void gemm_trans_b_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // TODO:
    // Implement tiled GEMM where B is indexed as B[col * K + k] instead of B[k * N + col].
    // Specifically, when loading s_B[threadIdx.y][threadIdx.x]:
    // let k_idx = t * TILE_DIM + threadIdx.y;
    // let n_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    // If n_idx < N && k_idx < K, s_B[threadIdx.y][threadIdx.x] = B[n_idx * K + k_idx];
    // else 0.0f.
}

// ==============================================================================
// Exercise 4: Batched Matrix Multiplication
// Computes C[b] = A[b] * B[b] for b in [0, batch_size - 1].
// Each batch has A [M x K], B [K x N], C [M x N].
// gridDim.z = batch_size.
// ==============================================================================
__global__ void batched_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K, int batch_size) {
    // TODO:
    // 1. Identify batch b = blockIdx.z. Return if b >= batch_size.
    // 2. Compute base pointers for batch b:
    //    const float* A_b = A + b * (M * K);
    //    const float* B_b = B + b * (K * N);
    //    float* C_b = C + b * (M * N);
    // 3. Compute row = blockIdx.y * blockDim.y + threadIdx.y
    //            col = blockIdx.x * blockDim.x + threadIdx.x
    // 4. If row < M && col < N, accumulate dot product over K and write to C_b[row * N + col].
}

// ==============================================================================
// Exercise 5: Linear Projection with Fused Bias Addition
// Computes C [M x N] = A [M x K] * B [K x N] + bias [N]
// Bias is 1D tensor of size N, broadcasted across all M rows.
// ==============================================================================
__global__ void gemm_proj_bias_kernel(const float* A, const float* B, const float* bias,
                                      float* C, int M, int N, int K) {
    // TODO:
    // 1. Compute dot product C[row, col] = sum_{k=0}^{K-1} A[row * K + k] * B[k * N + col]
    // 2. If row < M && col < N:
    //    C[row * N + col] = sum + bias[col];
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Naive GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 48, K = 64;
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = std::sin(static_cast<float>(i + 1));
        for (int i = 0; i < K * N; ++i) h_B[i] = std::cos(static_cast<float>(i + 1));

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += h_A[m * K + k] * h_B[k * N + n];
                h_ref[m * N + n] = acc;
            }
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 1: Naive GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Naive GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 2: Tiled 16x16 Shared Memory GEMM
    // --------------------------------------------------------------------------
    {
        int M = 48, N = 64, K = 80;
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.5f * std::sin(static_cast<float>(i));
        for (int i = 0; i < K * N; ++i) h_B[i] = 0.5f * std::cos(static_cast<float>(i));

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += h_A[m * K + k] * h_B[k * N + n];
                h_ref[m * N + n] = acc;
            }
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        tiled_gemm_16x16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Tiled 16x16 Shared Memory GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Tiled 16x16 Shared Memory GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 3: Transposed-B GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 48;
        std::vector<float> h_A(M * K), h_B(N * K), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.2f * (i % 7);
        for (int i = 0; i < N * K; ++i) h_B[i] = 0.3f * (i % 5);

        // B is [N x K], so B^T is [K x N] where B^T[k, n] = B[n * K + k]
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += h_A[m * K + k] * h_B[n * K + k];
                h_ref[m * N + n] = acc;
            }
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_trans_b_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Transposed-B GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Transposed-B GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 4: Batched GEMM
    // --------------------------------------------------------------------------
    {
        int B_dim = 4, M = 16, N = 16, K = 32;
        int batch_size = B_dim;
        std::vector<float> h_A(batch_size * M * K), h_B(batch_size * K * N),
                           h_C(batch_size * M * N, 0.0f), h_ref(batch_size * M * N, 0.0f);

        for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = 0.1f * ((i % 11) + 1);
        for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = 0.2f * ((i % 13) + 1);

        for (int b = 0; b < batch_size; ++b) {
            const float* A_b = h_A.data() + b * M * K;
            const float* B_b = h_B.data() + b * K * N;
            float* C_ref_b = h_ref.data() + b * M * N;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) acc += A_b[m * K + k] * B_b[k * N + n];
                    C_ref_b[m * N + n] = acc;
                }
            }
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size);
        batched_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (size_t i = 0; i < h_C.size(); ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: Batched GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Batched GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 5: Linear Projection with Bias Addition
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_A(M * K), h_B(K * N), h_bias(N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.25f * (i % 9);
        for (int i = 0; i < K * N; ++i) h_B[i] = 0.35f * (i % 7);
        for (int i = 0; i < N; ++i) h_bias[i] = 1.5f * (i + 1);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += h_A[m * K + k] * h_B[k * N + n];
                h_ref[m * N + n] = acc + h_bias[n];
            }
        }

        float *d_A, *d_B, *d_bias, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bias, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_proj_bias_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 5: Projection with Bias] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Projection with Bias] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
