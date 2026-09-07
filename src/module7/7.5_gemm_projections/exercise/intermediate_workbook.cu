// ==============================================================================
// Module 7.5: GEMM & Linear Projection Suite — Intermediate Workbook
// ==============================================================================
// In this workbook, you will advance to production-grade GEMM techniques used
// in LLM training backpropagation and attention architectures:
// 1. Bank-Conflict-Free 32x32 Tiled GEMM with Shared Memory Padding
// 2. Backward Input Gradient GEMM: dX [M x K] = dO [M x N] * W [N x K]
// 3. Backward Weight Gradient GEMM: dW [N x K] = dO^T [N x M] * X [M x K]
// 4. Grouped-Query Attention (GQA) Projection with Head Broadcasting
// 5. 2x2 Register Micro-Tiled GEMM (Thread block computes 32x32 tile using 16x16 threads)
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

const int TILE_32 = 32;

// ==============================================================================
// Exercise 1: Bank-Conflict-Free 32x32 Tiled GEMM
// 32 banks on NVIDIA GPUs. Stride 32 causes a 32-way bank conflict on column loads!
// Allocate shared memory with 1 element padding:
// __shared__ float s_A[32][33];
// __shared__ float s_B[32][33];
// Block: (32, 32), Grid: (ceil(N / 32), ceil(M / 32))
// ==============================================================================
__global__ void gemm_padded_32x32_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    // TODO:
    // 1. Declare __shared__ float s_A[32][33]; and __shared__ float s_B[32][33];
    // 2. Map row = blockIdx.y * 32 + threadIdx.y, col = blockIdx.x * 32 + threadIdx.x
    // 3. Loop over K tiles in chunks of 32
    // 4. Guard against out-of-bounds when loading into s_A and s_B
    // 5. __syncthreads(), accumulate dot product for 32 steps, __syncthreads()
    // 6. Write out C[row * N + col]
}

// ==============================================================================
// Exercise 2: Backward Input Activation Gradient GEMM
// Forward: Y [M x N] = X [M x K] * W^T [K x N], where W is stored as [N x K].
// Backward: dX [M x K] = dO [M x N] * W [N x K].
// Here dO is [M x N], W is [N x K] in memory.
// Output dX is [M x K].
// Block: (16, 16), Grid: (ceil(K / 16), ceil(M / 16))
// ==============================================================================
__global__ void gemm_backward_input_kernel(const float* dO, const float* W, float* dX,
                                           int M, int N, int K) {
    // TODO:
    // Compute dX [M x K] = dO [M x N] * W [N x K]
    // dX[row, k] = sum_{n=0}^{N-1} dO[row * N + n] * W[n * K + k]
    // You may use naive or 16x16 shared memory tiling.
}

// ==============================================================================
// Exercise 3: Backward Weight Gradient GEMM
// In backpropagation, weight gradient is computed as:
// dW [N x K] += dO^T [N x M] * X [M x K]
// dO is physically stored as [M x N]. Transposed access: dO^T[n, m] = dO[m * N + n].
// X is physically stored as [M x K].
// Output dW is [N x K].
// Block: (16, 16), Grid: (ceil(K / 16), ceil(N / 16))
// ==============================================================================
__global__ void gemm_backward_weight_kernel(const float* dO, const float* X, float* dW,
                                            int M, int N, int K) {
    // TODO:
    // row in [0, N-1], col in [0, K-1]
    // dW[row, col] = sum_{m=0}^{M-1} dO[m * N + row] * X[m * K + col]
}

// ==============================================================================
// Exercise 4: Grouped-Query Attention (GQA) Head Projection
// In GQA, H_q query heads share H_kv key/value heads.
// group_size = H_q / H_kv.
// Each block processes 1 query head: blockIdx.z = q_head (0 to H_q - 1).
// The corresponding KV head index is: kv_head = q_head / group_size.
// This kernel projects Q using head-specific W_q [H_q, D, D] and
// broadcast-projects K using shared KV weights W_k [H_kv, D, D].
// Q_proj [H_q, S, D] = Q [H_q, S, D] * W_q [H_q, D, D]
// K_proj [H_q, S, D] = K [H_kv, S, D] * W_k [H_kv, D, D] (broadcasted to H_q)
// To keep things focused, implement K_proj broadcasting:
// Inputs: K [H_kv * S * D], W_k [H_kv * D * D]
// Output: K_proj [H_q * S * D]
// gridDim.z = H_q, gridDim.y = (S + 15)/16, gridDim.x = (D + 15)/16
// ==============================================================================
__global__ void gemm_gqa_kv_broadcast_kernel(const float* K, const float* W_k,
                                             float* K_proj,
                                             int S, int D, int H_q, int H_kv, int group_size) {
    // TODO:
    // 1. Identify q_head = blockIdx.z. Return if q_head >= H_q.
    // 2. Determine kv_head = q_head / group_size;
    // 3. Pointer for K of this KV head: const float* K_head = K + kv_head * (S * D);
    // 4. Pointer for W_k of this KV head: const float* W_head = W_k + kv_head * (D * D);
    // 5. Pointer for K_proj of this query head: float* Out_head = K_proj + q_head * (S * D);
    // 6. row = blockIdx.y * blockDim.y + threadIdx.y (token index in S)
    //    col = blockIdx.x * blockDim.x + threadIdx.x (feature index in D)
    // 7. If row < S && col < D:
    //    acc = sum_{d=0}^{D-1} K_head[row * D + d] * W_head[d * D + col]
    //    Out_head[row * D + col] = acc;
}

// ==============================================================================
// Exercise 5: 2x2 Register Micro-Tiled GEMM
// Block size: (16, 16) threads = 256 threads.
// Each thread computes a 2x2 sub-matrix of C.
// Therefore, the block computes a (16*2) x (16*2) = 32 x 32 tile of C.
// Shared memory:
// __shared__ float s_A[32][33];
// __shared__ float s_B[32][33];
// Each thread loads 4 elements into shared memory, and accumulates 2x2 in registers:
// float c[2][2] = {0};
// ==============================================================================
__global__ void gemm_reg_tiled_2x2_kernel(const float* A, const float* B, float* C,
                                          int M, int N, int K) {
    __shared__ float s_A[32][33];
    __shared__ float s_B[32][33];

    // TODO:
    // 1. Block covers C from [blockIdx.y * 32, blockIdx.x * 32]
    // 2. Thread has threadIdx.x in [0, 15], threadIdx.y in [0, 15]
    //    Thread index tid = threadIdx.y * 16 + threadIdx.x (0 to 255)
    // 3. Load s_A and s_B (32x32 = 1024 elements each):
    //    Each of the 256 threads loads 4 elements of A and 4 elements of B into shared memory per tile!
    //    Or simpler mapping:
    //    tid loads (row0, col0), (row1, col1), etc.
    // 4. In the compute loop over k from 0 to 31:
    //    float a0 = s_A[threadIdx.y * 2 + 0][k];
    //    float a1 = s_A[threadIdx.y * 2 + 1][k];
    //    float b0 = s_B[k][threadIdx.x * 2 + 0];
    //    float b1 = s_B[k][threadIdx.x * 2 + 1];
    //    c[0][0] += a0 * b0;
    //    c[0][1] += a0 * b1;
    //    c[1][0] += a1 * b0;
    //    c[1][1] += a1 * b1;
    // 5. Write the 4 accumulated values into C:
    //    C[(blockIdx.y * 32 + threadIdx.y * 2 + 0) * N + (blockIdx.x * 32 + threadIdx.x * 2 + 0)] = c[0][0]
    //    ... (with boundary checks)
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Bank-Conflict-Free 32x32 Tiled GEMM
    // --------------------------------------------------------------------------
    {
        int M = 64, N = 64, K = 64;
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.2f * std::cos(static_cast<float>(i));
        for (int i = 0; i < K * N; ++i) h_B[i] = 0.3f * std::sin(static_cast<float>(i));

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

        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        gemm_padded_32x32_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
            std::cout << "[Test 1: 32x32 Padded GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: 32x32 Padded GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 2: Backward Input Activation Gradient GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 48, K = 40;
        std::vector<float> h_dO(M * N), h_W(N * K), h_dX(M * K, 0.0f), h_ref(M * K, 0.0f);
        for (int i = 0; i < M * N; ++i) h_dO[i] = 0.1f * (i % 7);
        for (int i = 0; i < N * K; ++i) h_W[i] = 0.2f * (i % 11);

        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                float acc = 0.0f;
                for (int n = 0; n < N; ++n) acc += h_dO[m * N + n] * h_W[n * K + k];
                h_ref[m * K + k] = acc;
            }
        }

        float *d_dO, *d_W, *d_dX;
        CHECK_CUDA(cudaMalloc(&d_dO, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W, N * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dX, M * K * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dX, 0, M * K * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (M + 15) / 16);
        gemm_backward_input_kernel<<<grid, block>>>(d_dO, d_W, d_dX, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dX.data(), d_dX, M * K * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * K; ++i) {
            if (std::fabs(h_dX[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Backward Input Gradient] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Backward Input Gradient] FAILED" << std::endl;
        }

        cudaFree(d_dO); cudaFree(d_W); cudaFree(d_dX);
    }

    // --------------------------------------------------------------------------
    // Test 3: Backward Weight Gradient GEMM
    // --------------------------------------------------------------------------
    {
        int M = 48, N = 32, K = 32;
        std::vector<float> h_dO(M * N), h_X(M * K), h_dW(N * K, 0.0f), h_ref(N * K, 0.0f);
        for (int i = 0; i < M * N; ++i) h_dO[i] = 0.15f * ((i % 5) + 1);
        for (int i = 0; i < M * K; ++i) h_X[i] = 0.25f * ((i % 7) + 1);

        // dW[n, k] = sum_{m=0}^{M-1} dO[m, n] * X[m, k]
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                float acc = 0.0f;
                for (int m = 0; m < M; ++m) acc += h_dO[m * N + n] * h_X[m * K + k];
                h_ref[n * K + k] = acc;
            }
        }

        float *d_dO, *d_X, *d_dW;
        CHECK_CUDA(cudaMalloc(&d_dO, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_X, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dW, N * K * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dW, 0, N * K * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (N + 15) / 16);
        gemm_backward_weight_kernel<<<grid, block>>>(d_dO, d_X, d_dW, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dW.data(), d_dW, N * K * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N * K; ++i) {
            if (std::fabs(h_dW[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Backward Weight Gradient] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Backward Weight Gradient] FAILED" << std::endl;
        }

        cudaFree(d_dO); cudaFree(d_X); cudaFree(d_dW);
    }

    // --------------------------------------------------------------------------
    // Test 4: GQA KV Head Broadcasting Projection
    // --------------------------------------------------------------------------
    {
        int H_q = 8, H_kv = 2, group_size = 4;
        int S = 16, D = 16;
        std::vector<float> h_K(H_kv * S * D), h_W_k(H_kv * D * D),
                           h_K_proj(H_q * S * D, 0.0f), h_ref(H_q * S * D, 0.0f);

        for (size_t i = 0; i < h_K.size(); ++i) h_K[i] = 0.1f * ((i % 13) + 1);
        for (size_t i = 0; i < h_W_k.size(); ++i) h_W_k[i] = 0.05f * ((i % 17) + 1);

        for (int q_h = 0; q_h < H_q; ++q_h) {
            int kv_h = q_h / group_size;
            const float* K_head = h_K.data() + kv_h * (S * D);
            const float* W_head = h_W_k.data() + kv_h * (D * D);
            float* Out_head = h_ref.data() + q_h * (S * D);
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int k = 0; k < D; ++k) acc += K_head[s * D + k] * W_head[k * D + d];
                    Out_head[s * D + d] = acc;
                }
            }
        }

        float *d_K, *d_W_k, *d_K_proj;
        CHECK_CUDA(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_k, h_W_k.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K_proj, h_K_proj.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_k, h_W_k.data(), h_W_k.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_K_proj, 0, h_K_proj.size() * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (S + 15) / 16, H_q);
        gemm_gqa_kv_broadcast_kernel<<<grid, block>>>(d_K, d_W_k, d_K_proj, S, D, H_q, H_kv, group_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_K_proj.data(), d_K_proj, h_K_proj.size() * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (size_t i = 0; i < h_K_proj.size(); ++i) {
            if (std::fabs(h_K_proj[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: GQA KV Head Broadcasting] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: GQA KV Head Broadcasting] FAILED" << std::endl;
        }

        cudaFree(d_K); cudaFree(d_W_k); cudaFree(d_K_proj);
    }

    // --------------------------------------------------------------------------
    // Test 5: 2x2 Register Micro-Tiled GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.2f * ((i % 11) + 1);
        for (int i = 0; i < K * N; ++i) h_B[i] = 0.3f * ((i % 7) + 1);

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
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        gemm_reg_tiled_2x2_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
            std::cout << "[Test 5: 2x2 Register Tiled GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: 2x2 Register Tiled GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
