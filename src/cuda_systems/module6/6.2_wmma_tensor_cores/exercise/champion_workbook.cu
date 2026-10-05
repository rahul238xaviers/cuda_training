// ============================================================================
// Module 6.2: WMMA Tensor Cores - Champion Workbook
//
// Concepts Covered:
//   - 2D Grid of Blocks WMMA GEMM (arbitrary M, N, K multiples of 16)
//   - Scaled Dot-Product Attention Tile (Q * K^T with in-register 1/sqrt(d_k))
//   - Transposed Matrix Multiplication (A * B^T via layout manipulation)
//   - Fused WMMA GEMM + GELU activation in registers
//   - Boundary predication for non-multiple-of-16 dimensions
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(1); \
        } \
    } while (0)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Exercise 1: 2D Grid of Blocks WMMA GEMM
// Grid dimensions: gridDim.x = N / 16, gridDim.y = M / 16.
// Each block has 1 warp (32 threads) and computes one 16x16 output tile across K.
// A is M x K (row-major, lda = K), B is K x N (col-major, ldb = K).
// C is M x N (row-major, ldc = N).
// ============================================================================
__global__ void wmma_grid_gemm_kernel(const half* A, const half* B, float* C,
                                      int M, int N, int K) {
    int tile_r = blockIdx.y; // row index in 16x16 tiles
    int tile_c = blockIdx.x; // col index in 16x16 tiles

    // TODO:
    // 1. Initialize c_frag to 0.0f.
    // 2. Loop k = 0; k < K; k += 16:
    //    Load a_frag from A + tile_r * 16 * K + k (lda = K).
    //    Load b_frag from B + tile_c * 16 * K + k (ldb = K).
    //    mma_sync(c_frag, a_frag, b_frag, c_frag).
    // 3. Store c_frag to C + tile_r * 16 * N + tile_c * 16 (ldc = N).
}

// ============================================================================
// Exercise 2: Scaled Dot-Product Attention Tile: Q * K^T * (1 / sqrt(d_k))
// Q is 16 x d_k (row-major), K_mat is 16 x d_k (row-major, representing K^T when viewed col-major).
// Output is 16 x 16 attention score tile S.
// Scale the accumulated c_frag in-register by scale = 1.0f / sqrtf((float)d_k) before storing.
// ============================================================================
__global__ void wmma_attention_qk_kernel(const half* Q, const half* K_mat, float* S, int d_k) {
    // TODO:
    // 1. Accumulate Q * K^T over d_k in steps of 16 into c_frag.
    //    (Q is row-major, K_mat is col-major in fragment to perform transpose).
    // 2. Compute float scale = rsqrtf((float)d_k);
    // 3. Scale all c_frag.x[t] by scale.
    // 4. Store to S with stride 16.
}

// ============================================================================
// Exercise 3: Explicit A * B^T Transpose via WMMA Layout
// A is 16 x 16 (row-major). B is 16 x 16 (row-major).
// We want to compute C = A * B^T.
// Notice: If B is row-major, B[r][c] = B_flat[r * 16 + c].
// When loaded with wmma::col_major, element (r, c) is loaded as (c, r) = B^T!
// ============================================================================
__global__ void wmma_gemm_transpose_b_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // 1. Load A as row-major.
    // 2. Load B as col-major fragment (which interprets row-major memory as B^T!).
    // 3. Compute mma_sync into c_frag.
    // 4. Store to C.
}

// ============================================================================
// Exercise 4: Fused WMMA GEMM + GELU Activation in Registers
// Compute C = GELU(A * B) where GELU is applied directly to accumulator registers.
// GELU(x) = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)))
// ============================================================================
__global__ void wmma_fused_gemm_gelu_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // 1. Compute A * B into c_frag (B is col-major).
    // 2. Loop over c_frag.num_elements and apply GELU formula.
    // 3. Store result to C.
}

// ============================================================================
// Exercise 5: Boundary Predicated WMMA Store (Non-Multiple Dimensions)
// Suppose M = 20, N = 20 (not divisible by 16).
// Compute 16x16 tile, but store through shared memory staging so only valid
// indices (r < M && c < N) are written to global memory.
// ============================================================================
__global__ void wmma_boundary_predicated_store_kernel(const half* A, const half* B,
                                                      float* C, int M, int N) {
    __shared__ float s_tile[16][16];
    int tid = threadIdx.x; // 32 threads

    // TODO:
    // 1. Declare and compute 16x16 tile into c_frag.
    // 2. Store c_frag into s_tile: wmma::store_matrix_sync(&s_tile[0][0], c_frag, 16, wmma::mem_row_major);
    // 3. __syncthreads();
    // 4. Cooperatively copy from s_tile to C with boundary check:
    //    for (int i = tid; i < 256; i += 32) {
    //        int r = i / 16;
    //        int c = i % 16;
    //        if (r < M && c < N) {
    //            C[r * N + c] = s_tile[r][c];
    //        }
    //    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: 2D Grid WMMA GEMM ---
    {
        const int M = 32, N = 32, K = 32;
        std::vector<half> h_A(M * K, __float2half(1.0f));
        std::vector<half> h_B(K * N, __float2half(1.0f));
        std::vector<float> h_C(M * N, 0.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 grid(N / 16, M / 16);
        wmma_grid_gemm_kernel<<<grid, 32>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 1.0 * 1.0 * 32 = 32.0f
        for (int i = 0; i < M * N; ++i) {
            if (std::abs(h_C[i] - 32.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 32.0f) {
            std::cout << "Test 1 Passed: 2D Grid of WMMA blocks." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: 2D Grid of WMMA blocks (got " << h_C[0] << ", exp 32)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 2: Attention QK^T Scaling ---
    {
        const int d_k = 64;
        std::vector<half> h_Q(16 * d_k, __float2half(1.0f));
        std::vector<half> h_K(16 * d_k, __float2half(1.0f));
        std::vector<float> h_S(16 * 16, 0.0f);

        half *d_Q, *d_K; float *d_S;
        CUDA_CHECK(cudaMalloc(&d_Q, 16 * d_k * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_K, 16 * d_k * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_S, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), 16 * d_k * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), 16 * d_k * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_S, 0, 16 * 16 * sizeof(float)));

        wmma_attention_qk_kernel<<<1, 32>>>(d_Q, d_K, d_S, d_k);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Dot product = 64.0f, scaled by 1/sqrt(64) = 1/8 = 8.0f
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_S[i] - 8.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_S[0] == 8.0f) {
            std::cout << "Test 2 Passed: Attention QK^T tile with in-fragment 1/sqrt(d_k) scale." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Attention QK^T tile (got " << h_S[0] << ", exp 8.0)." << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_S);
    }

    // --- Test 3: A * B^T Transpose GEMM ---
    {
        std::vector<half> h_A(16 * 16), h_B(16 * 16);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                h_A[r * 16 + c] = __float2half(1.0f);
                // B row r has value r + 1
                h_B[r * 16 + c] = __float2half(static_cast<float>(r + 1));
            }
        }

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_gemm_transpose_b_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(16 * 16);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // In A * B^T, C[r][c] = sum_k (A[r][k] * B[c][k]) = 16 * (c + 1)
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                float exp = 16.0f * static_cast<float>(c + 1);
                if (std::abs(h_C[r * 16 + c] - exp) > 0.1f) { ok = false; break; }
            }
        }
        if (ok && h_C[0] == 16.0f) {
            std::cout << "Test 3 Passed: A * B^T transpose via layout." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: A * B^T transpose via layout (got " << h_C[0] << ", exp 16)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 4: Fused GEMM + GELU ---
    {
        std::vector<half> h_A(16 * 16, __float2half(0.25f));
        std::vector<half> h_B(16 * 16, __float2half(0.5f));
        std::vector<float> h_C(16 * 16, 0.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_fused_gemm_gelu_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        // Dot product = 0.25 * 0.5 * 16 = 2.0f
        float x = 2.0f;
        float exp_gelu = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));

        bool ok = true;
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - exp_gelu) > 0.05f) { ok = false; break; }
        }
        if (ok && h_C[0] != 0.0f) {
            std::cout << "Test 4 Passed: Fused WMMA GEMM + GELU activation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused WMMA GEMM + GELU activation (got " << h_C[0] << ", exp " << exp_gelu << ")." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 5: Boundary Predicated Store ---
    {
        const int M = 10, N = 12; // sub-16 matrix!
        std::vector<half> h_A(16 * 16, __float2half(1.0f));
        std::vector<half> h_B(16 * 16, __float2half(1.0f));
        std::vector<float> h_C(M * N, -1.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

        wmma_boundary_predicated_store_kernel<<<1, 32>>>(d_A, d_B, d_C, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // All valid entries should be 16.0f
        for (int i = 0; i < M * N; ++i) {
            if (std::abs(h_C[i] - 16.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 16.0f) {
            std::cout << "Test 5 Passed: Boundary predicated WMMA store." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Boundary predicated WMMA store." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
