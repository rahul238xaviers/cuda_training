// ============================================================================
// Module 6.2: WMMA Tensor Cores - Intermediate Workbook
//
// Concepts Covered:
//   - Multi-warp 2D tile partitioning (32x32 tile via 4 warps)
//   - K-dimension accumulation loop for arbitrary K (e.g. K=64)
//   - Staging data in shared memory before WMMA loads
//   - Fused bias addition
//   - In-fragment activation function (ReLU)
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
// Exercise 1: Multi-Warp 32x32 Tiling (4 Warps = 128 Threads)
// A is 32x16 (row-major), B is 16x32 (row-major), C is 32x32 (row-major).
// Block has 128 threads (4 warps).
// Warp 0 computes C[0..15, 0..15]
// Warp 1 computes C[0..15, 16..31]
// Warp 2 computes C[16..31, 0..15]
// Warp 3 computes C[16..31, 16..31]
// ============================================================================
__global__ void wmma_32x32_multi_warp_gemm_kernel(const half* A, const half* B, float* C) {
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2; // 0 or 1 (row tile 0..15 or 16..31)
    int warp_col = warp_id % 2; // 0 or 1 (col tile 0..15 or 16..31)

    // TODO:
    // 1. Calculate A pointer offset for this warp: warp_row * 16 * 16 (since lda = 16)
    // 2. Calculate B pointer offset for this warp: warp_col * 16 (since ldb = 32)
    // 3. Calculate C pointer offset for this warp: warp_row * 16 * 32 + warp_col * 16 (ldc = 32)
    // 4. Declare a_frag, b_frag, c_frag. Fill c_frag with 0.0f.
    // 5. Load a_frag (lda = 16), load b_frag (ldb = 32).
    // 6. mma_sync, then store c_frag to C offset (ldc = 32).
}

// ============================================================================
// Exercise 2: K-Dimension Accumulation Loop (K = 64)
// A is 16x64 (row-major), B is 64x16 (col-major), C is 16x16 (row-major).
// A single warp loops over K in chunks of 16:
// for (int k = 0; k < 64; k += 16) { load A tile, load B tile, mma_sync }
// Store accumulated c_frag into C.
// ============================================================================
__global__ void wmma_gemm_k_loop_kernel(const half* A, const half* B, float* C, int K) {
    // TODO:
    // 1. Declare a_frag, b_frag, c_frag.
    // 2. Initialize c_frag to 0.0f.
    // 3. Loop k = 0; k < K; k += 16:
    //    Load A tile at A + k with lda = K (64).
    //    Load B tile at B + k with ldb = K (64, since B is col-major, leading dimension is K).
    //    mma_sync(c_frag, a_frag, b_frag, c_frag).
    // 4. Store c_frag to C with ldc = 16.
}

// ============================================================================
// Exercise 3: Shared Memory Staging for Tensor Cores
// Load 16x16 A and B into __shared__ memory first, synchronize,
// then load fragments from shared memory and compute GEMM.
// ============================================================================
__global__ void wmma_shared_mem_stage_kernel(const half* A, const half* B, float* C) {
    __shared__ half s_A[16 * 16];
    __shared__ half s_B[16 * 16];
    int tid = threadIdx.x; // 32 threads in block

    // TODO:
    // 1. Cooperatively load A and B into s_A and s_B.
    //    Since there are 256 elements and 32 threads, each thread loads 8 elements:
    //    for (int i = tid; i < 256; i += 32) { s_A[i] = A[i]; s_B[i] = B[i]; }
    // 2. __syncthreads();
    // 3. Load a_frag from s_A (lda = 16), b_frag from s_B (col-major, ldb = 16).
    // 4. Compute mma_sync and store result to C.
}

// ============================================================================
// Exercise 4: Fused Bias Addition
// Compute C = A * B + bias where A is 16x16, B is 16x16 (col-major),
// and bias is 1x16 (applied to each row of C).
// ============================================================================
__global__ void wmma_fused_bias_kernel(const half* A, const half* B, const float* bias, float* C) {
    // TODO:
    // 1. Compute C = A * B into c_frag using WMMA.
    // 2. Store c_frag into C.
    // 3. Cooperative elementwise addition: add bias[col] to C[row * 16 + col].
    //    int tid = threadIdx.x;
    //    for (int i = tid; i < 256; i += 32) { C[i] += bias[i % 16]; }
}

// ============================================================================
// Exercise 5: In-Fragment ReLU Activation
// Compute C = ReLU(A * B)
// Use in-fragment element modification: c_frag.x[t] = fmaxf(0.0f, c_frag.x[t]).
// ============================================================================
__global__ void wmma_relu_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // 1. Compute A * B into c_frag (B is col-major).
    // 2. Apply ReLU to all c_frag.x[t] elements.
    // 3. Store to C.
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: 32x32 Multi-Warp GEMM ---
    {
        // A is 32x16, B is 16x32, C is 32x32
        std::vector<half> h_A(32 * 16, __float2half(1.0f));
        std::vector<half> h_B(16 * 32, __float2half(2.0f));
        std::vector<float> h_C(32 * 32, 0.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 32 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 32 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 32 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 32 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 32 * 32 * sizeof(float)));

        wmma_32x32_multi_warp_gemm_kernel<<<1, 128>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 1.0 * 2.0 * 16 = 32.0f
        for (int i = 0; i < 32 * 32; ++i) {
            if (std::abs(h_C[i] - 32.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 32.0f) {
            std::cout << "Test 1 Passed: 32x32 multi-warp WMMA GEMM." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: 32x32 multi-warp WMMA GEMM (got " << h_C[0] << ", exp 32)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 2: K-Loop Accumulation ---
    {
        const int K = 64;
        std::vector<half> h_A(16 * K, __float2half(1.0f));
        std::vector<half> h_B(K * 16, __float2half(1.5f)); // col-major (16 cols of length K)
        std::vector<float> h_C(16 * 16, 0.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, K * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_gemm_k_loop_kernel<<<1, 32>>>(d_A, d_B, d_C, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 1.0 * 1.5 * 64 = 96.0f
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - 96.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 96.0f) {
            std::cout << "Test 2 Passed: K-loop WMMA accumulation (K=64)." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: K-loop WMMA accumulation (got " << h_C[0] << ", exp 96)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 3: Shared Memory Staged GEMM ---
    {
        std::vector<half> h_A(16 * 16, __float2half(3.0f));
        std::vector<half> h_B(16 * 16, __float2half(2.0f));
        std::vector<float> h_C(16 * 16, 0.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_shared_mem_stage_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 3.0 * 2.0 * 16 = 96.0f
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - 96.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 96.0f) {
            std::cout << "Test 3 Passed: Shared memory staged WMMA GEMM." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Shared memory staged WMMA GEMM (got " << h_C[0] << ", exp 96)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 4: Fused Bias Addition ---
    {
        std::vector<half> h_A(16 * 16, __float2half(1.0f));
        std::vector<half> h_B(16 * 16, __float2half(1.0f));
        std::vector<float> h_bias(16);
        for (int i = 0; i < 16; ++i) h_bias[i] = static_cast<float>(i);
        std::vector<float> h_C(16 * 16, 0.0f);

        half *d_A, *d_B; float *d_bias, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_bias, 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_fused_bias_kernel<<<1, 32>>>(d_A, d_B, d_bias, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                float exp = 16.0f + static_cast<float>(c);
                if (std::abs(h_C[r * 16 + c] - exp) > 0.1f) { ok = false; break; }
            }
        }
        if (ok && h_C[15] == 31.0f) {
            std::cout << "Test 4 Passed: Fused bias addition with WMMA GEMM." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused bias addition with WMMA GEMM." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    }

    // --- Test 5: In-Fragment ReLU Activation ---
    {
        // Set A elements to negative, so product is negative
        std::vector<half> h_A(16 * 16, __float2half(-1.0f));
        std::vector<half> h_B(16 * 16, __float2half(1.0f));
        std::vector<float> h_C(16 * 16, -99.0f);

        half *d_A, *d_B; float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), 16 * 16 * sizeof(float), cudaMemcpyHostToDevice));

        wmma_relu_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // (-1 * 1 * 16) = -16 -> ReLU(-16) = 0.0f
        for (int i = 0; i < 16 * 16; ++i) {
            if (h_C[i] != 0.0f) { ok = false; break; }
        }
        if (ok) {
            std::cout << "Test 5 Passed: In-fragment ReLU activation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: In-fragment ReLU activation (got " << h_C[0] << ", exp 0.0)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
