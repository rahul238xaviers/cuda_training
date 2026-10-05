// ============================================================================
// Module 6.2: WMMA Tensor Cores - Beginner Workbook
//
// Concepts Covered:
//   - Introduction to nvcuda::wmma namespace
//   - wmma::fragment types (matrix_a, matrix_b, accumulator)
//   - wmma::fill_fragment, load_matrix_sync, mma_sync, store_matrix_sync
//   - Col-major vs Row-major layouts in WMMA
//   - In-fragment scaling using .num_elements
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
// Exercise 1: Single-Warp 16x16 Matrix Multiplication (B is Col-Major)
// Compute C = A * B where A is 16x16 row-major, B is 16x16 col-major.
// C is 16x16 row-major float.
// ============================================================================
__global__ void wmma_gemm_col_b_kernel(const half* A, const half* B, float* C) {
    // 1 warp = 32 threads
    // TODO:
    // 1. Declare fragments:
    //    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    //    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    //    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    // 2. Initialize c_frag to 0.0f using wmma::fill_fragment.
    // 3. Load A and B using wmma::load_matrix_sync(frag, ptr, lda).
    // 4. Perform mma: wmma::mma_sync(c_frag, a_frag, b_frag, c_frag).
    // 5. Store result into C using wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major).
}

// ============================================================================
// Exercise 2: Single-Warp 16x16 Matrix Multiplication (B is Row-Major)
// Compute C = A * B where A is 16x16 row-major, B is 16x16 row-major.
// ============================================================================
__global__ void wmma_gemm_row_b_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // Similar to Exercise 1, but declare matrix_b fragment with wmma::row_major.
}

// ============================================================================
// Exercise 3: Single-Warp FMA with Pre-loaded Accumulator
// Compute C = A * B + C_init where C_init is already in C.
// ============================================================================
__global__ void wmma_fma_accum_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // 1. Load A, B fragments.
    // 2. Load existing C into accumulator fragment:
    //    wmma::load_matrix_sync(c_frag, C, 16, wmma::mem_row_major);
    // 3. Compute mma: wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // 4. Store updated c_frag back to C.
}

// ============================================================================
// Exercise 4: In-Fragment Element Scaling
// Compute C = alpha * (A * B)
// Scale the accumulator fragment elements in registers before writing to global memory.
// ============================================================================
__global__ void wmma_scaled_gemm_kernel(const half* A, const half* B, float* C, float alpha) {
    // TODO:
    // 1. Compute C = A * B into c_frag.
    // 2. Multiply each element in c_frag:
    //    for (int t = 0; t < c_frag.num_elements; ++t) {
    //        c_frag.x[t] *= alpha;
    //    }
    // 3. Store c_frag to C.
}

// ============================================================================
// Exercise 5: 16x32 Matrix Multiplication (Two 16x16 Tiles)
// A is 16x16 row-major. B is 16x32 row-major. C is 16x32 row-major.
// A single warp computes tile 0 (columns 0..15) then tile 1 (columns 16..31).
// Note: leading dimension ldb = 32, ldc = 32.
// ============================================================================
__global__ void wmma_16x32_gemm_kernel(const half* A, const half* B, float* C) {
    // TODO:
    // 1. Load A fragment (lda = 16).
    // 2. Tile 0: load B from &B[0] with stride 32. Compute and store to &C[0] with stride 32.
    // 3. Tile 1: load B from &B[16] with stride 32. Compute and store to &C[16] with stride 32.
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: Col-Major B GEMM ---
    {
        std::vector<half> h_A(16 * 16), h_B(16 * 16);
        for (int i = 0; i < 16 * 16; ++i) {
            h_A[i] = __float2half(1.0f);
            h_B[i] = __float2half(1.0f);
        }

        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_gemm_col_b_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(16 * 16);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - 16.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] != 0.0f) {
            std::cout << "Test 1 Passed: Single-warp WMMA GEMM with Col-Major B." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Single-warp WMMA GEMM with Col-Major B." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 2: Row-Major B GEMM ---
    {
        std::vector<half> h_A(16 * 16), h_B(16 * 16);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                h_A[r * 16 + c] = __float2half(static_cast<float>(r + 1));
                h_B[r * 16 + c] = __float2half(1.0f);
            }
        }

        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_gemm_row_b_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(16 * 16);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < 16; ++r) {
            float exp = static_cast<float>(r + 1) * 16.0f;
            for (int c = 0; c < 16; ++c) {
                if (std::abs(h_C[r * 16 + c] - exp) > 0.1f) { ok = false; break; }
            }
        }
        if (ok && h_C[0] != 0.0f) {
            std::cout << "Test 2 Passed: Single-warp WMMA GEMM with Row-Major B." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Single-warp WMMA GEMM with Row-Major B." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 3: Pre-loaded Accumulator FMA ---
    {
        std::vector<half> h_A(16 * 16, __float2half(1.0f));
        std::vector<half> h_B(16 * 16, __float2half(1.0f));
        std::vector<float> h_C(16 * 16, 5.0f);

        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), 16 * 16 * sizeof(float), cudaMemcpyHostToDevice));

        wmma_fma_accum_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - 21.0f) > 0.1f) { ok = false; break; } // 16 + 5 = 21
        }
        if (ok && h_C[0] == 21.0f) {
            std::cout << "Test 3 Passed: WMMA FMA with pre-loaded accumulator." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: WMMA FMA with pre-loaded accumulator (got " << h_C[0] << ", exp 21)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 4: In-Fragment Element Scaling ---
    {
        std::vector<half> h_A(16 * 16, __float2half(2.0f));
        std::vector<half> h_B(16 * 16, __float2half(1.0f));
        float alpha = 0.5f;

        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 16 * sizeof(float)));

        wmma_scaled_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C, alpha);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(16 * 16);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 2.0 * 1.0 * 16 = 32, * 0.5 = 16.0
        for (int i = 0; i < 16 * 16; ++i) {
            if (std::abs(h_C[i] - 16.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 16.0f) {
            std::cout << "Test 4 Passed: WMMA in-fragment scaling." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: WMMA in-fragment scaling." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --- Test 5: 16x32 Multi-Tile GEMM ---
    {
        std::vector<half> h_A(16 * 16, __float2half(1.0f));
        std::vector<half> h_B(16 * 32, __float2half(2.0f));
        std::vector<float> h_C(16 * 32, 0.0f);

        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, 16 * 32 * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, 16 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 16 * 32 * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, 16 * 32 * sizeof(float)));

        wmma_16x32_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, 16 * 32 * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // 1.0 * 2.0 * 16 = 32.0f
        for (int i = 0; i < 16 * 32; ++i) {
            if (std::abs(h_C[i] - 32.0f) > 0.1f) { ok = false; break; }
        }
        if (ok && h_C[0] == 32.0f) {
            std::cout << "Test 5 Passed: 16x32 multi-tile WMMA GEMM." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: 16x32 multi-tile WMMA GEMM (got " << h_C[0] << ", exp 32)." << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
