// ============================================================================
// Module 7.3: Normalization & RoPE - Beginner Workbook
// Kernels Covered:
//   - rms_norm_single_row: Single-warp intra-row RMSNorm forward
//   - rope_forward_pairs: 2D feature rotation on (x_2i, x_2i+1)
//   - rope_backward_pairs: Transposed 2D feature rotation
//   - fused_add_and_sq: Fused residual add + square accumulation
//   - rms_norm_unweighted: Unscaled RMSNorm normalization
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(1); \
        } \
    } while (0)

#define FULL_MASK 0xffffffff

// ============================================================================
// Exercise 1: Single-Warp RMSNorm Forward (D = 32)
// For a single row x of length D (1 warp = 32 threads):
// 1. Thread lane loads x[lane].
// 2. Compute square: sq = x[lane] * x[lane].
// 3. Warp shuffle down to sum all 32 squares.
// 4. Broadcast sum to all threads using __shfl_sync.
// 5. rms = rsqrtf(sum / 32.0f + eps).
// 6. out[lane] = x[lane] * rms * weight[lane].
// ============================================================================
__global__ void rms_norm_single_row_kernel(const float* x, const float* weight,
                                           float* out, int D, float eps) {
    int lane = threadIdx.x;
    if (lane < D) {
        // TODO:
        // float val = x[lane];
        // float sq = val * val;
        // for (int offset = 16; offset > 0; offset /= 2) {
        //     sq += __shfl_down_sync(FULL_MASK, sq, offset);
        // }
        // float sum_sq = __shfl_sync(FULL_MASK, sq, 0);
        // float rms = rsqrtf(sum_sq / (float)D + eps);
        // out[lane] = val * rms * weight[lane];
    }
}

// ============================================================================
// Exercise 2: RoPE Forward Rotation on Pairs
// Given 1D array of length D (even). Each thread i handles pair (2i, 2i+1).
// x0' = x0 * cos - x1 * sin
// x1' = x0 * sin + x1 * cos
// num_pairs = D / 2.
// ============================================================================
__global__ void rope_forward_pairs_kernel(float* x, const float* cos_table,
                                          const float* sin_table, int num_pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        // TODO:
        // int idx0 = 2 * i;
        // int idx1 = 2 * i + 1;
        // float c = cos_table[i];
        // float s = sin_table[i];
        // float x0 = x[idx0];
        // float x1 = x[idx1];
        // x[idx0] = x0 * c - x1 * s;
        // x[idx1] = x0 * s + x1 * c;
    }
}

// ============================================================================
// Exercise 3: RoPE Backward Transposed Rotation
// Inverse of rotation matrix:
// x0' = x0 * cos + x1 * sin
// x1' = x1 * cos - x0 * sin
// ============================================================================
__global__ void rope_backward_pairs_kernel(float* grad, const float* cos_table,
                                           const float* sin_table, int num_pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pairs) {
        // TODO:
        // int idx0 = 2 * i;
        // int idx1 = 2 * i + 1;
        // float c = cos_table[i];
        // float s = sin_table[i];
        // float x0 = grad[idx0];
        // float x1 = grad[idx1];
        // grad[idx0] = x0 * c + x1 * s;
        // grad[idx1] = x1 * c - x0 * s;
    }
}

// ============================================================================
// Exercise 4: Fused In-Place Residual Add + Square Accumulation
// x[i] += residual[i]
// out_sq[i] = x[i] * x[i]
// ============================================================================
__global__ void fused_add_and_sq_kernel(float* x, const float* residual,
                                        float* out_sq, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // float nv = x[idx] + residual[idx];
        // x[idx] = nv;
        // out_sq[idx] = nv * nv;
    }
}

// ============================================================================
// Exercise 5: Unweighted RMSNorm Normalization
// out[lane] = x[lane] * rsqrtf(mean(x^2) + eps)
// ============================================================================
__global__ void rms_norm_unweighted_kernel(const float* x, float* out,
                                           int D, float eps) {
    int lane = threadIdx.x;
    if (lane < D) {
        // TODO:
        // float val = x[lane];
        // float sq = val * val;
        // for (int offset = 16; offset > 0; offset /= 2) {
        //     sq += __shfl_down_sync(FULL_MASK, sq, offset);
        // }
        // float sum_sq = __shfl_sync(FULL_MASK, sq, 0);
        // float rms = rsqrtf(sum_sq / (float)D + eps);
        // out[lane] = val * rms;
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: Single-Warp RMSNorm Forward ---
    {
        const int D = 32;
        float eps = 1e-5f;
        std::vector<float> h_x(D), h_w(D, 1.0f), h_out(D);
        float sum_sq = 0.0f;
        for (int i = 0; i < D; ++i) {
            h_x[i] = static_cast<float>(i + 1);
            sum_sq += h_x[i] * h_x[i];
        }
        float expected_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);

        float *d_x, *d_w, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, D * sizeof(float)));

        rms_norm_single_row_kernel<<<1, 32>>>(d_x, d_w, d_out, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < D; ++i) {
            float exp = h_x[i] * expected_rms * h_w[i];
            if (std::abs(h_out[i] - exp) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 1 Passed: Single-warp RMSNorm forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Single-warp RMSNorm forward." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    }

    // --- Test 2: RoPE Forward Pairs ---
    {
        const int num_pairs = 16;
        const int D = 32;
        std::vector<float> h_x(D), h_cos(num_pairs), h_sin(num_pairs);
        for (int i = 0; i < num_pairs; ++i) {
            h_x[2 * i] = 1.0f;
            h_x[2 * i + 1] = 0.0f;
            float angle = static_cast<float>(i) * 0.1f;
            h_cos[i] = std::cos(angle);
            h_sin[i] = std::sin(angle);
        }

        float *d_x, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_x, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, num_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, num_pairs * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), num_pairs * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), num_pairs * sizeof(float), cudaMemcpyHostToDevice));

        rope_forward_pairs_kernel<<<1, 32>>>(d_x, d_cos, d_sin, num_pairs);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(D);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < num_pairs; ++i) {
            // (1, 0) rotated by angle gives (cos, sin)
            if (std::abs(h_out[2 * i] - h_cos[i]) > 1e-4f) { ok = false; break; }
            if (std::abs(h_out[2 * i + 1] - h_sin[i]) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 2 Passed: RoPE forward 2D rotation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: RoPE forward 2D rotation." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin);
    }

    // --- Test 3: RoPE Backward Transposed Rotation ---
    {
        const int num_pairs = 16;
        const int D = 32;
        std::vector<float> h_grad(D), h_cos(num_pairs), h_sin(num_pairs);
        for (int i = 0; i < num_pairs; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            h_cos[i] = std::cos(angle);
            h_sin[i] = std::sin(angle);
            // Rotated vector (cos, sin)
            h_grad[2 * i] = h_cos[i];
            h_grad[2 * i + 1] = h_sin[i];
        }

        float *d_grad, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_grad, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, num_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, num_pairs * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_grad, h_grad.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), num_pairs * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), num_pairs * sizeof(float), cudaMemcpyHostToDevice));

        rope_backward_pairs_kernel<<<1, 32>>>(d_grad, d_cos, d_sin, num_pairs);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(D);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_grad, D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Rotating (cos, sin) backwards by angle should recover (1, 0)
        for (int i = 0; i < num_pairs; ++i) {
            if (std::abs(h_out[2 * i] - 1.0f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_out[2 * i + 1] - 0.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[0] == 1.0f) {
            std::cout << "Test 3 Passed: RoPE backward transposed rotation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: RoPE backward transposed rotation." << std::endl;
        }

        cudaFree(d_grad); cudaFree(d_cos); cudaFree(d_sin);
    }

    // --- Test 4: Fused Add and Sq ---
    {
        const int N = 256;
        std::vector<float> h_x(N, 2.0f), h_res(N, 3.0f), h_sq(N);

        float *d_x, *d_res, *d_sq;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sq, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sq, 0, N * sizeof(float)));

        fused_add_and_sq_kernel<<<1, 256>>>(d_x, d_res, d_sq, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sq.data(), d_sq, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_x[i] - 5.0f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_sq[i] - 25.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_x[0] == 5.0f) {
            std::cout << "Test 4 Passed: Fused residual add + square accumulation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused residual add + square accumulation." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_sq);
    }

    // --- Test 5: Unweighted RMSNorm ---
    {
        const int D = 32;
        float eps = 1e-5f;
        std::vector<float> h_x(D, 2.0f), h_out(D);
        // For array of all 2.0f, sum_sq = 32 * 4 = 128. mean = 4.0. rms = 1 / sqrt(4 + eps) ~ 0.5f.
        // out[i] = 2.0 * 0.5 = 1.0f!

        float *d_x, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, D * sizeof(float)));

        rms_norm_unweighted_kernel<<<1, 32>>>(d_x, d_out, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < D; ++i) {
            if (std::abs(h_out[i] - 1.0f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 5 Passed: Unweighted RMSNorm normalization." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Unweighted RMSNorm normalization." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
