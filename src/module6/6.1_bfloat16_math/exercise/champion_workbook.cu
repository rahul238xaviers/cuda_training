// ============================================================================
// Module 6.1: BFloat16 Math - Champion Workbook
//
// Concepts Covered:
//   - 128-bit vectorized loads/stores (float4 holding 8x BF16 values)
//   - Fused BF16 SwiGLU activation (FP32 compute for numerical stability)
//   - Block-wide mixed-precision L2 normalization
//   - BF16 gradient clipping kernel
//   - Fused residual add + bias addition with broadcasting
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(1); \
        } \
    } while (0)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ============================================================================
// Exercise 1: 128-Bit Vectorized BF16 Memory Copy / Scale
// Each thread loads a float4 (16 bytes = 8 x __nv_bfloat16 = 4 x __nv_bfloat162),
// scales each BF16 value by scale, and writes back via a float4 store.
// num_vec = N / 8.
// ============================================================================
__global__ void vectorized_128bit_bf16_scale_kernel(const float4* in_vec,
                                                    float4* out_vec,
                                                    float scale,
                                                    int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // TODO:
        // 1. Load float4 raw = in_vec[idx];
        // 2. Reinterpret as __nv_bfloat162 pairs (4 pairs in 16 bytes).
        // 3. For each pair, convert to float2, scale by scale, convert back.
        // 4. Store scaled float4 into out_vec[idx].
    }
}

// ============================================================================
// Exercise 2: Fused BF16 SwiGLU Forward Kernel
// SwiGLU(x1, x2) = x1 * SiLU(x2) = x1 * (x2 / (1 + exp(-x2)))
// Load x1 and x2 (size N) in BF16, compute SiLU(x2) * x1 in FP32,
// and write back result to out in BF16.
// ============================================================================
__global__ void bf16_swiglu_kernel(const __nv_bfloat16* x1,
                                   const __nv_bfloat16* x2,
                                   __nv_bfloat16* out,
                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // 1. Load v1 = x1[idx], v2 = x2[idx] as float.
        // 2. Compute silu_v2 = v2 / (1.0f + expf(-v2)).
        // 3. Compute res = v1 * silu_v2.
        // 4. Convert res to __nv_bfloat16 and write to out[idx].
    }
}

// ============================================================================
// Exercise 3: Block-Wide Mixed-Precision L2 Normalization
// Given a vector x of length N (N <= 1024, 1 block):
// 1. Compute sum of squares: sum_sq = sum(x[i]^2) using FP32 shared memory reduction.
// 2. Compute norm = sqrtf(sum_sq + 1e-6f).
// 3. Normalize: out[i] = x[i] / norm in BF16.
// ============================================================================
__global__ void bf16_l2_norm_kernel(const __nv_bfloat16* x,
                                    __nv_bfloat16* out,
                                    int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // TODO:
    // 1. Load x[tid] (if tid < N), convert to float, square it. Store in sdata[tid].
    // 2. __syncthreads() and perform shared memory tree reduction down to sdata[0].
    // 3. Compute norm = sqrtf(sdata[0] + 1e-6f) and store in sdata[0].
    // 4. __syncthreads()
    // 5. Each thread divides x[tid] by sdata[0] and writes to out[tid].
}

// ============================================================================
// Exercise 4: BF16 Gradient Threshold Clipping
// Given gradients g (length N), and max_norm threshold:
// If abs(g[i]) > max_norm, clamp to sign(g[i]) * max_norm.
// ============================================================================
__global__ void bf16_grad_clip_kernel(__nv_bfloat16* g, float max_norm, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // 1. Load g[idx] as float.
        // 2. If val > max_norm, val = max_norm; if val < -max_norm, val = -max_norm.
        // 3. Store clamped value back to g[idx] as __nv_bfloat16.
    }
}

// ============================================================================
// Exercise 5: Fused BF16 Residual Add + Bias Broadcast
// out[i] = x[i] + residual[i] + bias[i % D]
// x, residual, out have length N. bias has length D.
// ============================================================================
__global__ void bf16_fused_residual_bias_kernel(const __nv_bfloat16* x,
                                                const __nv_bfloat16* residual,
                                                const __nv_bfloat16* bias,
                                                __nv_bfloat16* out,
                                                int N,
                                                int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // 1. Load x[idx], residual[idx], and bias[idx % D] in float.
        // 2. Compute float sum = x_f + res_f + bias_f.
        // 3. Convert to __nv_bfloat16 and write to out[idx].
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int N = 1024;
    const int blockSize = 256;

    // --- Test 1: 128-Bit Vectorized Scale ---
    {
        int num_vec = N / 8;
        std::vector<__nv_bfloat16> h_in(N), h_out(N);
        float scale = 4.0f;
        for (int i = 0; i < N; ++i) h_in[i] = __float2bfloat16(static_cast<float>(i + 1) * 0.25f);

        __nv_bfloat16 *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (num_vec + blockSize - 1) / blockSize;
        vectorized_128bit_bf16_scale_kernel<<<blocks, blockSize>>>(
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            scale,
            num_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = static_cast<float>(i + 1) * 0.25f * scale;
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp - act) > 0.05f * exp) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 1 Passed: 128-bit vectorized float4 BF16 scaling." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: 128-bit vectorized float4 BF16 scaling." << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --- Test 2: Fused BF16 SwiGLU Forward ---
    {
        std::vector<__nv_bfloat16> h_x1(N), h_x2(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_x1[i] = __float2bfloat16(static_cast<float>(i + 1) * 0.01f);
            h_x2[i] = __float2bfloat16(static_cast<float>(i - N / 2) * 0.02f);
        }

        __nv_bfloat16 *d_x1, *d_x2, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x1, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_x2, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_x1, h_x1.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x2, h_x2.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (N + blockSize - 1) / blockSize;
        bf16_swiglu_kernel<<<blocks, blockSize>>>(d_x1, d_x2, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float v1 = __bfloat162float(h_x1[i]);
            float v2 = __bfloat162float(h_x2[i]);
            float silu_v2 = v2 / (1.0f + std::exp(-v2));
            float exp_val = v1 * silu_v2;
            float act_val = __bfloat162float(h_out[i]);
            if (std::abs(exp_val - act_val) > 0.05f * (std::abs(exp_val) + 1e-4f)) {
                ok = false;
                break;
            }
        }
        if (ok && __bfloat162float(h_out[N - 1]) != 0.0f) {
            std::cout << "Test 2 Passed: Fused BF16 SwiGLU activation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Fused BF16 SwiGLU activation." << std::endl;
        }

        cudaFree(d_x1); cudaFree(d_x2); cudaFree(d_out);
    }

    // --- Test 3: Block-Wide Mixed-Precision L2 Norm ---
    {
        const int L = 512;
        std::vector<__nv_bfloat16> h_x(L), h_out(L);
        float sum_sq = 0.0f;
        for (int i = 0; i < L; ++i) {
            float val = static_cast<float>(i + 1) * 0.1f;
            h_x[i] = __float2bfloat16(val);
            sum_sq += val * val;
        }
        float expected_norm = std::sqrt(sum_sq + 1e-6f);

        __nv_bfloat16 *d_x, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, L * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, L * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), L * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, L * sizeof(__nv_bfloat16)));

        bf16_l2_norm_kernel<<<1, L, L * sizeof(float)>>>(d_x, d_out, L);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, L * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < L; ++i) {
            float exp_val = __bfloat162float(h_x[i]) / expected_norm;
            float act_val = __bfloat162float(h_out[i]);
            if (std::abs(exp_val - act_val) > 0.05f * exp_val) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 3 Passed: Block-wide mixed-precision L2 normalization." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Block-wide mixed-precision L2 normalization." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_out);
    }

    // --- Test 4: BF16 Gradient Clipping ---
    {
        std::vector<__nv_bfloat16> h_g(N);
        float max_norm = 1.5f;
        for (int i = 0; i < N; ++i) {
            float val = static_cast<float>(i - N / 2) * 0.01f;
            h_g[i] = __float2bfloat16(val);
        }

        __nv_bfloat16* d_g;
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        int blocks = (N + blockSize - 1) / blockSize;
        bf16_grad_clip_kernel<<<blocks, blockSize>>>(d_g, max_norm, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> h_out(N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_g, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float orig = __bfloat162float(h_g[i]);
            float exp = orig;
            if (exp > max_norm) exp = max_norm;
            if (exp < -max_norm) exp = -max_norm;
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp - act) > 0.05f) { ok = false; break; }
        }
        // Check that clamping actually happened
        if (ok && __bfloat162float(h_out[N - 1]) == max_norm) {
            std::cout << "Test 4 Passed: BF16 gradient threshold clipping." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: BF16 gradient threshold clipping." << std::endl;
        }

        cudaFree(d_g);
    }

    // --- Test 5: Fused BF16 Residual Add + Bias ---
    {
        const int D = 64;
        std::vector<__nv_bfloat16> h_x(N), h_res(N), h_bias(D), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_x[i] = __float2bfloat16(1.0f);
            h_res[i] = __float2bfloat16(2.0f);
        }
        for (int d = 0; d < D; ++d) {
            h_bias[d] = __float2bfloat16(static_cast<float>(d) * 0.1f);
        }

        __nv_bfloat16 *d_x, *d_res, *d_bias, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_res, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_bias, D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (N + blockSize - 1) / blockSize;
        bf16_fused_residual_bias_kernel<<<blocks, blockSize>>>(d_x, d_res, d_bias, d_out, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = 1.0f + 2.0f + static_cast<float>(i % D) * 0.1f;
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp - act) > 0.05f) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 5 Passed: Fused BF16 residual add with bias broadcasting." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Fused BF16 residual add with bias broadcasting." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_bias); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
