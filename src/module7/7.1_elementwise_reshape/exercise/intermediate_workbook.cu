// ============================================================================
// Module 7.1: Elementwise & Reshaping - Intermediate Workbook
// Kernels Covered:
//   - swiglu_backward_gate: Full gradient w.r.t gate projection
//   - residual_add_vec4: 128-bit float4 vectorized residual addition
//   - reshape_to_4d: Attention layout transpose [B, S, H, D] -> [B, H, S, D]
//   - swiglu_forward_bf16: Native __nv_bfloat16 SwiGLU activation
//   - fused_residual_swiglu: In-register residual add followed by SwiGLU
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

// ============================================================================
// Exercise 1: SwiGLU Backward Gate Gradient
// sig = 1 / (1 + exp(-g))
// dsilu = sig * (1 + g * (1 - sig))
// grad_gate[idx] = grad_out[idx] * up[idx] * dsilu
// ============================================================================
__global__ void swiglu_backward_gate_kernel(const float* grad_out, const float* gate,
                                            const float* up, float* grad_gate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // float g = gate[idx];
        // float u = up[idx];
        // float dy = grad_out[idx];
        // float sig = 1.0f / (1.0f + expf(-g));
        // float dsilu = sig * (1.0f + g * (1.0f - sig));
        // grad_gate[idx] = dy * u * dsilu;
    }
}

// ============================================================================
// Exercise 2: Vectorized 128-bit Residual Addition (float4)
// num_vec = n / 4. Each thread loads and adds 4 float elements at once.
// ============================================================================
__global__ void residual_add_vec4_kernel(float4* a, const float4* b, int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // TODO:
        // float4 va = a[idx];
        // float4 vb = b[idx];
        // va.x += vb.x; va.y += vb.y; va.z += vb.z; va.w += vb.w;
        // a[idx] = va;
    }
}

// ============================================================================
// Exercise 3: 4D Reshape [B, S, H, D] -> [B, H, S, D]
// Matching reshape_4d.metal
// ============================================================================
__global__ void reshape_to_4d_kernel(const float* src, float* dst,
                                     int batch, int n_heads, int seq_len, int head_dim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_heads * seq_len * head_dim;
    if (gid < total) {
        // TODO:
        // int tmp = gid;
        // int d = tmp % head_dim; tmp /= head_dim;
        // int s = tmp % seq_len;  tmp /= seq_len;
        // int h = tmp % n_heads;  tmp /= n_heads;
        // int b = tmp;
        // int src_idx = (b * seq_len * n_heads + s * n_heads + h) * head_dim + d;
        // dst[gid] = src[src_idx];
    }
}

// ============================================================================
// Exercise 4: SwiGLU Forward in BF16 (__nv_bfloat16)
// out[i] = (bfloat)((g / (1.0f + exp(-g))) * u)
// ============================================================================
__global__ void swiglu_forward_bf16_kernel(const __nv_bfloat16* gate,
                                           const __nv_bfloat16* up,
                                           __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // float g = __bfloat162float(gate[idx]);
        // float u = __bfloat162float(up[idx]);
        // float silu_g = g / (1.0f + expf(-g));
        // out[idx] = __float2bfloat16(silu_g * u);
    }
}

// ============================================================================
// Exercise 5: Fused In-Place Residual Add + SwiGLU Forward
// u[idx] += residual[idx] (in-place)
// out[idx] = SiLU(gate[idx]) * u[idx]
// ============================================================================
__global__ void fused_residual_swiglu_kernel(const float* gate, float* u,
                                             const float* residual,
                                             float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // float updated_u = u[idx] + residual[idx];
        // u[idx] = updated_u;
        // float g = gate[idx];
        // float silu_g = g / (1.0f + expf(-g));
        // out[idx] = silu_g * updated_u;
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int blockSize = 256;

    // --- Test 1: swiglu_backward_gate ---
    {
        const int N = 512;
        std::vector<float> h_dy(N), h_g(N), h_u(N), h_dg(N);
        for (int i = 0; i < N; ++i) {
            h_dy[i] = 1.0f;
            h_g[i] = static_cast<float>(i - 256) * 0.01f;
            h_u[i] = 2.0f;
        }

        float *d_dy, *d_g, *d_u, *d_dg;
        CUDA_CHECK(cudaMalloc(&d_dy, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dg, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dg, 0, N * sizeof(float)));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_backward_gate_kernel<<<blocks, blockSize>>>(d_dy, d_g, d_u, d_dg, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dg.data(), d_dg, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = h_g[i];
            float u = h_u[i];
            float dy = h_dy[i];
            float sig = 1.0f / (1.0f + std::exp(-g));
            float dsilu = sig * (1.0f + g * (1.0f - sig));
            float exp_val = dy * u * dsilu;
            if (std::abs(h_dg[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_dg[N - 1] != 0.0f) {
            std::cout << "Test 1 Passed: swiglu_backward_gate." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: swiglu_backward_gate." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_g); cudaFree(d_u); cudaFree(d_dg);
    }

    // --- Test 2: residual_add_vec4 ---
    {
        const int N = 1024;
        const int num_vec = N / 4;
        std::vector<float> h_a(N, 1.0f), h_b(N, 3.0f);

        float *d_a, *d_b;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (num_vec + blockSize - 1) / blockSize;
        residual_add_vec4_kernel<<<blocks, blockSize>>>(reinterpret_cast<float4*>(d_a),
                                                        reinterpret_cast<const float4*>(d_b),
                                                        num_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_a[i] - 4.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_a[0] == 4.0f) {
            std::cout << "Test 2 Passed: residual_add_vec4 (128-bit loads)." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: residual_add_vec4." << std::endl;
        }

        cudaFree(d_a); cudaFree(d_b);
    }

    // --- Test 3: reshape_to_4d ---
    {
        const int B = 2, H = 2, S = 4, D = 4;
        const int total_elem = B * H * S * D;
        std::vector<float> h_src(total_elem), h_dst(total_elem);
        for (int i = 0; i < total_elem; ++i) h_src[i] = static_cast<float>(i + 1);

        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dst, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        reshape_to_4d_kernel<<<blocks, blockSize>>>(d_src, d_dst, B, H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int gid = 0; gid < total_elem; ++gid) {
            int tmp = gid;
            int d = tmp % D; tmp /= D;
            int s = tmp % S; tmp /= S;
            int h = tmp % H; tmp /= H;
            int b = tmp;
            int src_idx = (b * S * H + s * H + h) * D + d;
            if (h_dst[gid] != h_src[src_idx]) { ok = false; break; }
        }
        if (ok && h_dst[0] != 0.0f) {
            std::cout << "Test 3 Passed: reshape_to_4d." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: reshape_to_4d." << std::endl;
        }

        cudaFree(d_src); cudaFree(d_dst);
    }

    // --- Test 4: swiglu_forward_bf16 ---
    {
        const int N = 512;
        std::vector<__nv_bfloat16> h_gate(N), h_up(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_gate[i] = __float2bfloat16(static_cast<float>(i - 256) * 0.02f);
            h_up[i] = __float2bfloat16(2.0f);
        }

        __nv_bfloat16 *d_gate, *d_up, *d_out;
        CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_up, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_forward_bf16_kernel<<<blocks, blockSize>>>(d_gate, d_up, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = __bfloat162float(h_gate[i]);
            float u = __bfloat162float(h_up[i]);
            float exp_val = (g / (1.0f + std::exp(-g))) * u;
            float act_val = __bfloat162float(h_out[i]);
            if (std::abs(act_val - exp_val) > 0.05f * (std::abs(exp_val) + 1e-4f)) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[N - 1]) != 0.0f) {
            std::cout << "Test 4 Passed: swiglu_forward_bf16." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: swiglu_forward_bf16." << std::endl;
        }

        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    }

    // --- Test 5: fused_residual_swiglu ---
    {
        const int N = 512;
        std::vector<float> h_gate(N, 1.0f), h_u(N, 2.0f), h_res(N, 3.0f), h_out(N);

        float *d_gate, *d_u, *d_res, *d_out;
        CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        int blocks = (N + blockSize - 1) / blockSize;
        fused_residual_swiglu_kernel<<<blocks, blockSize>>>(d_gate, d_u, d_res, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // updated_u = 2 + 3 = 5.0
        // silu(1.0) = 1.0 / (1 + exp(-1.0)) = 0.7310585786
        // out = 5.0 * 0.7310585786 = 3.65529
        float exp_out = 5.0f / (1.0f + std::exp(-1.0f));
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_u[i] - 5.0f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_out[i] - exp_out) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 5 Passed: fused_residual_swiglu." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: fused_residual_swiglu." << std::endl;
        }

        cudaFree(d_gate); cudaFree(d_u); cudaFree(d_res); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
