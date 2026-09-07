// ============================================================================
// Module 7.1: Elementwise & Reshaping - Champion Workbook
// Kernels Covered:
//   - swiglu_backward: Full dual-gradient kernel matching swiglu_backward.metal
//   - swiglu_forward_vec4: 128-bit vectorized SwiGLU forward pass
//   - tiled_reshape_4d: Shared-memory tiled 4D transpose for coalesced memory access
//   - fused_dual_residual_add: h += attn_out + ffn_out
//   - swiglu_backward_accum: Backprop with in-place gradient accumulation (+=)
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
// Exercise 1: Full SwiGLU Backward Kernel
// Matching swiglu_backward.metal
// Computes both grad_gate and grad_up simultaneously.
// ============================================================================
__global__ void swiglu_backward_kernel(const __nv_bfloat16* grad_output,
                                       const __nv_bfloat16* gate,
                                       const __nv_bfloat16* up,
                                       __nv_bfloat16* grad_gate,
                                       __nv_bfloat16* grad_up,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // float g = __bfloat162float(gate[idx]);
        // float u = __bfloat162float(up[idx]);
        // float dy = __bfloat162float(grad_output[idx]);
        // float sig = 1.0f / (1.0f + expf(-g));
        // float silu_val = g * sig;
        // float dsilu = sig * (1.0f + g * (1.0f - sig));
        // grad_up[idx] = __float2bfloat16(dy * silu_val);
        // grad_gate[idx] = __float2bfloat16(dy * u * dsilu);
    }
}

// ============================================================================
// Exercise 2: 128-Bit Vectorized SwiGLU Forward (float4 loads)
// num_vec = n / 4. Process 4 float elements per thread.
// ============================================================================
__global__ void swiglu_forward_vec4_kernel(const float4* gate_vec,
                                           const float4* up_vec,
                                           float4* out_vec, int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // TODO:
        // 1. Load float4 g4 = gate_vec[idx], u4 = up_vec[idx];
        // 2. Compute SwiGLU for .x, .y, .z, .w:
        //    res.x = (g4.x / (1.0f + expf(-g4.x))) * u4.x; ...
        // 3. Store to out_vec[idx].
    }
}

// ============================================================================
// Exercise 3: Tiled 4D Transpose with Shared Memory
// Transposing sequence and heads dimensions: [B, S, H, D] -> [B, H, S, D]
// Uses a 16x16 shared memory tile to ensure coalesced global memory reads and writes.
// S and H are multiples of 16.
// ============================================================================
__global__ void tiled_reshape_4d_kernel(const float* src, float* dst,
                                        int B, int S, int H, int D) {
    // TODO:
    // Compute tile indices and transpose coordinates using shared memory staging.
}

// ============================================================================
// Exercise 4: Fused Dual Residual Add: h[i] += attn[i] + ffn[i]
// Vectorized 128-bit operation.
// ============================================================================
__global__ void fused_dual_residual_add_kernel(float4* h, const float4* attn,
                                               const float4* ffn, int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // TODO:
        // float4 vh = h[idx];
        // float4 va = attn[idx];
        // float4 vf = ffn[idx];
        // vh.x += va.x + vf.x; ...
        // h[idx] = vh;
    }
}

// ============================================================================
// Exercise 5: SwiGLU Backward with Gradient Accumulation (+=)
// In training, gradients from multiple heads or micro-batches are accumulated.
// ============================================================================
__global__ void swiglu_backward_accum_kernel(const float* grad_output,
                                             const float* gate,
                                             const float* up,
                                             float* grad_gate_accum,
                                             float* grad_up_accum,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO:
        // Compute dL/du and dL/dg as in Exercise 1, then:
        // grad_up_accum[idx] += dy * silu_val;
        // grad_gate_accum[idx] += dy * u * dsilu;
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int blockSize = 256;

    // --- Test 1: Full SwiGLU Backward ---
    {
        const int N = 512;
        std::vector<__nv_bfloat16> h_dy(N), h_g(N), h_u(N), h_dg(N), h_du(N);
        for (int i = 0; i < N; ++i) {
            h_dy[i] = __float2bfloat16(1.5f);
            h_g[i] = __float2bfloat16(static_cast<float>(i - 256) * 0.01f);
            h_u[i] = __float2bfloat16(2.0f);
        }

        __nv_bfloat16 *d_dy, *d_g, *d_u, *d_dg, *d_du;
        CUDA_CHECK(cudaMalloc(&d_dy, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_dg, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_du, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dg, 0, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_du, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_backward_kernel<<<blocks, blockSize>>>(d_dy, d_g, d_u, d_dg, d_du, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dg.data(), d_dg, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_du.data(), d_du, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = __bfloat162float(h_g[i]);
            float u = __bfloat162float(h_u[i]);
            float dy = __bfloat162float(h_dy[i]);
            float sig = 1.0f / (1.0f + std::exp(-g));
            float silu_val = g * sig;
            float dsilu = sig * (1.0f + g * (1.0f - sig));
            float exp_du = dy * silu_val;
            float exp_dg = dy * u * dsilu;
            if (std::abs(__bfloat162float(h_du[i]) - exp_du) > 0.05f * (std::abs(exp_du) + 1e-4f)) { ok = false; break; }
            if (std::abs(__bfloat162float(h_dg[i]) - exp_dg) > 0.05f * (std::abs(exp_dg) + 1e-4f)) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_du[N - 1]) != 0.0f) {
            std::cout << "Test 1 Passed: Full SwiGLU backward pass." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Full SwiGLU backward pass." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_g); cudaFree(d_u); cudaFree(d_dg); cudaFree(d_du);
    }

    // --- Test 2: 128-bit Vectorized SwiGLU Forward ---
    {
        const int N = 1024;
        const int num_vec = N / 4;
        std::vector<float> h_g(N), h_u(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_g[i] = static_cast<float>(i - 512) * 0.01f;
            h_u[i] = 1.0f;
        }

        float *d_g, *d_u, *d_out;
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        int blocks = (num_vec + blockSize - 1) / blockSize;
        swiglu_forward_vec4_kernel<<<blocks, blockSize>>>(reinterpret_cast<const float4*>(d_g),
                                                          reinterpret_cast<const float4*>(d_u),
                                                          reinterpret_cast<float4*>(d_out),
                                                          num_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp_val = (h_g[i] / (1.0f + std::exp(-h_g[i]))) * h_u[i];
            if (std::abs(h_out[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[N - 1] != 0.0f) {
            std::cout << "Test 2 Passed: 128-bit vectorized SwiGLU forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: 128-bit vectorized SwiGLU forward." << std::endl;
        }

        cudaFree(d_g); cudaFree(d_u); cudaFree(d_out);
    }

    // --- Test 3: Tiled 4D Transpose ---
    {
        const int B = 1, S = 16, H = 16, D = 16;
        const int total_elem = B * S * H * D;
        std::vector<float> h_src(total_elem), h_dst(total_elem);
        for (int i = 0; i < total_elem; ++i) h_src[i] = static_cast<float>(i + 1);

        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dst, 0, total_elem * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((H + 15) / 16, (S + 15) / 16, B);
        tiled_reshape_4d_kernel<<<grid, block>>>(d_src, d_dst, B, S, H, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < D; ++d) {
                        int src_idx = (b * S * H + s * H + h) * D + d;
                        int dst_idx = (b * H * S + h * S + s) * D + d;
                        if (h_dst[dst_idx] != h_src[src_idx]) { ok = false; break; }
                    }
                }
            }
        }
        if (ok && h_dst[0] != 0.0f) {
            std::cout << "Test 3 Passed: Tiled 4D reshape." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Tiled 4D reshape." << std::endl;
        }

        cudaFree(d_src); cudaFree(d_dst);
    }

    // --- Test 4: Fused Dual Residual Add ---
    {
        const int N = 1024;
        const int num_vec = N / 4;
        std::vector<float> h_h(N, 1.0f), h_attn(N, 2.0f), h_ffn(N, 3.0f);

        float *d_h, *d_attn, *d_ffn;
        CUDA_CHECK(cudaMalloc(&d_h, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ffn, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_h, h_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn, h_attn.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ffn, h_ffn.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (num_vec + blockSize - 1) / blockSize;
        fused_dual_residual_add_kernel<<<blocks, blockSize>>>(reinterpret_cast<float4*>(d_h),
                                                              reinterpret_cast<const float4*>(d_attn),
                                                              reinterpret_cast<const float4*>(d_ffn),
                                                              num_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_h.data(), d_h, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_h[i] - 6.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_h[0] == 6.0f) {
            std::cout << "Test 4 Passed: Fused dual residual addition." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused dual residual addition." << std::endl;
        }

        cudaFree(d_h); cudaFree(d_attn); cudaFree(d_ffn);
    }

    // --- Test 5: SwiGLU Backward with Accumulation ---
    {
        const int N = 512;
        std::vector<float> h_dy(N, 1.0f), h_g(N, 1.0f), h_u(N, 2.0f);
        std::vector<float> h_dg_accum(N, 10.0f), h_du_accum(N, 20.0f);

        float *d_dy, *d_g, *d_u, *d_dg, *d_du;
        CUDA_CHECK(cudaMalloc(&d_dy, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dg, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_du, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dg, h_dg_accum.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_du, h_du_accum.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_backward_accum_kernel<<<blocks, blockSize>>>(d_dy, d_g, d_u, d_dg, d_du, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dg_accum.data(), d_dg, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_du_accum.data(), d_du, N * sizeof(float), cudaMemcpyDeviceToHost));

        float g = 1.0f, u = 2.0f, dy = 1.0f;
        float sig = 1.0f / (1.0f + std::exp(-g));
        float silu_val = g * sig;
        float dsilu = sig * (1.0f + g * (1.0f - sig));
        float exp_du = 20.0f + dy * silu_val;
        float exp_dg = 10.0f + dy * u * dsilu;

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_du_accum[i] - exp_du) > 1e-4f) { ok = false; break; }
            if (std::abs(h_dg_accum[i] - exp_dg) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_du_accum[0] != 20.0f) {
            std::cout << "Test 5 Passed: SwiGLU backward with accumulation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: SwiGLU backward with accumulation." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_g); cudaFree(d_u); cudaFree(d_dg); cudaFree(d_du);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
