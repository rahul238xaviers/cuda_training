// ============================================================================
// Module 7.3: Normalization & RoPE - Intermediate Workbook
// Kernels Covered:
//   - rms_norm_forward_block: Multi-warp block-level RMSNorm for large hidden dim
//   - rope_forward_mha: Multi-head RoPE forward across batch & seq
//   - fused_add_norm: Fused residual add + RMSNorm in registers
//   - rms_norm_backward_dx: Input gradient computation for RMSNorm
//   - rope_backward_mha: Multi-head RoPE backward pass
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

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// ============================================================================
// Exercise 1: Multi-Warp RMSNorm Forward (1 Block per Row)
// Matching rms_norm_forward.metal
// blockDim.x = 256 (8 warps). Each row has hidden dimension D (e.g. 512).
// ============================================================================
__global__ void rms_norm_forward_block_kernel(const float* input, float* output,
                                              const float* weight, int D, float eps) {
    __shared__ float s_warp_sums[8];
    __shared__ float s_rms;

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int base = row_idx * D;

    // TODO:
    // 1. Compute thread local sum of squares via grid-stride loop:
    //    float local_sum = 0.0f;
    //    for (int c = tid; c < D; c += blockDim.x) { float v = input[base + c]; local_sum += v * v; }
    // 2. Warp reduction: float w_sum = warp_reduce_sum(local_sum);
    // 3. If lane_id == 0, s_warp_sums[warp_id] = w_sum;
    // 4. __syncthreads();
    // 5. Warp 0 reduces s_warp_sums (8 values):
    //    if (warp_id == 0) {
    //        float v = (lane_id < 8) ? s_warp_sums[lane_id] : 0.0f;
    //        v = warp_reduce_sum(v);
    //        if (lane_id == 0) s_rms = rsqrtf(v / (float)D + eps);
    //    }
    // 6. __syncthreads();
    // 7. Normalize & scale output:
    //    for (int c = tid; c < D; c += blockDim.x) {
    //        output[base + c] = input[base + c] * s_rms * weight[c];
    //    }
}

// ============================================================================
// Exercise 2: Multi-Head RoPE Forward (Matching rope_forward.metal)
// Applies rotation to Q: shape [batch, heads, seq_len, head_dim]
// Each thread handles a feature pair: (2*i, 2*i + 1)
// ============================================================================
__global__ void rope_forward_mha_kernel(float* q, const float* cos_table,
                                        const float* sin_table,
                                        int batch, int heads, int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    int total_pairs = batch * heads * seq_len * half_dim;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < total_pairs) {
        // TODO:
        // int tmp = gid;
        // int i = tmp % half_dim; tmp /= half_dim;
        // int s = tmp % seq_len;  tmp /= seq_len;
        // int h = tmp % heads;
        // int b = tmp / heads;
        // int base = (b * heads * seq_len + h * seq_len + s) * head_dim;
        // float c = cos_table[s * half_dim + i];
        // float sn = sin_table[s * half_dim + i];
        // float x0 = q[base + 2 * i];
        // float x1 = q[base + 2 * i + 1];
        // q[base + 2 * i]     = x0 * c - x1 * sn;
        // q[base + 2 * i + 1] = x0 * sn + x1 * c;
    }
}

// ============================================================================
// Exercise 3: Fused Add + Norm (Matching fused_add_norm.metal)
// In-place x_residual[i] += residual[i]
// output[i] = x_residual[i] * rms * weight[c]
// ============================================================================
__global__ void fused_add_norm_kernel(float* x_residual, const float* residual,
                                      const float* weight, float* output,
                                      int D, float eps) {
    __shared__ float s_warp_sums[8];
    __shared__ float s_rms;

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int base = row_idx * D;

    // TODO:
    // 1. In-place add & square accumulation:
    //    float local_sq = 0.0f;
    //    for (int c = tid; c < D; c += blockDim.x) {
    //        float nv = x_residual[base + c] + residual[base + c];
    //        x_residual[base + c] = nv;
    //        local_sq += nv * nv;
    //    }
    // 2. Multi-warp reduction to compute s_rms = rsqrtf(sum / D + eps).
    // 3. Write output: output[base + c] = x_residual[base + c] * s_rms * weight[c].
}

// ============================================================================
// Exercise 4: RMSNorm Backward dX (Input Gradient)
// dx = (1 / rms) * (dy * w - xhat * (sum(dy * w * xhat) / D))
// ============================================================================
__global__ void rms_norm_backward_dx_kernel(const float* dy, const float* x,
                                            const float* weight, float* dx,
                                            int D, float eps) {
    __shared__ float s_warp_sums[8];
    __shared__ float s_rms;
    __shared__ float s_inner_mean;

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int base = row_idx * D;

    // TODO:
    // 1. Pass 1: Compute sum_sq of x[base + c] -> s_rms = 1.0f / sqrtf(sum_sq / D + eps).
    // 2. Pass 2: xhat = x * s_rms. Compute local_inner = dy * w * xhat.
    //            Reduce to s_inner_mean = sum(local_inner) / (float)D.
    // 3. Pass 3: dx[base + c] = s_rms * (dy[base + c] * weight[c] - xhat * s_inner_mean).
}

// ============================================================================
// Exercise 5: Multi-Head RoPE Backward
// Transpose rotation matrix: x0' = x0*c + x1*s; x1' = x1*c - x0*s;
// ============================================================================
__global__ void rope_backward_mha_kernel(float* grad_q, const float* cos_table,
                                         const float* sin_table,
                                         int batch, int heads, int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    int total_pairs = batch * heads * seq_len * half_dim;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < total_pairs) {
        // TODO:
        // int tmp = gid;
        // int i = tmp % half_dim; tmp /= half_dim;
        // int s = tmp % seq_len;  tmp /= seq_len;
        // int h = tmp % heads;
        // int b = tmp / heads;
        // int base = (b * heads * seq_len + h * seq_len + s) * head_dim;
        // float c = cos_table[s * half_dim + i];
        // float sn = sin_table[s * half_dim + i];
        // float g0 = grad_q[base + 2 * i];
        // float g1 = grad_q[base + 2 * i + 1];
        // grad_q[base + 2 * i]     = g0 * c + g1 * sn;
        // grad_q[base + 2 * i + 1] = g1 * c - g0 * sn;
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: Multi-Warp RMSNorm Forward ---
    {
        const int num_rows = 4;
        const int D = 512;
        const int total_elem = num_rows * D;
        float eps = 1e-5f;

        std::vector<float> h_in(total_elem), h_w(D, 1.5f), h_out(total_elem);
        for (int i = 0; i < total_elem; ++i) h_in[i] = static_cast<float>((i % D) + 1) * 0.05f;

        float *d_in, *d_w, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        rms_norm_forward_block_kernel<<<num_rows, 256>>>(d_in, d_out, d_w, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < num_rows; ++r) {
            float sum_sq = 0.0f;
            for (int c = 0; c < D; ++c) sum_sq += h_in[r * D + c] * h_in[r * D + c];
            float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
            for (int c = 0; c < D; ++c) {
                float exp = h_in[r * D + c] * rms * h_w[c];
                if (std::abs(h_out[r * D + c] - exp) > 1e-3f) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 1 Passed: Multi-warp RMSNorm forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Multi-warp RMSNorm forward." << std::endl;
        }

        cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    }

    // --- Test 2: Multi-Head RoPE Forward ---
    {
        const int B = 1, H = 2, S = 4, D = 16;
        const int half_dim = D / 2;
        const int total_pairs = B * H * S * half_dim;
        const int total_elem = B * H * S * D;

        std::vector<float> h_q(total_elem, 1.0f), h_cos(S * half_dim), h_sin(S * half_dim);
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < half_dim; ++i) {
                float angle = static_cast<float>(s * half_dim + i) * 0.1f;
                h_cos[s * half_dim + i] = std::cos(angle);
                h_sin[s * half_dim + i] = std::sin(angle);
            }
        }

        float *d_q, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_q, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (total_pairs + 255) / 256;
        rope_forward_mha_kernel<<<blocks, 256>>>(d_q, d_cos, d_sin, B, H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_q, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < half_dim; ++i) {
                float c = h_cos[s * half_dim + i];
                float sn = h_sin[s * half_dim + i];
                float exp0 = 1.0f * c - 1.0f * sn;
                float exp1 = 1.0f * sn + 1.0f * c;
                for (int h = 0; h < H; ++h) {
                    int base = (h * S + s) * D;
                    if (std::abs(h_out[base + 2 * i] - exp0) > 1e-4f) { ok = false; break; }
                    if (std::abs(h_out[base + 2 * i + 1] - exp1) > 1e-4f) { ok = false; break; }
                }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 2 Passed: Multi-head RoPE forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Multi-head RoPE forward." << std::endl;
        }

        cudaFree(d_q); cudaFree(d_cos); cudaFree(d_sin);
    }

    // --- Test 3: Fused Add + Norm ---
    {
        const int num_rows = 4;
        const int D = 512;
        const int total_elem = num_rows * D;
        float eps = 1e-5f;

        std::vector<float> h_x(total_elem, 1.0f), h_res(total_elem, 2.0f), h_w(D, 1.0f), h_out(total_elem);

        float *d_x, *d_res, *d_w, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        fused_add_norm_kernel<<<num_rows, 256>>>(d_x, d_res, d_w, d_out, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, total_elem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // x_new = 1 + 2 = 3.0f. rms of constant 3.0f is 1/3 = 0.33333f.
        // out = 3.0 * (1/3) * 1.0 = 1.0f.
        for (int i = 0; i < total_elem; ++i) {
            if (std::abs(h_x[i] - 3.0f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_out[i] - 1.0f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 3 Passed: Fused add-norm." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Fused add-norm." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_out);
    }

    // --- Test 4: RMSNorm Backward dX ---
    {
        const int num_rows = 2;
        const int D = 256;
        const int total_elem = num_rows * D;
        float eps = 1e-5f;

        std::vector<float> h_dy(total_elem, 1.0f), h_x(total_elem, 2.0f), h_w(D, 1.0f), h_dx(total_elem);

        for (int i = 0; i < total_elem; ++i) {
            h_x[i] = static_cast<float>((i % D) + 1) * 0.1f;
        }

        float *d_dy, *d_x, *d_w, *d_dx;
        CUDA_CHECK(cudaMalloc(&d_dy, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dx, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dx, 0, total_elem * sizeof(float)));

        rms_norm_backward_dx_kernel<<<num_rows, 256>>>(d_dy, d_x, d_w, d_dx, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < num_rows; ++r) {
            float sum_sq = 0.0f;
            for (int c = 0; c < D; ++c) sum_sq += h_x[r * D + c] * h_x[r * D + c];
            float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
            float inner_sum = 0.0f;
            for (int c = 0; c < D; ++c) {
                float xhat = h_x[r * D + c] * rms;
                inner_sum += h_dy[r * D + c] * h_w[c] * xhat;
            }
            float inner_mean = inner_sum / static_cast<float>(D);
            for (int c = 0; c < D; ++c) {
                float xhat = h_x[r * D + c] * rms;
                float exp_dx = rms * (h_dy[r * D + c] * h_w[c] - xhat * inner_mean);
                if (std::abs(h_dx[r * D + c] - exp_dx) > 1e-3f) { ok = false; break; }
            }
        }
        if (ok && h_dx[0] != 0.0f) {
            std::cout << "Test 4 Passed: RMSNorm backward input gradient (dx)." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: RMSNorm backward input gradient (dx)." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_x); cudaFree(d_w); cudaFree(d_dx);
    }

    // --- Test 5: Multi-Head RoPE Backward ---
    {
        const int B = 1, H = 2, S = 4, D = 16;
        const int half_dim = D / 2;
        const int total_pairs = B * H * S * half_dim;
        const int total_elem = B * H * S * D;

        std::vector<float> h_g(total_elem), h_cos(S * half_dim), h_sin(S * half_dim);
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < half_dim; ++i) {
                float angle = static_cast<float>(s * half_dim + i) * 0.1f;
                h_cos[s * half_dim + i] = std::cos(angle);
                h_sin[s * half_dim + i] = std::sin(angle);
                for (int h = 0; h < H; ++h) {
                    int base = (h * S + s) * D;
                    // Forward-rotated values
                    h_g[base + 2 * i]     = std::cos(angle) - std::sin(angle);
                    h_g[base + 2 * i + 1] = std::sin(angle) + std::cos(angle);
                }
            }
        }

        float *d_g, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_g, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (total_pairs + 255) / 256;
        rope_backward_mha_kernel<<<blocks, 256>>>(d_g, d_cos, d_sin, B, H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_g, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Transpose rotation should invert back to (1, 1)
        for (int i = 0; i < total_elem; ++i) {
            if (std::abs(h_out[i] - 1.0f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_out[0] == 1.0f) {
            std::cout << "Test 5 Passed: Multi-head RoPE backward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Multi-head RoPE backward." << std::endl;
        }

        cudaFree(d_g); cudaFree(d_cos); cudaFree(d_sin);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
