// ============================================================================
// Module 7.3: Normalization & RoPE - Champion Workbook
// Kernels Covered:
//   - rms_norm_backward_dw: Full weight gradient reduction across rows
//   - vectorized_rms_norm_fwd: 128-bit float4 multi-warp RMSNorm
//   - joint_rope_forward_qk: Interleaved Q & K RoPE forward matching rope_forward.metal
//   - fused_backward_add_norm: Dual output gradient passing to input & residual
//   - joint_rope_backward_qk: Inverted RoPE rotation on both dQ and dK
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

#define FULL_MASK 0xffffffff

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// ============================================================================
// Exercise 1: RMSNorm Weight Gradient Computation (dw)
// Matching rms_norm_backward_dw in rms_norm_backward.metal
// Grid: dims threads (or dims/4 if vectorized).
// Loop over all rows: dw[col] = sum_rows (dy[row, col] * (x[row, col] * inv_rms[row]))
// ============================================================================
__global__ void rms_norm_backward_dw_kernel(const float* dy, const float* x,
                                            const float* inv_rms_buf,
                                            float* dw,
                                            int num_rows, int dims) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < dims) {
        // TODO:
        // float acc = 0.0f;
        // for (int r = 0; r < num_rows; ++r) {
        //     float xhat = x[r * dims + col] * inv_rms_buf[r];
        //     acc += dy[r * dims + col] * xhat;
        // }
        // dw[col] = acc;
    }
}

// ============================================================================
// Exercise 2: Vectorized 128-bit RMSNorm Forward (float4)
// num_vec = dims / 4. 1 Block per row.
// ============================================================================
__global__ void vectorized_rms_norm_fwd_kernel(const float4* input, float4* output,
                                               const float4* weight,
                                               int num_vec, int dims, float eps) {
    __shared__ float s_warp_sums[8];
    __shared__ float s_rms;

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int base_vec = row_idx * num_vec;

    // TODO:
    // 1. Accumulate sum of squares using float4:
    //    float local_sq = 0.0f;
    //    for (int v = tid; v < num_vec; v += blockDim.x) {
    //        float4 val = input[base_vec + v];
    //        local_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    //    }
    // 2. Reduce to s_rms = rsqrtf(sum / (float)dims + eps).
    // 3. Write output:
    //    for (int v = tid; v < num_vec; v += blockDim.x) {
    //        float4 in_v = input[base_vec + v];
    //        float4 w_v = weight[v];
    //        float4 out_v;
    //        out_v.x = in_v.x * s_rms * w_v.x;
    //        out_v.y = in_v.y * s_rms * w_v.y;
    //        out_v.z = in_v.z * s_rms * w_v.z;
    //        out_v.w = in_v.w * s_rms * w_v.w;
    //        output[base_vec + v] = out_v;
    //    }
}

// ============================================================================
// Exercise 3: Joint Q and K RoPE Forward (Matching rope_forward.metal)
// Dispatches both Q and K in a single kernel launch.
// If gid < total_q, process Q; else process K (gid - total_q).
// ============================================================================
__global__ void joint_rope_forward_qk_kernel(float* q, float* k,
                                             const float* cos_table,
                                             const float* sin_table,
                                             int batch, int q_heads, int kv_heads,
                                             int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    int total_q  = batch * q_heads * seq_len * half_dim;
    int total_kv = batch * kv_heads * seq_len * half_dim;
    int total    = total_q + total_kv;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < total) {
        // TODO:
        // if (gid < total_q) {
        //     Decompose gid -> (b, h, s, i) for Q, apply rotation to q.
        // } else {
        //     Decompose (gid - total_q) -> (b, h, s, i) for K, apply rotation to k.
        // }
    }
}

// ============================================================================
// Exercise 4: Fused Backward Add-Norm
// Computes dx for RMSNorm, and simultaneously passes gradient to d_residual:
// d_residual[idx] = dx[idx]
// ============================================================================
__global__ void fused_backward_add_norm_kernel(const float* dy, const float* x,
                                               const float* weight,
                                               float* dx, float* d_residual,
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
    // 1. Multi-warp reduction to compute s_rms and s_inner_mean.
    // 2. Compute dx = s_rms * (dy * weight - xhat * s_inner_mean).
    // 3. Store dx[base + c] = dx_val, and d_residual[base + c] = dx_val.
}

// ============================================================================
// Exercise 5: Joint Q and K RoPE Backward
// Dispatches both dQ and dK in a single kernel with transposed rotation matrix.
// ============================================================================
__global__ void joint_rope_backward_qk_kernel(float* dq, float* dk,
                                              const float* cos_table,
                                              const float* sin_table,
                                              int batch, int q_heads, int kv_heads,
                                              int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    int total_q  = batch * q_heads * seq_len * half_dim;
    int total_kv = batch * kv_heads * seq_len * half_dim;
    int total    = total_q + total_kv;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < total) {
        // TODO:
        // if (gid < total_q) {
        //     apply transposed rotation to dq
        // } else {
        //     apply transposed rotation to dk
        // }
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: RMSNorm Backward dw ---
    {
        const int num_rows = 4;
        const int dims = 64;
        const int total_elem = num_rows * dims;

        std::vector<float> h_dy(total_elem, 1.0f), h_x(total_elem, 2.0f), h_inv_rms(num_rows, 0.5f);
        std::vector<float> h_dw(dims, 0.0f);

        float *d_dy, *d_x, *d_inv_rms, *d_dw;
        CUDA_CHECK(cudaMalloc(&d_dy, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inv_rms, num_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw, dims * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_rms, h_inv_rms.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dw, 0, dims * sizeof(float)));

        rms_norm_backward_dw_kernel<<<1, dims>>>(d_dy, d_x, d_inv_rms, d_dw, num_rows, dims);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dw.data(), d_dw, dims * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Each row contributes dy * x * inv_rms = 1.0 * 2.0 * 0.5 = 1.0f
        // 4 rows -> dw = 4.0f
        for (int c = 0; c < dims; ++c) {
            if (std::abs(h_dw[c] - 4.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_dw[0] == 4.0f) {
            std::cout << "Test 1 Passed: RMSNorm backward weight gradient (dw)." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: RMSNorm backward weight gradient (dw)." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_x); cudaFree(d_inv_rms); cudaFree(d_dw);
    }

    // --- Test 2: Vectorized 128-bit RMSNorm Forward ---
    {
        const int num_rows = 2;
        const int dims = 128;
        const int num_vec = dims / 4;
        const int total_elem = num_rows * dims;
        float eps = 1e-5f;

        std::vector<float> h_in(total_elem, 2.0f), h_w(dims, 1.0f), h_out(total_elem);

        float *d_in, *d_w, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, dims * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), dims * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        vectorized_rms_norm_fwd_kernel<<<num_rows, 128>>>(
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            reinterpret_cast<const float4*>(d_w),
            num_vec, dims, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // in = 2.0 -> sum_sq = 128 * 4 -> rms = 0.5f -> out = 1.0f
        for (int i = 0; i < total_elem; ++i) {
            if (std::abs(h_out[i] - 1.0f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 2 Passed: 128-bit vectorized RMSNorm forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: 128-bit vectorized RMSNorm forward." << std::endl;
        }

        cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    }

    // --- Test 3: Joint Q and K RoPE Forward ---
    {
        const int B = 1, Q_H = 2, KV_H = 1, S = 2, D = 8;
        const int half_dim = D / 2;
        const int total_q = B * Q_H * S * D;
        const int total_kv = B * KV_H * S * D;

        std::vector<float> h_q(total_q), h_k(total_kv);
        std::vector<float> h_cos(S * half_dim), h_sin(S * half_dim);
        for (int i = 0; i < total_q; ++i) h_q[i] = 1.0f;
        for (int i = 0; i < total_kv; ++i) h_k[i] = 1.0f;
        for (int i = 0; i < S * half_dim; ++i) {
            float a = 0.785398f; // 45 degrees
            h_cos[i] = std::cos(a);
            h_sin[i] = std::sin(a);
        }

        float *d_q, *d_k, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_q, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, total_kv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total_kv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));

        int total_pairs = (total_q + total_kv) / 2;
        joint_rope_forward_qk_kernel<<<(total_pairs + 63) / 64, 64>>>(
            d_q, d_k, d_cos, d_sin, B, Q_H, KV_H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, total_q * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_k.data(), d_k, total_kv * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // With x=(1, 1), x0' = cos - sin = 0, x1' = sin + cos = sqrt(2) ~ 1.4142f
        float exp0 = h_cos[0] - h_sin[0];
        float exp1 = h_sin[0] + h_cos[0];
        for (int i = 0; i < total_q / 2; ++i) {
            if (std::abs(h_q[2 * i] - exp0) > 1e-4f) { ok = false; break; }
            if (std::abs(h_q[2 * i + 1] - exp1) > 1e-4f) { ok = false; break; }
        }
        for (int i = 0; i < total_kv / 2; ++i) {
            if (std::abs(h_k[2 * i] - exp0) > 1e-4f) { ok = false; break; }
            if (std::abs(h_k[2 * i + 1] - exp1) > 1e-4f) { ok = false; break; }
        }

        if (ok && std::abs(h_q[0] - exp0) <= 1e-4f) {
            std::cout << "Test 3 Passed: Joint Q and K RoPE forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Joint Q and K RoPE forward." << std::endl;
        }

        cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    }

    // --- Test 4: Fused Backward Add-Norm ---
    {
        const int num_rows = 2;
        const int D = 128;
        const int total_elem = num_rows * D;
        float eps = 1e-5f;

        std::vector<float> h_dy(total_elem, 1.0f), h_x(total_elem), h_w(D, 1.0f);
        std::vector<float> h_dx(total_elem), h_dres(total_elem);
        for (int i = 0; i < total_elem; ++i) h_x[i] = static_cast<float>((i % D) + 1) * 0.1f;

        float *d_dy, *d_x, *d_w, *d_dx, *d_dres;
        CUDA_CHECK(cudaMalloc(&d_dy, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dx, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dres, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dx, 0, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dres, 0, total_elem * sizeof(float)));

        fused_backward_add_norm_kernel<<<num_rows, 128>>>(d_dy, d_x, d_w, d_dx, d_dres, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, total_elem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dres.data(), d_dres, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < total_elem; ++i) {
            if (h_dx[i] != h_dres[i]) { ok = false; break; }
        }
        if (ok && h_dx[0] != 0.0f) {
            std::cout << "Test 4 Passed: Fused backward add-norm." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused backward add-norm." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_x); cudaFree(d_w); cudaFree(d_dx); cudaFree(d_dres);
    }

    // --- Test 5: Joint Q and K RoPE Backward ---
    {
        const int B = 1, Q_H = 2, KV_H = 1, S = 2, D = 8;
        const int half_dim = D / 2;
        const int total_q = B * Q_H * S * D;
        const int total_kv = B * KV_H * S * D;

        std::vector<float> h_dq(total_q), h_dk(total_kv);
        std::vector<float> h_cos(S * half_dim), h_sin(S * half_dim);
        float a = 0.785398f;
        for (int i = 0; i < S * half_dim; ++i) {
            h_cos[i] = std::cos(a);
            h_sin[i] = std::sin(a);
        }
        // Set rotated values
        for (int i = 0; i < total_q / 2; ++i) {
            h_dq[2 * i] = h_cos[0] - h_sin[0];
            h_dq[2 * i + 1] = h_sin[0] + h_cos[0];
        }
        for (int i = 0; i < total_kv / 2; ++i) {
            h_dk[2 * i] = h_cos[0] - h_sin[0];
            h_dk[2 * i + 1] = h_sin[0] + h_cos[0];
        }

        float *d_dq, *d_dk, *d_cos, *d_sin;
        CUDA_CHECK(cudaMalloc(&d_dq, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dk, total_kv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, S * half_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dq, h_dq.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dk, h_dk.data(), total_kv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), S * half_dim * sizeof(float), cudaMemcpyHostToDevice));

        int total_pairs = (total_q + total_kv) / 2;
        joint_rope_backward_qk_kernel<<<(total_pairs + 63) / 64, 64>>>(
            d_dq, d_dk, d_cos, d_sin, B, Q_H, KV_H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dq.data(), d_dq, total_q * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, total_kv * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Inverse rotation should recover original (1.0, 1.0)
        for (int i = 0; i < total_q; ++i) { if (std::abs(h_dq[i] - 1.0f) > 1e-4f) { ok = false; break; } }
        for (int i = 0; i < total_kv; ++i) { if (std::abs(h_dk[i] - 1.0f) > 1e-4f) { ok = false; break; } }

        if (ok && std::abs(h_dq[0] - 1.0f) <= 1e-4f) {
            std::cout << "Test 5 Passed: Joint Q and K RoPE backward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Joint Q and K RoPE backward." << std::endl;
        }

        cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_cos); cudaFree(d_sin);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
