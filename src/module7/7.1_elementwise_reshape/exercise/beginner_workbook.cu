// ============================================================================
// Module 7.1: Elementwise & Reshaping - Beginner Workbook
// Kernels Covered:
//   - reshape_3d: 3D coordinate decomposition and transpose
//   - reshape_4d: 4D multi-head tensor permutation [B, H, S, D] -> [B, S, H, D]
//   - residual_add: In-place residual addition a[i] += b[i]
//   - swiglu_forward: SwiGLU activation out = SiLU(gate) * up
//   - swiglu_backward_up: Gradient of SwiGLU w.r.t up projection
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

// ============================================================================
// Exercise 1: 3D Coordinate Transpose [B, S, D] -> [S, B, D]
// ============================================================================
__global__ void reshape_3d_kernel(const float* src, float* dst,
                                  int B, int S, int D) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (gid < total) {
        // TODO:
        // 1. Decompose gid into (b, s, d) for src shape [B, S, D]:
        //    int tmp = gid;
        //    int d = tmp % D; tmp /= D;
        //    int s = tmp % S; tmp /= S;
        //    int b = tmp;
        // 2. Compute dst_idx for shape [S, B, D]:
        //    dst_idx = (s * B + b) * D + d;
        // 3. Write dst[dst_idx] = src[gid];
    }
}

// ============================================================================
// Exercise 2: 4D Multi-Head Attention Reshape [B, H, S, D] -> [B, S, H, D]
// Matching reshape_3d.metal in cpp/src/gpu_kernel
// ============================================================================
__global__ void reshape_4d_kernel(const float* src, float* dst,
                                  int batch, int n_heads, int seq_len, int head_dim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_heads * seq_len * head_dim;
    if (gid < total) {
        // TODO:
        // Decompose gid:
        //   int tmp = gid;
        //   int d = tmp % head_dim; tmp /= head_dim;
        //   int s = tmp % seq_len;  tmp /= seq_len;
        //   int h = tmp % n_heads;  tmp /= n_heads;
        //   int b = tmp;
        // Compute src_idx: (b * n_heads * seq_len + h * seq_len + s) * head_dim + d;
        // Compute dst_idx: (b * seq_len * n_heads + s * n_heads + h) * head_dim + d;
        // dst[dst_idx] = src[src_idx];
    }
}

// ============================================================================
// Exercise 3: In-Place Residual Add: a[i] += b[i]
// Matching residual_add.metal
// ============================================================================
__global__ void residual_add_kernel(float* a, const float* b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO: Perform in-place addition a[gid] += b[gid];
    }
}

// ============================================================================
// Exercise 4: SwiGLU Forward Activation
// out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i]
// Matching swiglu_forward.metal
// ============================================================================
__global__ void swiglu_forward_kernel(const float* gate, const float* up,
                                      float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO:
        // float g = gate[gid];
        // float u = up[gid];
        // float silu_g = g / (1.0f + expf(-g));
        // out[gid] = silu_g * u;
    }
}

// ============================================================================
// Exercise 5: SwiGLU Backward w.r.t Up: grad_up[i] = grad_out[i] * SiLU(gate[i])
// ============================================================================
__global__ void swiglu_backward_up_kernel(const float* grad_out, const float* gate,
                                          float* grad_up, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO:
        // float dy = grad_out[gid];
        // float g = gate[gid];
        // float silu_g = g / (1.0f + expf(-g));
        // grad_up[gid] = dy * silu_g;
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int blockSize = 256;

    // --- Test 1: reshape_3d ---
    {
        const int B = 2, S = 4, D = 8;
        const int total_elem = B * S * D;
        std::vector<float> h_src(total_elem), h_dst(total_elem);
        for (int i = 0; i < total_elem; ++i) h_src[i] = static_cast<float>(i + 1);

        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dst, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        reshape_3d_kernel<<<blocks, blockSize>>>(d_src, d_dst, B, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    int src_idx = (b * S + s) * D + d;
                    int dst_idx = (s * B + b) * D + d;
                    if (h_dst[dst_idx] != h_src[src_idx]) { ok = false; break; }
                }
            }
        }
        if (ok && h_dst[0] != 0.0f) {
            std::cout << "Test 1 Passed: reshape_3d [B, S, D] -> [S, B, D]." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: reshape_3d." << std::endl;
        }

        cudaFree(d_src); cudaFree(d_dst);
    }

    // --- Test 2: reshape_4d ---
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
        reshape_4d_kernel<<<blocks, blockSize>>>(d_src, d_dst, B, H, S, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < D; ++d) {
                        int src_idx = (b * H * S + h * S + s) * D + d;
                        int dst_idx = (b * S * H + s * H + h) * D + d;
                        if (h_dst[dst_idx] != h_src[src_idx]) { ok = false; break; }
                    }
                }
            }
        }
        if (ok && h_dst[0] != 0.0f) {
            std::cout << "Test 2 Passed: reshape_4d [B, H, S, D] -> [B, S, H, D]." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: reshape_4d." << std::endl;
        }

        cudaFree(d_src); cudaFree(d_dst);
    }

    // --- Test 3: residual_add ---
    {
        const int N = 1024;
        std::vector<float> h_a(N, 1.5f), h_b(N, 2.5f);

        float *d_a, *d_b;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (N + blockSize - 1) / blockSize;
        residual_add_kernel<<<blocks, blockSize>>>(d_a, d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_a[i] - 4.0f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_a[0] == 4.0f) {
            std::cout << "Test 3 Passed: residual_add in-place a[i] += b[i]." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: residual_add." << std::endl;
        }

        cudaFree(d_a); cudaFree(d_b);
    }

    // --- Test 4: swiglu_forward ---
    {
        const int N = 512;
        std::vector<float> h_gate(N), h_up(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_gate[i] = static_cast<float>(i - 256) * 0.05f;
            h_up[i] = 1.0f + static_cast<float>(i) * 0.01f;
        }

        float *d_gate, *d_up, *d_out;
        CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_forward_kernel<<<blocks, blockSize>>>(d_gate, d_up, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = h_gate[i];
            float u = h_up[i];
            float exp_val = (g / (1.0f + std::exp(-g))) * u;
            if (std::abs(h_out[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[N - 1] != 0.0f) {
            std::cout << "Test 4 Passed: swiglu_forward." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: swiglu_forward." << std::endl;
        }

        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    }

    // --- Test 5: swiglu_backward_up ---
    {
        const int N = 512;
        std::vector<float> h_dy(N), h_gate(N), h_dup(N);
        for (int i = 0; i < N; ++i) {
            h_dy[i] = 2.0f;
            h_gate[i] = static_cast<float>(i) * 0.01f;
        }

        float *d_dy, *d_gate, *d_dup;
        CUDA_CHECK(cudaMalloc(&d_dy, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dup, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dup, 0, N * sizeof(float)));

        int blocks = (N + blockSize - 1) / blockSize;
        swiglu_backward_up_kernel<<<blocks, blockSize>>>(d_dy, d_gate, d_dup, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dup.data(), d_dup, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = h_gate[i];
            float silu_g = g / (1.0f + std::exp(-g));
            float exp_val = h_dy[i] * silu_g;
            if (std::abs(h_dup[i] - exp_val) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_dup[N - 1] != 0.0f) {
            std::cout << "Test 5 Passed: swiglu_backward_up." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: swiglu_backward_up." << std::endl;
        }

        cudaFree(d_dy); cudaFree(d_gate); cudaFree(d_dup);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
