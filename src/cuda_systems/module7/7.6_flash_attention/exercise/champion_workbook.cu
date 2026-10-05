// ==============================================================================
// Module 7.6: FlashAttention Suite — Champion Workbook
// ==============================================================================
// In this final champion workbook, you will implement the production-grade
// FlashAttention-2 forward and backward attention engine (flash_attn_fwd.metal
// & fused_attn_bwd.metal):
// 1. Multi-Head FlashAttention-2 Forward Kernel with Online Softmax & Stats Storage
// 2. Backward Precomputation Kernel: D_i = sum_d (dO_{i, d} * O_{i, d})
// 3. Backward Value Gradient Kernel: dV = P^T * dO with on-the-fly P recomputation
// 4. Backward Query & Key Gradient Kernel: dQ, dK via dS = P * (dO * V^T - D)
// 5. Fused Attention Backward Kernel (fused_attn_bwd.metal):
//    Jointly computing dQ, dK, dV in a single pass
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ==============================================================================
// Exercise 1: Multi-Head FlashAttention-2 Forward Kernel (flash_attn_fwd.metal)
// Computes O [B, H, S, D] and stores m [B, H, S] and l [B, H, S] for backprop.
// Grid: (S / 16, H, B), Block: (16, 16)
// S is sequence length, D is head dimension (e.g. 16).
// ==============================================================================
__global__ void flash_attn2_mha_fwd_kernel(const float* Q, const float* K, const float* V,
                                           float* O, float* stats_m, float* stats_l,
                                           int B, int H, int S, int D, float scale) {
    __shared__ float s_Q[16][16];
    __shared__ float s_K[16][16];
    __shared__ float s_V[16][16];
    __shared__ float s_S[16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];
    __shared__ float s_O[16][16];

    // TODO:
    // 1. Identify batch b = blockIdx.z, head h = blockIdx.y, q_tile = blockIdx.x.
    // 2. Offsets for this (batch, head):
    //    head_offset = (b * H + h) * (S * D);
    //    stats_offset = (b * H + h) * S;
    // 3. Initialize s_m = -1e30f, s_l = 0.0f, s_O = 0.0f.
    // 4. Load Q tile for rows [q_tile * 16, (q_tile + 1) * 16 - 1].
    // 5. Loop over K/V tiles j in [0, S / 16 - 1]:
    //    - Skip if j > q_tile
    //    - Load K, V tiles into shared memory
    //    - Compute S_tile = Q_tile * K_tile^T * scale
    //    - If j == q_tile, apply causal mask (if col > row, dot = -1e30f)
    //    - Update running max m, sum l, alpha
    //    - Rescale s_O and accumulate P_tile * V_tile
    // 6. Normalize s_O / s_l and write to O[head_offset + ...].
    // 7. Store stats_m and stats_l into global memory.
}

// ==============================================================================
// Exercise 2: Backward Precomputation Kernel (D vector)
// For each token row i:
// D_i = sum_{d=0}^{D-1} dO_{i, d} * O_{i, d}
// Shape of dO, O: [N_tokens x D]. Output D: [N_tokens].
// Block: 256, Grid: (N_tokens + 255) / 256
// ==============================================================================
__global__ void compute_D_vector_kernel(const float* dO, const float* O, float* D_vec,
                                        int N_tokens, int D) {
    // TODO:
    // 1. row = blockIdx.x * blockDim.x + threadIdx.x
    // 2. If row < N_tokens:
    //    float sum = 0.0f;
    //    for d = 0 to D-1: sum += dO[row * D + d] * O[row * D + d];
    //    D_vec[row] = sum;
}

// ==============================================================================
// Exercise 3: FlashAttention-2 Backward Value Gradient Kernel
// dV_j = sum_i P_{ij}^T * dO_i
// P_{ij} is recomputed in SRAM on-the-fly using stored m_i and l_i:
//   S_{ij} = (Q_i . K_j) / sqrt(d) (with causal mask)
//   P_{ij} = exp(S_{ij} - m_i) / l_i
//   dV[j, d] = sum_{i >= j} P_{ij} * dO[i, d]
// For single head: Q, K, dO [S x D], stats_m, stats_l [S], output dV [S x D].
// Block: (16, 16), Grid: ((D + 15) / 16, (S + 15) / 16)
// ==============================================================================
__global__ void flash_attn2_bwd_dV_kernel(const float* Q, const float* K, const float* dO,
                                          const float* stats_m, const float* stats_l,
                                          float* dV, int S, int D, float scale) {
    // TODO:
    // Each thread (row, d) computes dV[j, d] where j = row (0 to S-1).
    // row = blockIdx.y * blockDim.y + threadIdx.y (key index j)
    // col = blockIdx.x * blockDim.x + threadIdx.x (feature d)
    // If row < S && col < D:
    //   float acc = 0.0f;
    //   for i = row to S-1:
    //     float dot = 0.0f;
    //     for d2 = 0 to D-1: dot += Q[i * D + d2] * K[row * D + d2];
    //     float s = dot * scale;
    //     float p = expf(s - stats_m[i]) / stats_l[i];
    //     acc += p * dO[i * D + col];
    //   dV[row * D + col] = acc;
}

// ==============================================================================
// Exercise 4: FlashAttention-2 Backward Query & Key Gradients Kernel
// Softmax gradient:
//   dS_{ij} = P_{ij} * (sum_d dO_{i, d} * V_{j, d} - D_i)
// Gradients:
//   dQ_i = scale * sum_j dS_{ij} * K_j
//   dK_j = scale * sum_i dS_{ij} * Q_i
// Block: (16, 16), Grid: ((D + 15) / 16, (S + 15) / 16)
// ==============================================================================
__global__ void flash_attn2_bwd_dQ_dK_kernel(const float* Q, const float* K, const float* V,
                                             const float* dO, const float* D_vec,
                                             const float* stats_m, const float* stats_l,
                                             float* dQ, float* dK,
                                             int S, int D, float scale) {
    // TODO:
    // Compute dQ and dK elements.
}

// ==============================================================================
// Exercise 5: Fused Attention Backward Kernel (fused_attn_bwd.metal)
// Jointly computes dQ, dK, and dV in a single consolidated kernel pass.
// Inputs: Q, K, V, dO, D_vec, stats_m, stats_l [S x D or S]
// Outputs: dQ, dK, dV [S x D]
// Block: (16, 16), Grid: ((D + 15) / 16, (S + 15) / 16)
// ==============================================================================
__global__ void fused_attn_bwd_joint_kernel(const float* Q, const float* K, const float* V,
                                            const float* dO, const float* D_vec,
                                            const float* stats_m, const float* stats_l,
                                            float* dQ, float* dK, float* dV,
                                            int S, int D, float scale) {
    // TODO:
    // For token row in [0, S-1] and feature col in [0, D-1]:
    // Jointly accumulate:
    // 1. dQ[row, col] = scale * sum_{j <= row} [ P_{row, j} * (sum_d dO[row, d]*V[j, d] - D_vec[row]) ] * K[j, col]
    // 2. dK[row, col] = scale * sum_{i >= row} [ P_{i, row} * (sum_d dO[i, d]*V[row, d] - D_vec[i]) ] * Q[i, col]
    // 3. dV[row, col] = sum_{i >= row} P_{i, row} * dO[i, col]
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Multi-Head FlashAttention-2 Forward Kernel
    // --------------------------------------------------------------------------
    {
        int B = 1, H = 2, S = 32, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        int total_elements = B * H * S * D;
        int total_stats = B * H * S;

        std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
        std::vector<float> h_O(total_elements, 0.0f), h_ref_O(total_elements, 0.0f);
        std::vector<float> h_m(total_stats, 0.0f), h_l(total_stats, 0.0f);

        for (int i = 0; i < total_elements; ++i) {
            h_Q[i] = 0.05f * ((i % 7) + 1);
            h_K[i] = 0.05f * ((i % 11) + 1);
            h_V[i] = 0.1f  * ((i % 13) + 1);
        }

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                int off = (b * H + h) * S * D;
                for (int r = 0; r < S; ++r) {
                    float max_s = -1e30f;
                    std::vector<float> s_row(r + 1);
                    for (int j = 0; j <= r; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) dot += h_Q[off + r * D + d] * h_K[off + j * D + d];
                        s_row[j] = dot * scale;
                        if (s_row[j] > max_s) max_s = s_row[j];
                    }
                    float sum_exp = 0.0f;
                    for (int j = 0; j <= r; ++j) sum_exp += std::exp(s_row[j] - max_s);
                    for (int d = 0; d < D; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j <= r; ++j) {
                            float p = std::exp(s_row[j] - max_s) / sum_exp;
                            acc += p * h_V[off + j * D + d];
                        }
                        h_ref_O[off + r * D + d] = acc;
                    }
                }
            }
        }

        float *d_Q, *d_K, *d_V, *d_O, *d_m, *d_l;
        CHECK_CUDA(cudaMalloc(&d_Q, total_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, total_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, total_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_O, total_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_m, total_stats * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_l, total_stats * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_O, 0, total_elements * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid(S / 16, H, B);
        flash_attn2_mha_fwd_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_m, d_l, B, H, S, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < total_elements; ++i) {
            if (std::fabs(h_O[i] - h_ref_O[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 1: Multi-Head FlashAttn-2 Forward] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Multi-Head FlashAttn-2 Forward] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_m); cudaFree(d_l);
    }

    // --------------------------------------------------------------------------
    // Test 2: Backward Precomputation (D vector)
    // --------------------------------------------------------------------------
    {
        int N_tokens = 32, D = 16;
        std::vector<float> h_dO(N_tokens * D), h_O(N_tokens * D),
                           h_D(N_tokens, 0.0f), h_ref(N_tokens, 0.0f);
        for (int i = 0; i < N_tokens * D; ++i) {
            h_dO[i] = 0.1f * ((i % 7) - 3);
            h_O[i]  = 0.2f * ((i % 5) + 1);
        }

        for (int i = 0; i < N_tokens; ++i) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) sum += h_dO[i * D + d] * h_O[i * D + d];
            h_ref[i] = sum;
        }

        float *d_dO, *d_O, *d_D;
        CHECK_CUDA(cudaMalloc(&d_dO, N_tokens * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_O, N_tokens * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_D, N_tokens * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), N_tokens * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_O, h_O.data(), N_tokens * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_D, 0, N_tokens * sizeof(float)));

        compute_D_vector_kernel<<<(N_tokens + 255) / 256, 256>>>(d_dO, d_O, d_D, N_tokens, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, N_tokens * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N_tokens; ++i) {
            if (std::fabs(h_D[i] - h_ref[i]) > 1e-4f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Backward Precompute D Vector] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Backward Precompute D Vector] FAILED" << std::endl;
        }

        cudaFree(d_dO); cudaFree(d_O); cudaFree(d_D);
    }

    // --------------------------------------------------------------------------
    // Test 3: Backward Value Gradient (dV)
    // --------------------------------------------------------------------------
    {
        int S = 16, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> h_Q(S * D), h_K(S * D), h_dO(S * D),
                           h_m(S), h_l(S), h_dV(S * D, 0.0f), h_ref(S * D, 0.0f);

        for (int i = 0; i < S * D; ++i) {
            h_Q[i]  = 0.1f * ((i % 5) + 1);
            h_K[i]  = 0.1f * ((i % 7) + 1);
            h_dO[i] = 0.05f * ((i % 9) - 4);
        }

        for (int i = 0; i < S; ++i) {
            float max_s = -1e30f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                float s = dot * scale;
                if (s > max_s) max_s = s;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                sum_exp += std::exp(dot * scale - max_s);
            }
            h_m[i] = max_s;
            h_l[i] = sum_exp;
        }

        for (int j = 0; j < S; ++j) {
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int i = j; i < S; ++i) {
                    float dot = 0.0f;
                    for (int d2 = 0; d2 < D; ++d2) dot += h_Q[i * D + d2] * h_K[j * D + d2];
                    float p = std::exp(dot * scale - h_m[i]) / h_l[i];
                    acc += p * h_dO[i * D + d];
                }
                h_ref[j * D + d] = acc;
            }
        }

        float *d_Q, *d_K, *d_dO, *d_m, *d_l, *d_dV;
        CHECK_CUDA(cudaMalloc(&d_Q, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dO, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_m, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_l, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dV, S * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_m, h_m.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_l, h_l.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dV, 0, S * D * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (S + 15) / 16);
        flash_attn2_bwd_dV_kernel<<<grid, block>>>(d_Q, d_K, d_dO, d_m, d_l, d_dV, S, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dV.data(), d_dV, S * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S * D; ++i) {
            if (std::fabs(h_dV[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Backward Value Gradient dV] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Backward Value Gradient dV] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_dO); cudaFree(d_m); cudaFree(d_l); cudaFree(d_dV);
    }

    // --------------------------------------------------------------------------
    // Test 4: Backward Query & Key Gradients (dQ, dK)
    // --------------------------------------------------------------------------
    {
        int S = 16, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> h_Q(S * D), h_K(S * D), h_V(S * D), h_dO(S * D),
                           h_D(S, 0.0f), h_m(S), h_l(S),
                           h_dQ(S * D, 0.0f), h_dK(S * D, 0.0f),
                           h_ref_dQ(S * D, 0.0f), h_ref_dK(S * D, 0.0f);

        for (int i = 0; i < S * D; ++i) {
            h_Q[i]  = 0.1f * ((i % 5) + 1);
            h_K[i]  = 0.1f * ((i % 7) + 1);
            h_V[i]  = 0.15f * ((i % 11) + 1);
            h_dO[i] = 0.05f * ((i % 9) - 4);
        }

        std::vector<float> h_O(S * D, 0.0f);
        for (int i = 0; i < S; ++i) {
            float max_s = -1e30f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                float s = dot * scale;
                if (s > max_s) max_s = s;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                sum_exp += std::exp(dot * scale - max_s);
            }
            h_m[i] = max_s;
            h_l[i] = sum_exp;
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d2 = 0; d2 < D; ++d2) dot += h_Q[i * D + d2] * h_K[j * D + d2];
                    float p = std::exp(dot * scale - max_s) / sum_exp;
                    acc += p * h_V[j * D + d];
                }
                h_O[i * D + d] = acc;
            }
            float d_sum = 0.0f;
            for (int d = 0; d < D; ++d) d_sum += h_dO[i * D + d] * h_O[i * D + d];
            h_D[i] = d_sum;
        }

        for (int i = 0; i < S; ++i) {
            for (int col = 0; col < D; ++col) {
                float acc_q = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float dot_qk = 0.0f;
                    for (int d = 0; d < D; ++d) dot_qk += h_Q[i * D + d] * h_K[j * D + d];
                    float p = std::exp(dot_qk * scale - h_m[i]) / h_l[i];
                    float dot_dov = 0.0f;
                    for (int d = 0; d < D; ++d) dot_dov += h_dO[i * D + d] * h_V[j * D + d];
                    float ds = p * (dot_dov - h_D[i]);
                    acc_q += ds * h_K[j * D + col];
                }
                h_ref_dQ[i * D + col] = acc_q * scale;
            }
        }

        for (int j = 0; j < S; ++j) {
            for (int col = 0; col < D; ++col) {
                float acc_k = 0.0f;
                for (int i = j; i < S; ++i) {
                    float dot_qk = 0.0f;
                    for (int d = 0; d < D; ++d) dot_qk += h_Q[i * D + d] * h_K[j * D + d];
                    float p = std::exp(dot_qk * scale - h_m[i]) / h_l[i];
                    float dot_dov = 0.0f;
                    for (int d = 0; d < D; ++d) dot_dov += h_dO[i * D + d] * h_V[j * D + d];
                    float ds = p * (dot_dov - h_D[i]);
                    acc_k += ds * h_Q[i * D + col];
                }
                h_ref_dK[j * D + col] = acc_k * scale;
            }
        }

        float *d_Q, *d_K, *d_V, *d_dO, *d_D, *d_m, *d_l, *d_dQ, *d_dK;
        CHECK_CUDA(cudaMalloc(&d_Q, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dO, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_D, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_m, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_l, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dQ, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dK, S * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_D, h_D.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_m, h_m.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_l, h_l.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dQ, 0, S * D * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dK, 0, S * D * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (S + 15) / 16);
        flash_attn2_bwd_dQ_dK_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_D, d_m, d_l, d_dQ, d_dK, S, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dQ.data(), d_dQ, S * D * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_dK.data(), d_dK, S * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S * D; ++i) {
            if (std::fabs(h_dQ[i] - h_ref_dQ[i]) > 1e-3f ||
                std::fabs(h_dK[i] - h_ref_dK[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: Backward dQ & dK Gradients] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Backward dQ & dK Gradients] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_dO); cudaFree(d_D);
        cudaFree(d_m); cudaFree(d_l); cudaFree(d_dQ); cudaFree(d_dK);
    }

    // --------------------------------------------------------------------------
    // Test 5: Fused Joint Attention Backward Kernel
    // --------------------------------------------------------------------------
    {
        int S = 16, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> h_Q(S * D), h_K(S * D), h_V(S * D), h_dO(S * D),
                           h_D(S, 0.0f), h_m(S), h_l(S),
                           h_dQ(S * D, 0.0f), h_dK(S * D, 0.0f), h_dV(S * D, 0.0f),
                           h_ref_dQ(S * D, 0.0f), h_ref_dK(S * D, 0.0f), h_ref_dV(S * D, 0.0f);

        for (int i = 0; i < S * D; ++i) {
            h_Q[i]  = 0.1f * ((i % 5) + 1);
            h_K[i]  = 0.1f * ((i % 7) + 1);
            h_V[i]  = 0.15f * ((i % 11) + 1);
            h_dO[i] = 0.05f * ((i % 9) - 4);
        }

        std::vector<float> h_O(S * D, 0.0f);
        for (int i = 0; i < S; ++i) {
            float max_s = -1e30f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                float s = dot * scale;
                if (s > max_s) max_s = s;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[i * D + d] * h_K[j * D + d];
                sum_exp += std::exp(dot * scale - max_s);
            }
            h_m[i] = max_s;
            h_l[i] = sum_exp;
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d2 = 0; d2 < D; ++d2) dot += h_Q[i * D + d2] * h_K[j * D + d2];
                    float p = std::exp(dot * scale - max_s) / sum_exp;
                    acc += p * h_V[j * D + d];
                }
                h_O[i * D + d] = acc;
            }
            float d_sum = 0.0f;
            for (int d = 0; d < D; ++d) d_sum += h_dO[i * D + d] * h_O[i * D + d];
            h_D[i] = d_sum;
        }

        for (int i = 0; i < S; ++i) {
            for (int col = 0; col < D; ++col) {
                float acc_q = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float dot_qk = 0.0f;
                    for (int d = 0; d < D; ++d) dot_qk += h_Q[i * D + d] * h_K[j * D + d];
                    float p = std::exp(dot_qk * scale - h_m[i]) / h_l[i];
                    float dot_dov = 0.0f;
                    for (int d = 0; d < D; ++d) dot_dov += h_dO[i * D + d] * h_V[j * D + d];
                    float ds = p * (dot_dov - h_D[i]);
                    acc_q += ds * h_K[j * D + col];
                }
                h_ref_dQ[i * D + col] = acc_q * scale;
            }
        }

        for (int j = 0; j < S; ++j) {
            for (int col = 0; col < D; ++col) {
                float acc_k = 0.0f;
                float acc_v = 0.0f;
                for (int i = j; i < S; ++i) {
                    float dot_qk = 0.0f;
                    for (int d = 0; d < D; ++d) dot_qk += h_Q[i * D + d] * h_K[j * D + d];
                    float p = std::exp(dot_qk * scale - h_m[i]) / h_l[i];
                    float dot_dov = 0.0f;
                    for (int d = 0; d < D; ++d) dot_dov += h_dO[i * D + d] * h_V[j * D + d];
                    float ds = p * (dot_dov - h_D[i]);
                    acc_k += ds * h_Q[i * D + col];
                    acc_v += p * h_dO[i * D + col];
                }
                h_ref_dK[j * D + col] = acc_k * scale;
                h_ref_dV[j * D + col] = acc_v;
            }
        }

        float *d_Q, *d_K, *d_V, *d_dO, *d_D, *d_m, *d_l, *d_dQ, *d_dK, *d_dV;
        CHECK_CUDA(cudaMalloc(&d_Q, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dO, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_D, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_m, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_l, S * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dQ, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dK, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dV, S * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_dO, h_dO.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_D, h_D.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_m, h_m.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_l, h_l.data(), S * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dQ, 0, S * D * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dK, 0, S * D * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dV, 0, S * D * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (S + 15) / 16);
        fused_attn_bwd_joint_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_D, d_m, d_l, d_dQ, d_dK, d_dV, S, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dQ.data(), d_dQ, S * D * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_dK.data(), d_dK, S * D * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_dV.data(), d_dV, S * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S * D; ++i) {
            if (std::fabs(h_dQ[i] - h_ref_dQ[i]) > 1e-3f ||
                std::fabs(h_dK[i] - h_ref_dK[i]) > 1e-3f ||
                std::fabs(h_dV[i] - h_ref_dV[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 5: Fused Joint Attention Backward Kernel] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Fused Joint Attention Backward Kernel] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_dO); cudaFree(d_D);
        cudaFree(d_m); cudaFree(d_l); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
