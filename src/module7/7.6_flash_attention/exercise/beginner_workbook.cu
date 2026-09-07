// ==============================================================================
// Module 7.6: FlashAttention Suite — Beginner Workbook
// ==============================================================================
// In this workbook, you will implement foundational algorithmic building blocks
// of FlashAttention (Tri Dao et al.):
// 1. Causal Masking & Attention Scaling Kernel (S_ij = (Q_i . K_j) / sqrt(d), mask j > i)
// 2. Online Softmax Running State Update (m_new, l_new)
// 3. Output Accumulator Rescaling Formula: O_new = O_old * exp(m_old - m_new) + Delta_O
// 4. Shared Memory Tiled QK^T Dot Product for Attention Tiles
// 5. Reference Multi-Head Attention Forward Kernel
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>

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
// Exercise 1: Causal Masking & Scaling Kernel
// Given raw dot product scores S [S_len x S_len]:
// For each element (i, j):
//   if j <= i: S[i, j] = S[i, j] * scale
//   if j >  i: S[i, j] = -INFINITY (-1e9f)
// Block: (16, 16), Grid: ((S_len + 15) / 16, (S_len + 15) / 16)
// ==============================================================================
__global__ void causal_mask_scale_kernel(float* scores, int S_len, float scale) {
    // TODO:
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y (query index i)
    //    col = blockIdx.x * blockDim.x + threadIdx.x (key index j)
    // 2. If row < S_len && col < S_len:
    //    int idx = row * S_len + col;
    //    if (col <= row) scores[idx] = scores[idx] * scale;
    //    else scores[idx] = -1e9f;
}

// ==============================================================================
// Exercise 2: Online Softmax Running State Update
// Given two successive blocks for row i:
// Old state: running max m_old, running sum l_old
// Block state: local max m_tile, local sum l_tile
// Computes new running state:
//   m_new = max(m_old, m_tile)
//   l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
// ==============================================================================
__global__ void online_softmax_state_kernel(const float* m_old, const float* l_old,
                                            const float* m_tile, const float* l_tile,
                                            float* m_new, float* l_new, int N) {
    // TODO:
    // 1. tid = blockIdx.x * blockDim.x + threadIdx.x
    // 2. If tid < N:
    //    float mo = m_old[tid];
    //    float lo = l_old[tid];
    //    float mt = m_tile[tid];
    //    float lt = l_tile[tid];
    //    float mn = fmaxf(mo, mt);
    //    float ln = lo * expf(mo - mn) + lt * expf(mt - mn);
    //    m_new[tid] = mn;
    //    l_new[tid] = ln;
}

// ==============================================================================
// Exercise 3: Output Accumulator Rescaling Formula
// In FlashAttention-2, when transitioning from tile j-1 to tile j:
// The unnormalized output vector O [N x D] is rescaled by exp(m_old - m_new),
// and the current tile contribution P_tile * V_tile is added:
// O_new[i, d] = O_old[i, d] * exp(m_old[i] - m_new[i]) + delta_O[i, d]
// Grid: ((D + 15) / 16, (N + 15) / 16), Block: (16, 16)
// ==============================================================================
__global__ void output_rescale_accum_kernel(const float* O_old, const float* delta_O,
                                            const float* m_old, const float* m_new,
                                            float* O_new, int N, int D) {
    // TODO:
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y (row in N)
    //    col = blockIdx.x * blockDim.x + threadIdx.x (col in D)
    // 2. If row < N && col < D:
    //    float alpha = expf(m_old[row] - m_new[row]);
    //    int idx = row * D + col;
    //    O_new[idx] = O_old[idx] * alpha + delta_O[idx];
}

// ==============================================================================
// Exercise 4: Tiled QK^T Dot Product in Shared Memory
// Computes attention score tile S_tile [Br x Bc] = (Q_tile [Br x D] * K_tile^T [D x Bc]) * scale
// Br = 16, Bc = 16, D = 32.
// Block: (16, 16), Grid: 1 block
// Shared memory: __shared__ float s_Q[16][32]; __shared__ float s_K[16][32];
// ==============================================================================
const int BR = 16;
const int BC = 16;
const int HD_DIM = 32;

__global__ void tiled_qk_dot_kernel(const float* Q, const float* K, float* S,
                                    float scale) {
    __shared__ float s_Q[BR][HD_DIM];
    __shared__ float s_K[BC][HD_DIM];

    // TODO:
    // 1. Thread (threadIdx.y, threadIdx.x) in [0..15, 0..15]
    // 2. Each thread loads 2 elements of Q and 2 elements of K into shared memory:
    //    s_Q[threadIdx.y][threadIdx.x] = Q[threadIdx.y * HD_DIM + threadIdx.x];
    //    s_Q[threadIdx.y][threadIdx.x + 16] = Q[threadIdx.y * HD_DIM + threadIdx.x + 16];
    //    s_K[threadIdx.y][threadIdx.x] = K[threadIdx.y * HD_DIM + threadIdx.x];
    //    s_K[threadIdx.y][threadIdx.x + 16] = K[threadIdx.y * HD_DIM + threadIdx.x + 16];
    // 3. __syncthreads()
    // 4. Accumulate dot product over d in [0, HD_DIM-1]:
    //    sum = sum_{d=0}^{HD_DIM-1} s_Q[threadIdx.y][d] * s_K[threadIdx.x][d]
    // 5. Write S[threadIdx.y * BC + threadIdx.x] = sum * scale;
}

// ==============================================================================
// Exercise 5: Standard Naive Causal Multi-Head Attention (Reference Baseline)
// Computes Out = Softmax(Mask(Q * K^T * scale)) * V for single head [S_len x D].
// Q, K, V are [S_len x D], Out is [S_len x D].
// ==============================================================================
__global__ void naive_causal_attn_kernel(const float* Q, const float* K, const float* V,
                                         float* Out, int S_len, int D, float scale) {
    // TODO:
    // Each thread handles 1 query token: row = blockIdx.x * blockDim.x + threadIdx.x.
    // If row < S_len:
    // 1. Compute dot products S_j = sum_{d=0}^{D-1} Q[row * D + d] * K[j * D + d] * scale for j <= row.
    // 2. Find max_val = max_{j <= row} S_j.
    // 3. Compute sum_exp = sum_{j <= row} exp(S_j - max_val).
    // 4. For each feature d in [0, D-1]:
    //    out_d = sum_{j <= row} (exp(S_j - max_val) / sum_exp) * V[j * D + d];
    //    Out[row * D + d] = out_d;
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Causal Masking & Scaling
    // --------------------------------------------------------------------------
    {
        int S_len = 8;
        float scale = 0.5f;
        std::vector<float> h_scores(S_len * S_len), h_out(S_len * S_len, 0.0f);
        for (int i = 0; i < S_len * S_len; ++i) h_scores[i] = static_cast<float>(i + 1);

        float* d_scores;
        CHECK_CUDA(cudaMalloc(&d_scores, S_len * S_len * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_scores, h_scores.data(), S_len * S_len * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((S_len + 15) / 16, (S_len + 15) / 16);
        causal_mask_scale_kernel<<<grid, block>>>(d_scores, S_len, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_scores, S_len * S_len * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S_len; ++i) {
            for (int j = 0; j < S_len; ++j) {
                float val = h_out[i * S_len + j];
                if (j <= i) {
                    float expected = (i * S_len + j + 1) * scale;
                    if (std::fabs(val - expected) > 1e-4f) ok = false;
                } else {
                    if (val > -1e8f) ok = false;
                }
            }
        }
        if (ok) {
            std::cout << "[Test 1: Causal Masking & Scaling] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Causal Masking & Scaling] FAILED" << std::endl;
        }

        cudaFree(d_scores);
    }

    // --------------------------------------------------------------------------
    // Test 2: Online Softmax State Update
    // --------------------------------------------------------------------------
    {
        int N = 16;
        std::vector<float> h_mo(N), h_lo(N), h_mt(N), h_lt(N);
        std::vector<float> h_mn(N, 0.0f), h_ln(N, 0.0f), h_mn_ref(N), h_ln_ref(N);

        for (int i = 0; i < N; ++i) {
            h_mo[i] = 1.0f + 0.1f * i;
            h_lo[i] = 2.0f + 0.2f * i;
            h_mt[i] = 1.5f + 0.05f * i;
            h_lt[i] = 3.0f + 0.15f * i;

            float mn = std::max(h_mo[i], h_mt[i]);
            float ln = h_lo[i] * std::exp(h_mo[i] - mn) + h_lt[i] * std::exp(h_mt[i] - mn);
            h_mn_ref[i] = mn;
            h_ln_ref[i] = ln;
        }

        float *d_mo, *d_lo, *d_mt, *d_lt, *d_mn, *d_ln;
        CHECK_CUDA(cudaMalloc(&d_mo, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_lo, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mt, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_lt, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mn, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ln, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_mo, h_mo.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo, h_lo.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mt, h_mt.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lt, h_lt.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_mn, 0, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_ln, 0, N * sizeof(float)));

        online_softmax_state_kernel<<<1, 16>>>(d_mo, d_lo, d_mt, d_lt, d_mn, d_ln, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_mn.data(), d_mn, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_ln.data(), d_ln, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_mn[i] - h_mn_ref[i]) > 1e-4f ||
                std::fabs(h_ln[i] - h_ln_ref[i]) > 1e-4f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Online Softmax State Update] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Online Softmax State Update] FAILED" << std::endl;
        }

        cudaFree(d_mo); cudaFree(d_lo); cudaFree(d_mt); cudaFree(d_lt); cudaFree(d_mn); cudaFree(d_ln);
    }

    // --------------------------------------------------------------------------
    // Test 3: Output Accumulator Rescaling
    // --------------------------------------------------------------------------
    {
        int N = 8, D = 16;
        std::vector<float> h_O_old(N * D), h_delta_O(N * D), h_mo(N), h_mn(N),
                           h_O_new(N * D, 0.0f), h_ref(N * D, 0.0f);
        for (int i = 0; i < N * D; ++i) {
            h_O_old[i]   = 0.5f * (i % 7);
            h_delta_O[i] = 0.3f * (i % 5);
        }
        for (int i = 0; i < N; ++i) {
            h_mo[i] = 2.0f + 0.1f * i;
            h_mn[i] = 2.5f + 0.1f * i;
        }

        for (int i = 0; i < N; ++i) {
            float alpha = std::exp(h_mo[i] - h_mn[i]);
            for (int d = 0; d < D; ++d) {
                h_ref[i * D + d] = h_O_old[i * D + d] * alpha + h_delta_O[i * D + d];
            }
        }

        float *d_O_old, *d_delta_O, *d_mo, *d_mn, *d_O_new;
        CHECK_CUDA(cudaMalloc(&d_O_old, N * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_delta_O, N * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mo, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mn, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_O_new, N * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_O_old, h_O_old.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_delta_O, h_delta_O.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mo, h_mo.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mn, h_mn.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_O_new, 0, N * D * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (N + 15) / 16);
        output_rescale_accum_kernel<<<grid, block>>>(d_O_old, d_delta_O, d_mo, d_mn, d_O_new, N, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_O_new.data(), d_O_new, N * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N * D; ++i) {
            if (std::fabs(h_O_new[i] - h_ref[i]) > 1e-4f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Output Rescaling Accumulator] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Output Rescaling Accumulator] FAILED" << std::endl;
        }

        cudaFree(d_O_old); cudaFree(d_delta_O); cudaFree(d_mo); cudaFree(d_mn); cudaFree(d_O_new);
    }

    // --------------------------------------------------------------------------
    // Test 4: Tiled QK^T Dot Product
    // --------------------------------------------------------------------------
    {
        std::vector<float> h_Q(BR * HD_DIM), h_K(BC * HD_DIM), h_S(BR * BC, 0.0f), h_ref(BR * BC, 0.0f);
        float scale = 0.125f;
        for (size_t i = 0; i < h_Q.size(); ++i) h_Q[i] = 0.2f * ((i % 11) + 1);
        for (size_t i = 0; i < h_K.size(); ++i) h_K[i] = 0.1f * ((i % 7) + 1);

        for (int r = 0; r < BR; ++r) {
            for (int c = 0; c < BC; ++c) {
                float sum = 0.0f;
                for (int d = 0; d < HD_DIM; ++d) {
                    sum += h_Q[r * HD_DIM + d] * h_K[c * HD_DIM + d];
                }
                h_ref[r * BC + c] = sum * scale;
            }
        }

        float *d_Q, *d_K, *d_S;
        CHECK_CUDA(cudaMalloc(&d_Q, h_Q.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_S, h_S.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_S, 0, h_S.size() * sizeof(float)));

        dim3 block(16, 16);
        tiled_qk_dot_kernel<<<1, block>>>(d_Q, d_K, d_S, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, h_S.size() * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (size_t i = 0; i < h_S.size(); ++i) {
            if (std::fabs(h_S[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: Tiled QK^T Dot Product] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Tiled QK^T Dot Product] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_S);
    }

    // --------------------------------------------------------------------------
    // Test 5: Naive Causal Multi-Head Attention Reference
    // --------------------------------------------------------------------------
    {
        int S_len = 16, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> h_Q(S_len * D), h_K(S_len * D), h_V(S_len * D),
                           h_Out(S_len * D, 0.0f), h_ref(S_len * D, 0.0f);

        for (int i = 0; i < S_len * D; ++i) {
            h_Q[i] = 0.1f * ((i % 5) + 1);
            h_K[i] = 0.1f * ((i % 7) + 1);
            h_V[i] = 0.2f * ((i % 11) + 1);
        }

        // Compute ground truth attention row by row
        for (int r = 0; r < S_len; ++r) {
            std::vector<float> scores(r + 1);
            float max_s = -1e30f;
            for (int j = 0; j <= r; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += h_Q[r * D + d] * h_K[j * D + d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j <= r; ++j) sum_exp += std::exp(scores[j] - max_s);
            for (int d = 0; d < D; ++d) {
                float v_acc = 0.0f;
                for (int j = 0; j <= r; ++j) {
                    float p = std::exp(scores[j] - max_s) / sum_exp;
                    v_acc += p * h_V[j * D + d];
                }
                h_ref[r * D + d] = v_acc;
            }
        }

        float *d_Q, *d_K, *d_V, *d_Out;
        CHECK_CUDA(cudaMalloc(&d_Q, S_len * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, S_len * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, S_len * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Out, S_len * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S_len * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), S_len * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), S_len * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_Out, 0, S_len * D * sizeof(float)));

        naive_causal_attn_kernel<<<1, S_len>>>(d_Q, d_K, d_V, d_Out, S_len, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_Out.data(), d_Out, S_len * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S_len * D; ++i) {
            if (std::fabs(h_Out[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 5: Naive Causal Attention Reference] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Naive Causal Attention Reference] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
