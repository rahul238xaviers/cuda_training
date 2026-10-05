// ==============================================================================
// Module 7.6: FlashAttention Suite — Intermediate Workbook
// ==============================================================================
// In this workbook, you will assemble the SRAM tiling and online softmax
// infrastructure required for FlashAttention-2 forward execution:
// 1. Warp-Level Row Maximum and Row Sum Reductions via __shfl_xor_sync
// 2. Single-Tile Online Softmax Updating
// 3. Causal Block Predication Classifier (SKIP, FULL, PARTIAL)
// 4. Shared Memory P * V Output Tile Accumulation
// 5. Single-Head FlashAttention-2 Forward Kernel with Causal Tiling
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

#define WARP_SIZE 32
#define MASK_ALL 0xffffffff

// ==============================================================================
// Exercise 1: Warp-Level Row Maximum & Row Sum Reductions
// 32 threads in a warp hold 1 float value each: val = input[threadIdx.x].
// Use __shfl_xor_sync across offsets {16, 8, 4, 2, 1} to compute:
// - warp_max: maximum across all 32 lanes
// - warp_sum: sum across all 32 lanes
// Store lane 0 results in out_max[blockIdx.x] and out_sum[blockIdx.x].
// ==============================================================================
__global__ void warp_reduce_max_sum_kernel(const float* input, float* out_max,
                                           float* out_sum, int N) {
    // TODO:
    // 1. int lane = threadIdx.x; // 0 to 31
    //    int gid = blockIdx.x * WARP_SIZE + lane;
    //    float v = (gid < N) ? input[gid] : -1e30f;
    //    float v_sum = (gid < N) ? input[gid] : 0.0f;
    // 2. Loop offset = 16; offset > 0; offset /= 2:
    //    float other = __shfl_xor_sync(MASK_ALL, v, offset);
    //    v = fmaxf(v, other);
    //    float other_s = __shfl_xor_sync(MASK_ALL, v_sum, offset);
    //    v_sum += other_s;
    // 3. if (lane == 0) { out_max[blockIdx.x] = v; out_sum[blockIdx.x] = v_sum; }
}

// ==============================================================================
// Exercise 2: Single-Tile Online Softmax Updating
// Given a vector of raw scores S_tile [16] for a single query token:
// Old running statistics: m_old, l_old
// Tasks:
// 1. Find local max: m_local = max_{c=0..15} S_tile[c]
// 2. New global max: m_new = max(m_old, m_local)
// 3. Compute local sum of unnormalized exponentials:
//    l_local = sum_{c=0..15} exp(S_tile[c] - m_new)
// 4. Update normalization denominator:
//    l_new = l_old * exp(m_old - m_new) + l_local
// Store updated (m_new, l_new) for each row i.
// ==============================================================================
__global__ void single_tile_online_softmax_kernel(const float* S_tiles,
                                                  const float* m_old, const float* l_old,
                                                  float* m_new, float* l_new,
                                                  int num_rows, int tile_cols) {
    // TODO:
    // Each thread processes 1 row: row = blockIdx.x * blockDim.x + threadIdx.x
    // If row < num_rows:
    // 1. const float* s_row = S_tiles + row * tile_cols;
    // 2. Find m_local = max over c in [0, tile_cols - 1]
    // 3. float mn = fmaxf(m_old[row], m_local);
    // 4. float l_local = sum_{c=0}^{tile_cols-1} expf(s_row[c] - mn);
    // 5. float ln = l_old[row] * expf(m_old[row] - mn) + l_local;
    // 6. m_new[row] = mn; l_new[row] = ln;
}

// ==============================================================================
// Exercise 3: Causal Block Predication Classifier
// Given query block index i (size Br) and key block index j (size Bc):
// Query range: [i * Br, (i + 1) * Br - 1]
// Key range:   [j * Bc, (j + 1) * Bc - 1]
// In causal attention, token q attends only to token k <= q.
// Output relationship code:
//   0 = FULL_COMPUTE (all k <= all q, i.e. (j + 1) * Bc - 1 <= i * Br)
//   1 = SKIP_BLOCK   (all k > all q, i.e. j * Bc > (i + 1) * Br - 1)
//   2 = PARTIAL_MASK (intersects causal diagonal)
// ==============================================================================
enum BlockCausalType {
    BLOCK_FULL = 0,
    BLOCK_SKIP = 1,
    BLOCK_PARTIAL = 2
};

__device__ __forceinline__ int classify_causal_block(int i, int j, int Br, int Bc) {
    int q_start = i * Br;
    int q_end   = (i + 1) * Br - 1;
    int k_start = j * Bc;
    int k_end   = (j + 1) * Bc - 1;

    if (k_end <= q_start) return BLOCK_FULL;
    if (k_start > q_end)   return BLOCK_SKIP;
    return BLOCK_PARTIAL;
}

__global__ void block_classify_kernel(int* out_classification, int num_q_blocks,
                                      int num_k_blocks, int Br, int Bc) {
    // TODO:
    // i = blockIdx.y * blockDim.y + threadIdx.y (q block)
    // j = blockIdx.x * blockDim.x + threadIdx.x (k block)
    // If i < num_q_blocks && j < num_k_blocks:
    //   out_classification[i * num_k_blocks + j] = classify_causal_block(i, j, Br, Bc);
}

// ==============================================================================
// Exercise 4: Shared Memory P * V Output Tile Accumulation
// P_tile is [Br x Bc] (attention probabilities), V_tile is [Bc x D].
// Accumulate delta_O [Br x D] = P_tile * V_tile.
// Br = 16, Bc = 16, D = 16.
// Thread block: (16, 16).
// ==============================================================================
__global__ void pv_tile_accum_kernel(const float* P_tile, const float* V_tile,
                                     float* delta_O, int Br, int Bc, int D) {
    __shared__ float s_P[16][16];
    __shared__ float s_V[16][16];

    // TODO:
    // 1. Thread (threadIdx.y, threadIdx.x)
    // 2. Load s_P and s_V into shared memory
    //    s_P[threadIdx.y][threadIdx.x] = P_tile[threadIdx.y * Bc + threadIdx.x];
    //    s_V[threadIdx.y][threadIdx.x] = V_tile[threadIdx.y * D + threadIdx.x];
    // 3. __syncthreads()
    // 4. Compute dot product for row = threadIdx.y, col = threadIdx.x:
    //    acc = sum_{c=0}^{Bc-1} s_P[threadIdx.y][c] * s_V[c][threadIdx.x]
    // 5. delta_O[threadIdx.y * D + threadIdx.x] = acc;
}

// ==============================================================================
// Exercise 5: Single-Head FlashAttention-2 Forward Kernel
// Full causal FlashAttention-2 forward for 1 head:
// Q [S x D], K [S x D], V [S x D], Out [S x D]
// S = 32, D = 16, Br = 16, Bc = 16.
// Outer loop over Q blocks i in [0, S/Br - 1].
// Inner loop over K/V blocks j in [0, S/Bc - 1].
// Use classify_causal_block to skip or mask.
// Maintain running max m [Br] and denominator l [Br] in registers or shared memory.
// Finally write normalized Out[row * D + d] = O_acc[row, d] / l[row].
// ==============================================================================
__global__ void flash_attn2_single_head_kernel(const float* Q, const float* K, const float* V,
                                               float* Out, int S, int D, float scale) {
    // TODO:
    // Each thread block processes one Q-block: i = blockIdx.x (size Br = 16)
    // Threads in block: (16, 16)
    // Implement tiled FlashAttention-2 algorithm:
    // 1. Shared memory for Q_tile [16 x 16], K_tile [16 x 16], V_tile [16 x 16]
    // 2. Per-thread accumulator O_acc[d] initialized to 0.0f
    //    Per-row running max m = -1e30f, running sum l = 0.0f
    // 3. Loop over j in [0, S/Bc - 1]:
    //    int block_type = classify_causal_block(i, j, 16, 16);
    //    if (block_type == BLOCK_SKIP) continue;
    //    Compute S_tile = (Q * K^T) * scale
    //    Apply causal mask if block_type == BLOCK_PARTIAL
    //    Compute local row max, update m_new and l_new
    //    Rescale previous O_acc: O_acc *= exp(m_old - m_new)
    //    Add P_tile * V to O_acc
    // 4. Normalize: Out = O_acc / l
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Warp-Level Max and Sum Reductions
    // --------------------------------------------------------------------------
    {
        std::vector<float> h_in(WARP_SIZE);
        for (int i = 0; i < WARP_SIZE; ++i) h_in[i] = 1.0f + 0.5f * (i % 7);
        float expected_max = -1e30f, expected_sum = 0.0f;
        for (int i = 0; i < WARP_SIZE; ++i) {
            if (h_in[i] > expected_max) expected_max = h_in[i];
            expected_sum += h_in[i];
        }

        float *d_in, *d_max, *d_sum;
        CHECK_CUDA(cudaMalloc(&d_in, WARP_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_max, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), WARP_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        warp_reduce_max_sum_kernel<<<1, WARP_SIZE>>>(d_in, d_max, d_sum, WARP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        float h_max = 0.0f, h_sum = 0.0f;
        CHECK_CUDA(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

        if (std::fabs(h_max - expected_max) < 1e-4f &&
            std::fabs(h_sum - expected_sum) < 1e-3f) {
            std::cout << "[Test 1: Warp-Level Max & Sum Reductions] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Warp-Level Max & Sum Reductions] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_max); cudaFree(d_sum);
    }

    // --------------------------------------------------------------------------
    // Test 2: Single-Tile Online Softmax Updating
    // --------------------------------------------------------------------------
    {
        int num_rows = 8, tile_cols = 16;
        std::vector<float> h_S(num_rows * tile_cols), h_mo(num_rows), h_lo(num_rows);
        std::vector<float> h_mn(num_rows, 0.0f), h_ln(num_rows, 0.0f);
        std::vector<float> h_mn_ref(num_rows), h_ln_ref(num_rows);

        for (int r = 0; r < num_rows; ++r) {
            h_mo[r] = 1.0f + 0.1f * r;
            h_lo[r] = 2.0f + 0.05f * r;
            float local_max = -1e30f;
            for (int c = 0; c < tile_cols; ++c) {
                float val = 0.2f * ((r * tile_cols + c) % 11);
                h_S[r * tile_cols + c] = val;
                if (val > local_max) local_max = val;
            }
            float mn = std::max(h_mo[r], local_max);
            float l_local = 0.0f;
            for (int c = 0; c < tile_cols; ++c) {
                l_local += std::exp(h_S[r * tile_cols + c] - mn);
            }
            float ln = h_lo[r] * std::exp(h_mo[r] - mn) + l_local;
            h_mn_ref[r] = mn;
            h_ln_ref[r] = ln;
        }

        float *d_S, *d_mo, *d_lo, *d_mn, *d_ln;
        CHECK_CUDA(cudaMalloc(&d_S, h_S.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mo, num_rows * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_lo, num_rows * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mn, num_rows * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ln, num_rows * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_S, h_S.data(), h_S.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mo, h_mo.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo, h_lo.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice));

        single_tile_online_softmax_kernel<<<1, num_rows>>>(d_S, d_mo, d_lo, d_mn, d_ln, num_rows, tile_cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_mn.data(), d_mn, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_ln.data(), d_ln, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < num_rows; ++r) {
            if (std::fabs(h_mn[r] - h_mn_ref[r]) > 1e-4f ||
                std::fabs(h_ln[r] - h_ln_ref[r]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Single-Tile Online Softmax] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Single-Tile Online Softmax] FAILED" << std::endl;
        }

        cudaFree(d_S); cudaFree(d_mo); cudaFree(d_lo); cudaFree(d_mn); cudaFree(d_ln);
    }

    // --------------------------------------------------------------------------
    // Test 3: Causal Block Predication Classifier
    // --------------------------------------------------------------------------
    {
        int Br = 16, Bc = 16;
        int num_q_blocks = 4, num_k_blocks = 4;
        std::vector<int> h_class(num_q_blocks * num_k_blocks, -1);

        int* d_class;
        CHECK_CUDA(cudaMalloc(&d_class, num_q_blocks * num_k_blocks * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_class, -1, num_q_blocks * num_k_blocks * sizeof(int)));

        dim3 block(4, 4);
        block_classify_kernel<<<1, block>>>(d_class, num_q_blocks, num_k_blocks, Br, Bc);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_class.data(), d_class, num_q_blocks * num_k_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < num_q_blocks; ++i) {
            for (int j = 0; j < num_k_blocks; ++j) {
                int expected;
                if (j < i) expected = BLOCK_FULL; // 0
                else if (j == i) expected = BLOCK_PARTIAL; // 2
                else expected = BLOCK_SKIP; // 1
                if (h_class[i * num_k_blocks + j] != expected) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            std::cout << "[Test 3: Causal Block Predication] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Causal Block Predication] FAILED" << std::endl;
        }

        cudaFree(d_class);
    }

    // --------------------------------------------------------------------------
    // Test 4: Shared Memory P * V Output Tile Accumulation
    // --------------------------------------------------------------------------
    {
        int Br = 16, Bc = 16, D = 16;
        std::vector<float> h_P(Br * Bc), h_V(Bc * D), h_dO(Br * D, 0.0f), h_ref(Br * D, 0.0f);
        for (int i = 0; i < Br * Bc; ++i) h_P[i] = 0.05f * ((i % 7) + 1);
        for (int i = 0; i < Bc * D; ++i)  h_V[i] = 0.1f  * ((i % 11) + 1);

        for (int r = 0; r < Br; ++r) {
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int c = 0; c < Bc; ++c) {
                    sum += h_P[r * Bc + c] * h_V[c * D + d];
                }
                h_ref[r * D + d] = sum;
            }
        }

        float *d_P, *d_V, *d_dO;
        CHECK_CUDA(cudaMalloc(&d_P, Br * Bc * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, Bc * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dO, Br * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_P, h_P.data(), Br * Bc * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), Bc * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dO, 0, Br * D * sizeof(float)));

        dim3 block(16, 16);
        pv_tile_accum_kernel<<<1, block>>>(d_P, d_V, d_dO, Br, Bc, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_dO.data(), d_dO, Br * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (size_t i = 0; i < h_dO.size(); ++i) {
            if (std::fabs(h_dO[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: P * V Tile Accumulation] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: P * V Tile Accumulation] FAILED" << std::endl;
        }

        cudaFree(d_P); cudaFree(d_V); cudaFree(d_dO);
    }

    // --------------------------------------------------------------------------
    // Test 5: Single-Head FlashAttention-2 Forward Kernel
    // --------------------------------------------------------------------------
    {
        int S = 32, D = 16;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> h_Q(S * D), h_K(S * D), h_V(S * D),
                           h_Out(S * D, 0.0f), h_ref(S * D, 0.0f);

        for (int i = 0; i < S * D; ++i) {
            h_Q[i] = 0.1f * ((i % 5) + 1);
            h_K[i] = 0.1f * ((i % 7) + 1);
            h_V[i] = 0.2f * ((i % 11) + 1);
        }

        // Reference standard causal attention
        for (int r = 0; r < S; ++r) {
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
                float acc = 0.0f;
                for (int j = 0; j <= r; ++j) {
                    float p = std::exp(scores[j] - max_s) / sum_exp;
                    acc += p * h_V[j * D + d];
                }
                h_ref[r * D + d] = acc;
            }
        }

        float *d_Q, *d_K, *d_V, *d_Out;
        CHECK_CUDA(cudaMalloc(&d_Q, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, S * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Out, S * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), S * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_Out, 0, S * D * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid(S / 16); // 2 Q-blocks
        flash_attn2_single_head_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_Out, S, D, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_Out.data(), d_Out, S * D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < S * D; ++i) {
            if (std::fabs(h_Out[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 5: FlashAttention-2 Single Head Forward] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: FlashAttention-2 Single Head Forward] FAILED" << std::endl;
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
