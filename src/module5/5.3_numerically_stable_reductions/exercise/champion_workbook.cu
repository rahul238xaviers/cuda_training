#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void reportStatus(const std::string &name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 1: FlashAttention Online Rescaling of Output Accumulators
// O_new = O_old * exp(m_old - m_new) + P_tile * V_tile
// -----------------------------------------------------------------------------
__global__ void kernel_flash_online_rescale(
    float *O_acc, // Accumulator vector of length D
    const float *P_tile_times_V,
    float m_old,
    float m_new,
    int D
) {
    int tid = threadIdx.x;
    // TODO: if (tid < D) {
    //     float rescale = __expf(m_old - m_new);
    //     O_acc[tid] = O_acc[tid] * rescale + P_tile_times_V[tid];
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Vocab-Scale Cross-Entropy Loss (Vocab = 32,768)
// -----------------------------------------------------------------------------
__global__ void kernel_large_vocab_cross_entropy(
    const float *logits,
    int target_token,
    float *out_loss,
    int vocab_size
) {
    // Grid stride reduction to find max across 32,768 vocabulary
    // then grid stride to compute sum of exp(logit - max)
    // TODO: Implement large vocabulary stable loss
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Numerically Stable Log-Softmax (log_softmax(z_i) = z_i - max - log(sum(exp)))
// -----------------------------------------------------------------------------
__global__ void kernel_log_softmax_stable(const float *in, float *out, int n) {
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_log_sum;

    int tid = threadIdx.x;
    // TODO: Compute log_softmax in-place stably
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Top-2 Mixture-of-Experts (MoE) Gating Softmax
// -----------------------------------------------------------------------------
__global__ void kernel_moe_top2_softmax(
    const float *router_logits,
    int *expert_indices, // 2 indices
    float *expert_weights, // 2 weights normalized to sum to 1.0
    int num_experts
) {
    // 1. Find top-2 router logits
    // 2. Compute softmax only over the top-2 logits stably
    // TODO: Implement Top-2 MoE routing
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Fused Cross-Entropy Loss + Backward Pass
// Simultaneously computes scalar loss and outputs gradient tensor dZ
// -----------------------------------------------------------------------------
__global__ void kernel_fused_ce_fwd_bwd(
    const float *logits,
    int target,
    float *out_loss,
    float *grad_logits,
    int n
) {
    // Computes loss and sets grad_logits[i] = probs[i] - (i == target ? 1 : 0)
    // TODO: Implement fused forward + backward
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.3 Numerically Stable Reductions (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Flash Online Rescale
    {
        int D = 64;
        std::vector<float> h_O(D, 2.0f);
        std::vector<float> h_PV(D, 3.0f);
        float m_old = 5.0f, m_new = 6.0f; // diff is -1.0, exp(-1.0) = ~0.367879

        float *d_O, *d_PV;
        cudaMalloc(&d_O, D * sizeof(float));
        cudaMalloc(&d_PV, D * sizeof(float));
        cudaMemcpy(d_O, h_O.data(), D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_PV, h_PV.data(), D * sizeof(float), cudaMemcpyHostToDevice);

        kernel_flash_online_rescale<<<1, D>>>(d_O, d_PV, m_old, m_new, D);
        cudaDeviceSynchronize();

        cudaMemcpy(h_O.data(), d_O, D * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float expected = 2.0f * std::exp(-1.0f) + 3.0f;
        for (int i = 0; i < D; ++i) {
            if (std::fabs(h_O[i] - expected) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 1: FlashAttention Online Rescaling of Output Accumulators", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_O); cudaFree(d_PV);
    }

    // Test 2: Large Vocab Cross-Entropy
    {
        int vocab_size = 32768, target = 15000;
        std::vector<float> h_logits(vocab_size, 1.0f);
        h_logits[target] = 8.0f;
        float h_loss = 0.0f;

        float *d_logits, *d_loss;
        cudaMalloc(&d_logits, vocab_size * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);

        kernel_large_vocab_cross_entropy<<<8, 256>>>(d_logits, target, d_loss, vocab_size);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (!std::isnan(h_loss) && h_loss > 0.0f);
        reportStatus("Problem 2: Vocab-Scale Cross-Entropy Loss (Vocab = 32,768)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_loss);
    }

    // Test 3: Log-Softmax
    {
        int n = 256;
        std::vector<float> h_in(n, 5.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_log_softmax_stable<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // For uniform 256 elements, prob is 1/256, log_softmax is log(1/256) = -log(256) = -5.545
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (-std::log(256.0f))) > 1e-2f) { ok = false; break; }
        }
        reportStatus("Problem 3: Numerically Stable Log-Softmax Kernel", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Top-2 MoE Gating
    {
        int num_experts = 8;
        std::vector<float> h_logits = {1.0f, 5.0f, 0.5f, 2.0f, 6.0f, 0.1f, 3.0f, 0.2f};
        // Top 2 are index 4 (val 6.0) and index 1 (val 5.0)
        std::vector<int> h_idx(2, -1);
        std::vector<float> h_w(2, 0.0f);

        float *d_logits, *d_w;
        int *d_idx;
        cudaMalloc(&d_logits, num_experts * sizeof(float));
        cudaMalloc(&d_idx, 2 * sizeof(int));
        cudaMalloc(&d_w, 2 * sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), num_experts * sizeof(float), cudaMemcpyHostToDevice);

        kernel_moe_top2_softmax<<<1, 32>>>(d_logits, d_idx, d_w, num_experts);
        cudaDeviceSynchronize();

        cudaMemcpy(h_idx.data(), d_idx, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_w.data(), d_w, 2 * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = ((h_idx[0] == 4 && h_idx[1] == 1) || (h_idx[0] == 1 && h_idx[1] == 4));
        if (std::fabs((h_w[0] + h_w[1]) - 1.0f) > 1e-3f) ok = false;
        reportStatus("Problem 4: Top-2 Mixture-of-Experts (MoE) Gating Softmax", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_idx); cudaFree(d_w);
    }

    // Test 5: Fused CE Fwd + Bwd
    {
        int n = 256, target = 100;
        std::vector<float> h_logits(n, 2.0f);
        h_logits[target] = 5.0f;
        float h_loss = 0.0f;
        std::vector<float> h_grads(n, 0.0f);

        float *d_logits, *d_loss, *d_grads;
        cudaMalloc(&d_logits, n * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));
        cudaMalloc(&d_grads, n * sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_fused_ce_fwd_bwd<<<1, 256>>>(d_logits, target, d_loss, d_grads, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_grads.data(), d_grads, n * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = (!std::isnan(h_loss) && h_loss > 0.0f);
        // Target gradient must be negative (p - 1.0 < 0)
        if (h_grads[target] >= 0.0f) ok = false;
        reportStatus("Problem 5: Fused Cross-Entropy Loss + Backward in Single Pass", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_loss); cudaFree(d_grads);
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
