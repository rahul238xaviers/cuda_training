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

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// -----------------------------------------------------------------------------
// PROBLEM 1: Online Softmax Rescaling between Two Sequential Chunks
// Update running max m and running denominator d:
// m_new = max(m_old, m_curr);
// d_new = d_old * exp(m_old - m_new) + d_curr * exp(m_curr - m_new);
// -----------------------------------------------------------------------------
__global__ void kernel_online_softmax_rescale(
    float m_old, float d_old,
    float m_curr, float d_curr,
    float *out_m_new, float *out_d_new
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // TODO: Compute m_new and d_new using the online softmax formula
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Batched Cross-Entropy Loss with Target Gathering
// -----------------------------------------------------------------------------
__global__ void kernel_batched_cross_entropy(
    const float *logits,
    const int *targets,
    float *losses,
    int vocab_size
) {
    // 1 block per token position
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    const float *token_logits = logits + token_idx * vocab_size;
    int target = targets[token_idx];

    // TODO: 1. Find max over vocab_size
    // TODO: 2. Sum exp(z - s_max)
    // TODO: 3. Thread 0 writes losses[token_idx] = -token_logits[target] + s_max + logf(s_sum_exp)
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Softmax Cross-Entropy Backward Pass
// dL/dz_i = p_i - (i == target ? 1.0f : 0.0f)
// -----------------------------------------------------------------------------
__global__ void kernel_cross_entropy_backward(
    const float *probs,
    const int *targets,
    float *grad_logits,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int target = targets[token_idx];

    // TODO: For each class c: grad_logits[token_idx * vocab_size + c] = p_c - (c == target ? 1.0f : 0.0f)
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Softmax with Causal Masking (Attention Masking)
// Mask out keys where col > row by setting logit to -10000.0f
// -----------------------------------------------------------------------------
__global__ void kernel_causal_softmax(const float *scores, float *probs, int seq_len) {
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;

    int row = blockIdx.x; // Query position
    int col = threadIdx.x; // Key position

    float s = (col < seq_len) ? scores[row * seq_len + col] : -1e30f;
    // TODO: Causal masking: if (col > row) s = -10000.0f;
    // TODO: Compute stable softmax across valid keys
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Label Smoothing Cross-Entropy Loss
// Targets have (1 - eps) on target class and eps / vocab_size on other classes
// -----------------------------------------------------------------------------
__global__ void kernel_label_smoothing_loss(
    const float *logits,
    int target,
    float *loss,
    float eps_smooth,
    int vocab_size
) {
    // TODO: Compute stable LSE, and compute loss with label smoothing
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.3 Numerically Stable Reductions (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Online Rescale
    {
        float m_old = 10.0f, d_old = 2.0f;
        float m_curr = 12.0f, d_curr = 3.0f;
        float *d_m, *d_d;
        cudaMalloc(&d_m, sizeof(float));
        cudaMalloc(&d_d, sizeof(float));

        kernel_online_softmax_rescale<<<1, 1>>>(m_old, d_old, m_curr, d_curr, d_m, d_d);
        cudaDeviceSynchronize();

        float h_m = 0.0f, h_d = 0.0f;
        cudaMemcpy(&h_m, d_m, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_d, d_d, sizeof(float), cudaMemcpyDeviceToHost);

        // m_new = max(10, 12) = 12.0
        // d_new = 2.0 * exp(10 - 12) + 3.0 * exp(12 - 12) = 2.0 * e^-2 + 3.0 = 0.27067 + 3.0 = 3.27067
        bool ok = (h_m == 12.0f && std::fabs(h_d - (2.0f * std::exp(-2.0f) + 3.0f)) < 1e-3f);
        reportStatus("Problem 1: Online Softmax Rescaling between Sequential Chunks", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_m); cudaFree(d_d);
    }

    // Test 2: Batched Cross-Entropy
    {
        int num_tokens = 4, vocab_size = 256;
        std::vector<float> h_logits(num_tokens * vocab_size, 2.0f);
        std::vector<int> h_targets = {5, 10, 15, 20};
        for (int t = 0; t < num_tokens; ++t) {
            h_logits[t * vocab_size + h_targets[t]] = 6.0f; // Target has logit 6.0
        }
        std::vector<float> h_losses(num_tokens, 0.0f);

        float *d_logits, *d_losses;
        int *d_targets;
        cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
        cudaMalloc(&d_targets, num_tokens * sizeof(int));
        cudaMalloc(&d_losses, num_tokens * sizeof(float));

        cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

        kernel_batched_cross_entropy<<<num_tokens, 256>>>(d_logits, d_targets, d_losses, vocab_size);
        cudaDeviceSynchronize();

        cudaMemcpy(h_losses.data(), d_losses, num_tokens * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int t = 0; t < num_tokens; ++t) {
            if (std::isnan(h_losses[t]) || h_losses[t] <= 0.0f) { ok = false; break; }
        }
        reportStatus("Problem 2: Batched Cross-Entropy Loss with Target Gathering", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_losses);
    }

    // Test 3: Cross-Entropy Backward
    {
        int num_tokens = 2, vocab_size = 4;
        std::vector<float> h_probs = {
            0.1f, 0.7f, 0.1f, 0.1f, // token 0: target is 1
            0.2f, 0.2f, 0.5f, 0.1f  // token 1: target is 2
        };
        std::vector<int> h_targets = {1, 2};
        std::vector<float> h_grads(num_tokens * vocab_size, 0.0f);

        float *d_probs, *d_grads;
        int *d_targets;
        cudaMalloc(&d_probs, h_probs.size() * sizeof(float));
        cudaMalloc(&d_targets, num_tokens * sizeof(int));
        cudaMalloc(&d_grads, h_grads.size() * sizeof(float));

        cudaMemcpy(d_probs, h_probs.data(), h_probs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, h_targets.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

        kernel_cross_entropy_backward<<<num_tokens, vocab_size>>>(d_probs, d_targets, d_grads, vocab_size);
        cudaDeviceSynchronize();

        cudaMemcpy(h_grads.data(), d_grads, h_grads.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // For token 0 target 1: grad[1] = 0.7 - 1.0 = -0.3; grad[0] = 0.1
        if (std::fabs(h_grads[1] - (-0.3f)) > 1e-4f) ok = false;
        if (std::fabs(h_grads[0] - 0.1f) > 1e-4f) ok = false;
        reportStatus("Problem 3: Softmax Cross-Entropy Backward Pass (p - target)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_probs); cudaFree(d_targets); cudaFree(d_grads);
    }

    // Test 4: Causal Softmax
    {
        int seq_len = 8;
        std::vector<float> h_scores(seq_len * seq_len, 1.0f);
        std::vector<float> h_probs(seq_len * seq_len, 0.0f);

        float *d_scores, *d_probs;
        cudaMalloc(&d_scores, h_scores.size() * sizeof(float));
        cudaMalloc(&d_probs, h_probs.size() * sizeof(float));
        cudaMemcpy(d_scores, h_scores.data(), h_scores.size() * sizeof(float), cudaMemcpyHostToDevice);

        kernel_causal_softmax<<<seq_len, 32>>>(d_scores, d_probs, seq_len);
        cudaDeviceSynchronize();

        cudaMemcpy(h_probs.data(), d_probs, h_probs.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // Row 0 can only attend to col 0: prob[0, 0] must be 1.0; prob[0, 1] must be ~0.0
        if (std::fabs(h_probs[0 * seq_len + 0] - 1.0f) > 1e-3f) ok = false;
        if (h_probs[0 * seq_len + 1] > 1e-3f) ok = false;
        reportStatus("Problem 4: Softmax with Causal Masking (Autoregressive Attention)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_scores); cudaFree(d_probs);
    }

    // Test 5: Label Smoothing
    {
        int vocab_size = 256, target = 10;
        std::vector<float> h_logits(vocab_size, 1.0f);
        h_logits[target] = 5.0f;
        float h_loss = 0.0f;

        float *d_logits, *d_loss;
        cudaMalloc(&d_logits, vocab_size * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);

        kernel_label_smoothing_loss<<<1, 256>>>(d_logits, target, d_loss, 0.1f, vocab_size);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (!std::isnan(h_loss) && h_loss > 0.0f);
        reportStatus("Problem 5: Label Smoothing Cross-Entropy Loss", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_loss);
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
