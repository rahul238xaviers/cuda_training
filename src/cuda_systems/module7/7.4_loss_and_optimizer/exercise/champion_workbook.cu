// ============================================================================
// Module 7.4: Loss & Optimizer - Champion Workbook
// Kernels Covered:
//   - full_cross_entropy: Complete fused loss + grad kernel matching cross_entropy.metal
//   - vectorized_adamw_step: 128-bit float4 vectorized AdamW parameter update
//   - online_logsumexp_ce: Numerically stable online running LogSumExp
//   - struct_param_adamw: AdamW with AdamWParams struct matching adamw_step.metal
//   - hierarchical_loss_reduction: Warp-reduced atomic global loss aggregation
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

struct AdamWParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
    unsigned int n;
};

// ============================================================================
// Exercise 1: Full Fused Cross Entropy Kernel
// Matching cross_entropy.metal
// Inputs: logits in BF16, targets, loss_out (atomic), grad_logits (float)
// ============================================================================
__global__ void full_cross_entropy_kernel(const __nv_bfloat16* logits,
                                          const unsigned int* targets,
                                          float* loss_out,
                                          float* grad_logits,
                                          int vocab_size,
                                          int total_tokens) {
    int t = blockIdx.x;
    if (t >= total_tokens) return;

    __shared__ float shared_mem[256];
    int tid = threadIdx.x;
    const __nv_bfloat16* token_logits = logits + t * vocab_size;
    float* token_grads = grad_logits + t * vocab_size;
    unsigned int target_id = targets[t];

    // TODO:
    // 1. Pass 1: Local max over vocabulary, reduce across block.
    // 2. Pass 2: Local sum of exponentials, reduce across block.
    // 3. Thread 0: Compute token cross entropy loss, atomicAdd to loss_out.
    // 4. Pass 3: Loop over vocabulary and write:
    //    token_grads[v] = (prob - is_target) / (float)total_tokens;
}

// ============================================================================
// Exercise 2: Vectorized 128-bit AdamW Step (float4)
// num_vec = n / 4. Process 4 weights, gradients, moments per thread.
// ============================================================================
__global__ void vectorized_adamw_step_kernel(float4* param, const float4* grad,
                                             float4* m, float4* v,
                                             float lr, float beta1, float beta2,
                                             float eps, float weight_decay,
                                             float bc1, float bc2, int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // TODO:
        // Load float4 g = grad[idx], p = param[idx], m_vec = m[idx], v_vec = v[idx];
        // For .x, .y, .z, .w:
        //   m = beta1 * m + (1 - beta1) * g
        //   v = beta2 * v + (1 - beta2) * g * g
        //   if (weight_decay > 0) p -= lr * weight_decay * p
        //   p -= lr * (m / bc1) / (sqrtf(v / bc2) + eps)
        // Store updated p, m, v.
    }
}

// ============================================================================
// Exercise 3: Single-Pass Online LogSumExp Cross Entropy
// Uses online running max & exp-sum update to compute loss in a single pass.
// ============================================================================
__global__ void online_logsumexp_ce_kernel(const float* logits,
                                           const unsigned int* targets,
                                           float* loss_out,
                                           int vocab_size,
                                           int total_tokens) {
    int t = blockIdx.x;
    if (t >= total_tokens) return;

    // TODO:
    // Maintain running max m and running sum d:
    // for each chunk:
    //   m_new = max(m, chunk_max);
    //   d = d * exp(m - m_new) + chunk_sum * exp(chunk_max - m_new);
    //   m = m_new;
    // Final loss = (log(d) + m) - target_logit;
    // atomicAdd(loss_out, loss / total_tokens);
}

// ============================================================================
// Exercise 4: AdamW with AdamWParams Struct
// Matching adamw_step.metal
// ============================================================================
__global__ void struct_param_adamw_kernel(__nv_bfloat16* param,
                                          const __nv_bfloat16* grad,
                                          float* m, float* v,
                                          AdamWParams p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p.n) {
        // TODO:
        // float g = __bfloat162float(grad[idx]);
        // m[idx] = p.beta1 * m[idx] + (1.0f - p.beta1) * g;
        // v[idx] = p.beta2 * v[idx] + (1.0f - p.beta2) * g * g;
        // float val = __bfloat162float(param[idx]);
        // if (p.weight_decay > 0.0f) val -= p.lr * p.weight_decay * val;
        // float m_hat = m[idx] / p.bias_correction1;
        // float v_hat = v[idx] / p.bias_correction2;
        // val -= p.lr * m_hat / (sqrtf(v_hat) + p.eps);
        // param[idx] = __float2bfloat16(val);
    }
}

// ============================================================================
// Exercise 5: Hierarchical Warp-Reduced Loss Aggregation
// Intra-warp shuffle reduction before single atomicAdd per warp to reduce contention.
// ============================================================================
__global__ void hierarchical_loss_reduction_kernel(const float* token_losses,
                                                   float* global_loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    // TODO:
    // float val = (idx < n) ? token_losses[idx] : 0.0f;
    // for (int offset = 16; offset > 0; offset /= 2) {
    //     val += __shfl_down_sync(FULL_MASK, val, offset);
    // }
    // if (lane == 0) atomicAdd(global_loss, val);
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    const int total_tokens = 2;
    const int vocab_size = 512;
    const int total_elem = total_tokens * vocab_size;

    std::vector<__nv_bfloat16> h_bf_logits(total_elem, __float2bfloat16(0.0f));
    std::vector<unsigned int> h_targets = {5, 100};
    h_bf_logits[0 * vocab_size + 5] = __float2bfloat16(2.5f);
    h_bf_logits[1 * vocab_size + 100] = __float2bfloat16(2.5f);

    // --- Test 1: Full Fused Cross Entropy Kernel ---
    {
        __nv_bfloat16* d_logits;
        unsigned int* d_targets;
        float *d_loss, *d_grads;
        CUDA_CHECK(cudaMalloc(&d_logits, total_elem * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grads, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_bf_logits.data(), total_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grads, 0, total_elem * sizeof(float)));

        full_cross_entropy_kernel<<<total_tokens, 256>>>(d_logits, d_targets, d_loss, d_grads, vocab_size, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> h_grads(total_elem);
        CUDA_CHECK(cudaMemcpy(h_grads.data(), d_grads, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        float exp_loss = -std::log(std::exp(2.5f) / (std::exp(2.5f) + 511.0f));
        bool ok = true;
        if (std::abs(h_loss - exp_loss) > 0.05f * exp_loss || h_loss == 0.0f) ok = false;
        if (h_grads[5] >= 0.0f) ok = false; // Target gradient must be negative (prob - 1) / T

        if (ok) {
            std::cout << "Test 1 Passed: Full fused cross-entropy kernel." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Full fused cross-entropy kernel." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss); cudaFree(d_grads);
    }

    // --- Test 2: Vectorized 128-bit AdamW ---
    {
        const int N = 256;
        const int num_vec = N / 4;
        std::vector<float> h_p(N, 1.0f), h_g(N, 0.1f), h_m(N, 0.0f), h_v(N, 0.0f);
        float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        float bc1 = 0.1f, bc2 = 0.001f;

        float4 *d_p, *d_g, *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_p, num_vec * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_g, num_vec * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_m, num_vec * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_v, num_vec * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        vectorized_adamw_step_kernel<<<1, num_vec>>>(d_p, d_g, d_m, d_v, lr, beta1, beta2, eps, wd, bc1, bc2, num_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_m[i] - 0.01f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_p[i] - 0.9899f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_m[0] != 0.0f) {
            std::cout << "Test 2 Passed: 128-bit vectorized AdamW step." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: 128-bit vectorized AdamW step." << std::endl;
        }

        cudaFree(d_p); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    }

    // --- Test 3: Online LogSumExp Cross Entropy ---
    {
        std::vector<float> h_logits(total_elem, 0.0f);
        h_logits[0 * vocab_size + 5] = 2.5f;
        h_logits[1 * vocab_size + 100] = 2.5f;

        float *d_logits, *d_loss;
        unsigned int* d_targets;
        CUDA_CHECK(cudaMalloc(&d_logits, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        online_logsumexp_ce_kernel<<<total_tokens, 256>>>(d_logits, d_targets, d_loss, vocab_size, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

        float exp_loss = -std::log(std::exp(2.5f) / (std::exp(2.5f) + 511.0f));

        if (std::abs(h_loss - exp_loss) < 1e-3f && h_loss != 0.0f) {
            std::cout << "Test 3 Passed: Online LogSumExp cross entropy." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Online LogSumExp cross entropy (got " << h_loss << ", exp " << exp_loss << ")." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
    }

    // --- Test 4: AdamW with Struct Params ---
    {
        const int N = 256;
        std::vector<__nv_bfloat16> h_p(N, __float2bfloat16(1.0f));
        std::vector<__nv_bfloat16> h_g(N, __float2bfloat16(0.1f));
        std::vector<float> h_m(N, 0.0f), h_v(N, 0.0f);

        AdamWParams p;
        p.lr = 0.01f;
        p.beta1 = 0.9f;
        p.beta2 = 0.999f;
        p.eps = 1e-8f;
        p.weight_decay = 0.01f;
        p.bias_correction1 = 0.1f;
        p.bias_correction2 = 0.001f;
        p.n = N;

        __nv_bfloat16 *d_p, *d_g;
        float *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        struct_param_adamw_kernel<<<1, 256>>>(d_p, d_g, d_m, d_v, p);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_m[i] - 0.01f) > 1e-4f) { ok = false; break; }
            if (std::abs(__bfloat162float(h_p[i]) - 0.9899f) > 0.02f) { ok = false; break; }
        }
        if (ok && h_m[0] != 0.0f) {
            std::cout << "Test 4 Passed: AdamW with AdamWParams struct." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: AdamW with AdamWParams struct." << std::endl;
        }

        cudaFree(d_p); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    }

    // --- Test 5: Hierarchical Warp-Reduced Loss ---
    {
        const int N = 256;
        std::vector<float> h_losses(N, 0.5f);

        float *d_losses, *d_global;
        CUDA_CHECK(cudaMalloc(&d_losses, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_global, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_losses, h_losses.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_global, 0, sizeof(float)));

        hierarchical_loss_reduction_kernel<<<1, 256>>>(d_losses, d_global, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_global = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_global, d_global, sizeof(float), cudaMemcpyDeviceToHost));

        if (std::abs(h_global - 128.0f) < 1e-3f && h_global != 0.0f) {
            std::cout << "Test 5 Passed: Hierarchical warp-reduced loss." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Hierarchical warp-reduced loss." << std::endl;
        }

        cudaFree(d_losses); cudaFree(d_global);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
