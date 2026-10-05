// ============================================================================
// Module 7.4: Loss & Optimizer - Intermediate Workbook
// Kernels Covered:
//   - block_cross_entropy_loss: 3-pass block reduction for large vocab
//   - block_cross_entropy_grad: Full logits gradient generation
//   - compute_loss_only: Minimal loss without gradient writes (compute_loss.metal)
//   - adamw_mixed_precision: BF16 parameters with FP32 optimizer states
//   - adamw_clipped_step: AdamW update with elementwise gradient thresholding
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

// Helper tree reduction for block max
__device__ inline float block_reduce_max(float val, float* shared_mem, int tid, int bdim) {
    shared_mem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        __syncthreads();
    }
    return shared_mem[0];
}

// Helper tree reduction for block sum
__device__ inline float block_reduce_sum(float val, float* shared_mem, int tid, int bdim) {
    shared_mem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }
    return shared_mem[0];
}

// ============================================================================
// Exercise 1: Block-Level Cross Entropy Loss (V >= 256)
// 1 Block per token. gridDim.x = total_tokens. blockDim.x = 256.
// ============================================================================
__global__ void block_cross_entropy_loss_kernel(const float* logits,
                                                const unsigned int* targets,
                                                float* loss_out,
                                                int vocab_size,
                                                int total_tokens) {
    int t = blockIdx.x;
    if (t >= total_tokens) return;

    __shared__ float s_mem[256];
    int tid = threadIdx.x;
    const float* token_logits = logits + t * vocab_size;
    unsigned int target_id = targets[t];

    // TODO:
    // 1. Pass 1: local_max across vocab (v = tid; v < vocab_size; v += 256)
    //    float gmax = block_reduce_max(local_max, s_mem, tid, 256);
    // 2. Pass 2: local_sum_exp across vocab
    //    float gsum = block_reduce_sum(local_sum_exp, s_mem, tid, 256);
    // 3. Thread 0 computes:
    //    float target_logit = token_logits[target_id];
    //    float prob = expf(target_logit - gmax) / gsum;
    //    float token_loss = -logf(fmaxf(prob, 1e-15f)) / (float)total_tokens;
    //    atomicAdd(loss_out, token_loss);
}

// ============================================================================
// Exercise 2: Block-Level Logits Gradient Generation
// Pass 3 computes and writes: grad_logits[t * V + v] = (prob - is_target) / total_tokens
// ============================================================================
__global__ void block_cross_entropy_grad_kernel(const float* logits,
                                                const unsigned int* targets,
                                                float* grad_logits,
                                                int vocab_size,
                                                int total_tokens) {
    int t = blockIdx.x;
    if (t >= total_tokens) return;

    __shared__ float s_mem[256];
    __shared__ float s_gmax;
    __shared__ float s_gsum;

    int tid = threadIdx.x;
    const float* token_logits = logits + t * vocab_size;
    float* token_grads = grad_logits + t * vocab_size;
    unsigned int target_id = targets[t];

    // TODO:
    // 1. Pass 1 & 2: reduce max and sum. Store to s_gmax, s_gsum.
    // 2. Pass 3: loop over vocab and write:
    //    float prob = expf(token_logits[v] - s_gmax) / s_gsum;
    //    float is_target = (v == target_id) ? 1.0f : 0.0f;
    //    token_grads[v] = (prob - is_target) / (float)total_tokens;
}

// ============================================================================
// Exercise 3: Minimal Compute Loss Kernel (Matching compute_loss.metal)
// Input logits in BF16 (__nv_bfloat16). Computes loss only.
// ============================================================================
__global__ void compute_loss_only_kernel(const __nv_bfloat16* logits,
                                         const unsigned int* targets,
                                         float* loss_out,
                                         int vocab_size,
                                         int total_tokens) {
    int t = blockIdx.x;
    if (t >= total_tokens) return;

    __shared__ float s_mem[256];
    int tid = threadIdx.x;
    const __nv_bfloat16* token_logits = logits + t * vocab_size;
    unsigned int target_id = targets[t];

    // TODO:
    // Perform 2-pass reduction using __bfloat162float conversions and atomicAdd to loss_out.
}

// ============================================================================
// Exercise 4: Mixed-Precision AdamW Step (BF16 Params, FP32 Moments)
// param[i] and grad[i] are __nv_bfloat16.
// m[i] and v[i] are float.
// ============================================================================
__global__ void adamw_mixed_precision_kernel(__nv_bfloat16* param,
                                             const __nv_bfloat16* grad,
                                             float* m, float* v,
                                             float lr, float beta1, float beta2,
                                             float eps, float weight_decay,
                                             float bc1, float bc2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // float g = __bfloat162float(grad[idx]);
        // float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        // float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;
        // m[idx] = m_val;
        // v[idx] = v_val;
        // float p = __bfloat162float(param[idx]);
        // if (weight_decay > 0.0f) p -= lr * weight_decay * p;
        // float m_hat = m_val / bc1;
        // float v_hat = v_val / bc2;
        // p -= lr * m_hat / (sqrtf(v_hat) + eps);
        // param[idx] = __float2bfloat16(p);
    }
}

// ============================================================================
// Exercise 5: AdamW Step with Gradient Threshold Clipping
// If fabsf(g) > clip_val, g = copysignf(clip_val, g).
// ============================================================================
__global__ void adamw_clipped_step_kernel(float* param, const float* grad,
                                          float* m, float* v,
                                          float lr, float beta1, float beta2,
                                          float eps, float weight_decay,
                                          float bc1, float bc2,
                                          float clip_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // float g = grad[idx];
        // if (fabsf(g) > clip_val) g = copysignf(clip_val, g);
        // Apply standard AdamW equations with clipped g.
    }
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

    std::vector<float> h_logits(total_elem, 0.0f);
    std::vector<unsigned int> h_targets = {10, 200};
    h_logits[0 * vocab_size + 10] = 3.0f;  // token 0 target
    h_logits[1 * vocab_size + 200] = 3.0f; // token 1 target

    // --- Test 1: Block Cross Entropy Loss ---
    {
        float *d_logits, *d_loss;
        unsigned int* d_targets;
        CUDA_CHECK(cudaMalloc(&d_logits, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        block_cross_entropy_loss_kernel<<<total_tokens, 256>>>(d_logits, d_targets, d_loss, vocab_size, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

        // exp(3) ~ 20.0855. 511 others have exp(0) = 1.0 -> sum_exp = 20.0855 + 511 = 531.0855
        // prob = 20.0855 / 531.0855 = 0.0378198
        // per token loss = -log(0.0378198) = 3.2749
        // total loss averaged across 2 tokens = 3.2749
        float exp_loss = -std::log(std::exp(3.0f) / (std::exp(3.0f) + 511.0f));

        if (std::abs(h_loss - exp_loss) < 1e-3f && h_loss != 0.0f) {
            std::cout << "Test 1 Passed: Block cross-entropy loss." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Block cross-entropy loss (got " << h_loss << ", exp " << exp_loss << ")." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
    }

    // --- Test 2: Block Logits Gradient Generation ---
    {
        float *d_logits, *d_grads;
        unsigned int* d_targets;
        CUDA_CHECK(cudaMalloc(&d_logits, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_grads, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_grads, 0, total_elem * sizeof(float)));

        block_cross_entropy_grad_kernel<<<total_tokens, 256>>>(d_logits, d_targets, d_grads, vocab_size, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_grads(total_elem);
        CUDA_CHECK(cudaMemcpy(h_grads.data(), d_grads, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        float prob_target = std::exp(3.0f) / (std::exp(3.0f) + 511.0f);
        float exp_target_grad = (prob_target - 1.0f) / static_cast<float>(total_tokens);

        if (std::abs(h_grads[10] - exp_target_grad) < 1e-4f && h_grads[10] != 0.0f) {
            std::cout << "Test 2 Passed: Block logits gradient generation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Block logits gradient generation." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_grads);
    }

    // --- Test 3: Minimal Compute Loss Kernel ---
    {
        std::vector<__nv_bfloat16> h_bf_logits(total_elem);
        for (int i = 0; i < total_elem; ++i) h_bf_logits[i] = __float2bfloat16(h_logits[i]);

        __nv_bfloat16* d_logits;
        unsigned int* d_targets;
        float* d_loss;
        CUDA_CHECK(cudaMalloc(&d_logits, total_elem * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_bf_logits.data(), total_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        compute_loss_only_kernel<<<total_tokens, 256>>>(d_logits, d_targets, d_loss, vocab_size, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

        float exp_loss = -std::log(std::exp(3.0f) / (std::exp(3.0f) + 511.0f));

        if (std::abs(h_loss - exp_loss) < 0.05f * exp_loss && h_loss != 0.0f) {
            std::cout << "Test 3 Passed: Minimal compute loss kernel." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Minimal compute loss kernel (got " << h_loss << ", exp " << exp_loss << ")." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
    }

    // --- Test 4: Mixed-Precision AdamW ---
    {
        const int N = 256;
        std::vector<__nv_bfloat16> h_p(N, __float2bfloat16(1.0f));
        std::vector<__nv_bfloat16> h_g(N, __float2bfloat16(0.1f));
        std::vector<float> h_m(N, 0.0f), h_v(N, 0.0f);
        float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        float bc1 = 0.1f, bc2 = 0.001f;

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

        adamw_mixed_precision_kernel<<<1, 256>>>(d_p, d_g, d_m, d_v, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_m[i] - 0.01f) > 1e-4f) { ok = false; break; }
            if (std::abs(__bfloat162float(h_p[i]) - 0.9899f) > 0.02f) { ok = false; break; }
        }
        if (ok && h_m[0] != 0.0f) {
            std::cout << "Test 4 Passed: Mixed-precision AdamW update." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Mixed-precision AdamW update." << std::endl;
        }

        cudaFree(d_p); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    }

    // --- Test 5: Clipped AdamW Step ---
    {
        const int N = 256;
        std::vector<float> h_p(N, 1.0f), h_g(N, 10.0f); // large gradient 10.0f
        std::vector<float> h_m(N, 0.0f), h_v(N, 0.0f);
        float clip_val = 0.1f; // should be clipped to 0.1f
        float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.0f;
        float bc1 = 0.1f, bc2 = 0.001f;

        float *d_p, *d_g, *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        adamw_clipped_step_kernel<<<1, 256>>>(d_p, d_g, d_m, d_v, lr, beta1, beta2, eps, wd, bc1, bc2, clip_val, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Clipped to 0.1f -> m = (1-0.9)*0.1 = 0.01f
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_m[i] - 0.01f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_m[0] != 0.0f) {
            std::cout << "Test 5 Passed: AdamW step with gradient clipping." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: AdamW step with gradient clipping." << std::endl;
        }

        cudaFree(d_p); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
