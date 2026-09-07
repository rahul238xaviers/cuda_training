// ============================================================================
// Module 7.4: Loss & Optimizer - Beginner Workbook
// Kernels Covered:
//   - single_token_loss: Warp-level cross-entropy loss computation
//   - single_token_grad: Softmax logits gradient (prob - is_target)
//   - global_loss_atomic: Global scalar loss accumulation via atomicAdd
//   - adamw_step_basic: Fused AdamW elementwise update
//   - adamw_weight_decay: Decoupled weight decay step
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

// ============================================================================
// Exercise 1: Warp-Level Cross Entropy Loss (V = 32)
// For a single token with 32 classes:
// 1. Thread lane loads logit = logits[lane].
// 2. Compute warp max: max_val = warp_max(logit).
// 3. Compute exp(logit - max_val) and warp sum: sum_exp = warp_sum(exp_val).
// 4. If lane == target_id, compute loss = -logf(exp_val / sum_exp) and store to *out_loss.
// ============================================================================
__global__ void single_token_loss_kernel(const float* logits, unsigned int target_id,
                                         float* out_loss) {
    int lane = threadIdx.x;
    // TODO:
    // float z = logits[lane];
    // float max_val = z;
    // for (int offset = 16; offset > 0; offset /= 2) {
    //     max_val = fmaxf(max_val, __shfl_down_sync(FULL_MASK, max_val, offset));
    // }
    // max_val = __shfl_sync(FULL_MASK, max_val, 0);
    // float exp_z = expf(z - max_val);
    // float sum_exp = exp_z;
    // for (int offset = 16; offset > 0; offset /= 2) {
    //     sum_exp += __shfl_down_sync(FULL_MASK, sum_exp, offset);
    // }
    // sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);
    // if (lane == target_id) {
    //     float prob = exp_z / sum_exp;
    //     *out_loss = -logf(fmaxf(prob, 1e-15f));
    // }
}

// ============================================================================
// Exercise 2: Single-Token Softmax Logits Gradient
// grad_logits[lane] = prob[lane] - (lane == target_id ? 1.0f : 0.0f)
// ============================================================================
__global__ void single_token_grad_kernel(const float* logits, unsigned int target_id,
                                         float* grad_logits) {
    int lane = threadIdx.x;
    // TODO:
    // Compute max_val and sum_exp as in Exercise 1.
    // float prob = exp_z / sum_exp;
    // float is_target = (lane == target_id) ? 1.0f : 0.0f;
    // grad_logits[lane] = prob - is_target;
}

// ============================================================================
// Exercise 3: Global Loss Atomic Accumulation
// Given an array of token losses of length N, atomically sum them into *global_loss.
// ============================================================================
__global__ void global_loss_atomic_kernel(const float* token_losses, float* global_loss, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // atomicAdd(global_loss, token_losses[idx]);
    }
}

// ============================================================================
// Exercise 4: Basic Fused AdamW Step (FP32)
// m = beta1 * m + (1 - beta1) * g
// v = beta2 * v + (1 - beta2) * g^2
// param -= lr * weight_decay * param
// param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
// ============================================================================
__global__ void adamw_step_basic_kernel(float* param, const float* grad,
                                        float* m, float* v,
                                        float lr, float beta1, float beta2,
                                        float eps, float weight_decay,
                                        float bc1, float bc2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // float g = grad[idx];
        // float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        // float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;
        // m[idx] = m_val;
        // v[idx] = v_val;
        // float p_val = param[idx];
        // if (weight_decay > 0.0f) p_val -= lr * weight_decay * p_val;
        // float m_hat = m_val / bc1;
        // float v_hat = v_val / bc2;
        // p_val -= lr * m_hat / (sqrtf(v_hat) + eps);
        // param[idx] = p_val;
    }
}

// ============================================================================
// Exercise 5: Decoupled Weight Decay Only
// param[i] -= lr * weight_decay * param[i]
// ============================================================================
__global__ void adamw_weight_decay_kernel(float* param, float lr, float weight_decay, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO:
        // param[idx] -= lr * weight_decay * param[idx];
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;

    // --- Test 1: Single-Token Loss ---
    {
        const int V = 32;
        std::vector<float> h_logits(V, 0.0f);
        unsigned int target_id = 7;
        h_logits[target_id] = 2.0f; // Target has logit 2.0, others have 0.0

        float *d_logits, *d_loss;
        CUDA_CHECK(cudaMalloc(&d_logits, V * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        single_token_loss_kernel<<<1, 32>>>(d_logits, target_id, d_loss);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

        // exp(2) = 7.389056. 31 others have exp(0) = 1.0 -> sum_exp = 7.389056 + 31 = 38.389056
        // target_prob = 7.389056 / 38.389056 = 0.192478
        // loss = -log(0.192478) = 1.64776
        float exp_loss = -std::log(std::exp(2.0f) / (std::exp(2.0f) + 31.0f));

        if (std::abs(h_loss - exp_loss) < 1e-4f && h_loss != 0.0f) {
            std::cout << "Test 1 Passed: Single-token cross-entropy loss." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Single-token cross-entropy loss (got " << h_loss
                      << ", exp " << exp_loss << ")." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_loss);
    }

    // --- Test 2: Single-Token Logits Gradient ---
    {
        const int V = 32;
        std::vector<float> h_logits(V, 0.0f);
        unsigned int target_id = 0;
        float *d_logits, *d_grad;
        CUDA_CHECK(cudaMalloc(&d_logits, V * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, V * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_grad, 0, V * sizeof(float)));

        single_token_grad_kernel<<<1, 32>>>(d_logits, target_id, d_grad);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_grad(V);
        CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, V * sizeof(float), cudaMemcpyDeviceToHost));

        // For uniform logits = 0, prob = 1/32 = 0.03125f.
        // For target (0): prob - 1 = 0.03125 - 1 = -0.96875f.
        // For others (1..31): prob - 0 = 0.03125f.
        bool ok = true;
        if (std::abs(h_grad[0] - (-0.96875f)) > 1e-4f) ok = false;
        for (int i = 1; i < V; ++i) {
            if (std::abs(h_grad[i] - 0.03125f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_grad[0] != 0.0f) {
            std::cout << "Test 2 Passed: Single-token logits gradient." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Single-token logits gradient." << std::endl;
        }

        cudaFree(d_logits); cudaFree(d_grad);
    }

    // --- Test 3: Global Loss Atomic Accumulation ---
    {
        const int N = 256;
        std::vector<float> h_losses(N, 0.5f);

        float *d_losses, *d_global;
        CUDA_CHECK(cudaMalloc(&d_losses, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_global, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_losses, h_losses.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_global, 0, sizeof(float)));

        global_loss_atomic_kernel<<<1, 256>>>(d_losses, d_global, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_global = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_global, d_global, sizeof(float), cudaMemcpyDeviceToHost));

        if (std::abs(h_global - 128.0f) < 1e-3f && h_global != 0.0f) {
            std::cout << "Test 3 Passed: Global loss atomic accumulation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Global loss atomic accumulation." << std::endl;
        }

        cudaFree(d_losses); cudaFree(d_global);
    }

    // --- Test 4: Basic Fused AdamW Step ---
    {
        const int N = 128;
        std::vector<float> h_p(N, 1.0f), h_g(N, 0.1f), h_m(N, 0.0f), h_v(N, 0.0f);
        float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        float bc1 = 1.0f - 0.9f;    // step 1
        float bc2 = 1.0f - 0.999f;  // step 1

        float *d_p, *d_g, *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        adamw_step_basic_kernel<<<1, 128>>>(d_p, d_g, d_m, d_v, lr, beta1, beta2, eps, wd, bc1, bc2, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Manual Step 1 math:
        // m = 0.9*0 + 0.1*0.1 = 0.01
        // v = 0.999*0 + 0.001*(0.01) = 0.00001
        // m_hat = 0.01 / 0.1 = 0.1
        // v_hat = 0.00001 / 0.001 = 0.01 -> sqrt(v_hat) = 0.1
        // p = 1.0 - 0.01 * 0.01 * 1.0 = 0.9999
        // p -= 0.01 * 0.1 / (0.1 + 1e-8) ~ 0.9999 - 0.01 = 0.9899
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_m[i] - 0.01f) > 1e-4f) { ok = false; break; }
            if (std::abs(h_p[i] - 0.9899f) > 1e-3f) { ok = false; break; }
        }
        if (ok && h_m[0] != 0.0f) {
            std::cout << "Test 4 Passed: Basic fused AdamW step." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Basic fused AdamW step." << std::endl;
        }

        cudaFree(d_p); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    }

    // --- Test 5: Decoupled Weight Decay Only ---
    {
        const int N = 128;
        std::vector<float> h_p(N, 10.0f);
        float lr = 0.1f, wd = 0.2f;

        float* d_p;
        CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        adamw_weight_decay_kernel<<<1, 128>>>(d_p, lr, wd, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // p -= 0.1 * 0.2 * 10 = 10 - 0.2 = 9.8f
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_p[i] - 9.8f) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_p[0] == 9.8f) {
            std::cout << "Test 5 Passed: Decoupled weight decay." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Decoupled weight decay." << std::endl;
        }

        cudaFree(d_p);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
