// ============================================================================
// Module 7.2: Embedding Lookup - Champion Workbook
// Kernels Covered:
//   - vectorized_bf16_embedding: 128-bit vectorized BFloat16 embedding lookup
//   - fused_embedding_rmsnorm: Fused lookup + RMS pre-normalization
//   - grid_stride_embedding: Grid-stride embedding lookup for large models
//   - embedding_pad_masked: Dynamic padding token zero-masking
//   - deterministic_embedding_bwd: Coordinated embedding gradient reduction
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

// ============================================================================
// Exercise 1: 128-bit Vectorized BF16 Embedding Lookup
// 16 bytes = 8 x __nv_bfloat16 = float4.
// num_vec = (total_tokens * hidden_dim) / 8.
// ============================================================================
__global__ void vectorized_bf16_embedding_kernel(const unsigned int* token_ids,
                                                 const float4* embedding_table,
                                                 float4* output,
                                                 int hidden_vec8,
                                                 int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_vec8) {
        // TODO:
        // int token_pos = gid / hidden_vec8;
        // int v8 = gid % hidden_vec8;
        // unsigned int token_id = token_ids[token_pos];
        // output[gid] = embedding_table[token_id * hidden_vec8 + v8];
    }
}

// ============================================================================
// Exercise 2: Fused Embedding Lookup + RMS Pre-Norm
// 1 Block per token position. blockDim.x = hidden_dim (e.g. 128 or 256).
// 1. Thread tid looks up val = embedding_table[token_id * D + tid].
// 2. Cooperative sum of squares in shared memory: sum_sq = sum(val^2).
// 3. rms = rsqrtf(sum_sq / D + eps).
// 4. output[token_pos * D + tid] = val * rms.
// ============================================================================
__global__ void fused_embedding_rmsnorm_kernel(const unsigned int* token_ids,
                                               const float* embedding_table,
                                               float* output,
                                               int hidden_dim,
                                               float eps) {
    extern __shared__ float sdata[];
    int token_pos = blockIdx.x;
    int tid = threadIdx.x;

    // TODO:
    // 1. unsigned int token_id = token_ids[token_pos];
    // 2. float val = embedding_table[token_id * hidden_dim + tid];
    // 3. sdata[tid] = val * val;
    // 4. __syncthreads();
    // 5. Tree reduction to sdata[0];
    // 6. __syncthreads();
    // 7. float rms = rsqrtf(sdata[0] / (float)hidden_dim + eps);
    // 8. output[token_pos * hidden_dim + tid] = val * rms;
}

// ============================================================================
// Exercise 3: Grid-Stride Loop Embedding Forward
// Handles arbitrarily large total elements safely across any grid launch size.
// ============================================================================
__global__ void grid_stride_embedding_kernel(const unsigned int* token_ids,
                                             const float* embedding_table,
                                             float* output,
                                             int hidden_dim,
                                             int total_tokens) {
    int total_elements = total_tokens * hidden_dim;
    // TODO:
    // for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total_elements; gid += blockDim.x * gridDim.x) {
    //     int token_pos = gid / hidden_dim;
    //     int d = gid % hidden_dim;
    //     unsigned int token_id = token_ids[token_pos];
    //     output[gid] = embedding_table[token_id * hidden_dim + d];
    // }
}

// ============================================================================
// Exercise 4: Embedding Lookup with Dynamic Padding Suppression
// If token_id == pad_token_id, set entire row output to 0.0f.
// ============================================================================
__global__ void embedding_pad_masked_kernel(const unsigned int* token_ids,
                                            const float* embedding_table,
                                            float* output,
                                            int hidden_dim,
                                            int total_tokens,
                                            unsigned int pad_token_id) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // output[gid] = (token_id == pad_token_id) ? 0.0f : embedding_table[token_id * hidden_dim + d];
    }
}

// ============================================================================
// Exercise 5: Deterministic Embedding Backward
// Accumulates gradients into grad_table using atomic additions across threads.
// ============================================================================
__global__ void deterministic_embedding_bwd_kernel(const unsigned int* token_ids,
                                                   const float* grad_output,
                                                   float* grad_table,
                                                   int hidden_dim,
                                                   int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // atomicAdd(&grad_table[token_id * hidden_dim + d], grad_output[gid]);
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int blockSize = 256;

    const int total_tokens = 4;
    const int hidden_dim = 128;
    const int vocab_size = 100;
    const int total_elem = total_tokens * hidden_dim;

    std::vector<unsigned int> h_tokens = {7, 0, 42, 99};
    std::vector<float> h_table(vocab_size * hidden_dim);
    for (int v = 0; v < vocab_size; ++v) {
        for (int d = 0; d < hidden_dim; ++d) {
            h_table[v * hidden_dim + d] = static_cast<float>(v * 10 + d + 1) * 0.1f;
        }
    }

    // --- Test 1: Vectorized BF16 Embedding Lookup ---
    {
        std::vector<__nv_bfloat16> h_bf_table(vocab_size * hidden_dim);
        for (int i = 0; i < vocab_size * hidden_dim; ++i) {
            h_bf_table[i] = __float2bfloat16(h_table[i]);
        }

        int hidden_vec8 = hidden_dim / 8;
        int total_vec8 = total_tokens * hidden_vec8;

        unsigned int* d_tokens;
        float4 *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_vec8 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_out, total_vec8 * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_bf_table.data(), vocab_size * hidden_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_vec8 * sizeof(float4)));

        int blocks = (total_vec8 + blockSize - 1) / blockSize;
        vectorized_bf16_embedding_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_vec8, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float exp = __bfloat162float(h_bf_table[tok * hidden_dim + d]);
                float act = __bfloat162float(h_out[t * hidden_dim + d]);
                if (std::abs(exp - act) > 0.05f * exp) { ok = false; break; }
            }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 1 Passed: 128-bit vectorized BF16 embedding lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: 128-bit vectorized BF16 embedding lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 2: Fused Embedding + RMSNorm ---
    {
        float eps = 1e-6f;
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        fused_embedding_rmsnorm_kernel<<<total_tokens, hidden_dim, hidden_dim * sizeof(float)>>>(
            d_tokens, d_table, d_out, hidden_dim, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            float sum_sq = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                float v = h_table[tok * hidden_dim + d];
                sum_sq += v * v;
            }
            float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_dim) + eps);
            for (int d = 0; d < hidden_dim; ++d) {
                float exp = h_table[tok * hidden_dim + d] * rms;
                float act = h_out[t * hidden_dim + d];
                if (std::abs(exp - act) > 1e-3f) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 2 Passed: Fused embedding lookup + RMS pre-norm." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Fused embedding lookup + RMS pre-norm." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 3: Grid-Stride Embedding Lookup ---
    {
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        // Intentional small grid to stress grid-stride looping
        grid_stride_embedding_kernel<<<2, 64>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                if (h_out[t * hidden_dim + d] != h_table[tok * hidden_dim + d]) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 3 Passed: Grid-stride embedding lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Grid-stride embedding lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 4: Padding Token Suppression ---
    {
        unsigned int pad_token_id = 42; // token at index 2 is 42!
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_pad_masked_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens, pad_token_id);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int d = 0; d < hidden_dim; ++d) {
            if (h_out[2 * hidden_dim + d] != 0.0f) { ok = false; break; }
        }
        if (h_out[0 * hidden_dim + 0] != h_table[7 * hidden_dim + 0]) ok = false;

        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 4 Passed: Padding token zero-masking." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Padding token zero-masking." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 5: Deterministic Embedding Backward ---
    {
        std::vector<float> h_dy(total_elem, 1.0f);
        unsigned int* d_tokens;
        float *d_dy, *d_grad_table;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_dy, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_grad_table, 0, vocab_size * hidden_dim * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        deterministic_embedding_bwd_kernel<<<blocks, blockSize>>>(d_tokens, d_dy, d_grad_table, hidden_dim, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_grad_table(vocab_size * hidden_dim);
        CUDA_CHECK(cudaMemcpy(h_grad_table.data(), d_grad_table, vocab_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int d = 0; d < hidden_dim; ++d) {
            if (h_grad_table[7 * hidden_dim + d] != 1.0f) { ok = false; break; }
            if (h_grad_table[0 * hidden_dim + d] != 1.0f) { ok = false; break; }
            if (h_grad_table[42 * hidden_dim + d] != 1.0f) { ok = false; break; }
            if (h_grad_table[99 * hidden_dim + d] != 1.0f) { ok = false; break; }
            if (h_grad_table[1 * hidden_dim + d] != 0.0f) { ok = false; break; }
        }
        if (ok) {
            std::cout << "Test 5 Passed: Deterministic embedding backward pass." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: Deterministic embedding backward pass." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_dy); cudaFree(d_grad_table);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
