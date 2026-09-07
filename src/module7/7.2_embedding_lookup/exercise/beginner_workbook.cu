// ============================================================================
// Module 7.2: Embedding Lookup - Beginner Workbook
// Kernel Covered:
//   - embedding_forward: GPU gather lookup from embedding weight matrix
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
// Exercise 1: Basic Token Embedding Forward (FP32)
// For each thread gid:
//   token_pos = gid / hidden_dim
//   d = gid % hidden_dim
//   token_id = token_ids[token_pos]
//   output[gid] = embedding_table[token_id * hidden_dim + d]
// ============================================================================
__global__ void embedding_forward_kernel(const unsigned int* token_ids,
                                         const float* embedding_table,
                                         float* output,
                                         int hidden_dim,
                                         int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // output[gid] = embedding_table[token_id * hidden_dim + d];
    }
}

// ============================================================================
// Exercise 2: Embedding Lookup with Vocab Bounds Check
// If token_id >= vocab_size, output 0.0f (or padding value)
// ============================================================================
__global__ void embedding_bounds_checked_kernel(const unsigned int* token_ids,
                                                const float* embedding_table,
                                                float* output,
                                                int hidden_dim,
                                                int total_tokens,
                                                unsigned int vocab_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // if (token_id < vocab_size) {
        //     output[gid] = embedding_table[token_id * hidden_dim + d];
        // } else {
        //     output[gid] = 0.0f;
        // }
    }
}

// ============================================================================
// Exercise 3: Scaled Embedding Lookup (scale = sqrt(hidden_dim))
// In Attention Is All You Need, embeddings are multiplied by sqrt(d_model).
// ============================================================================
__global__ void embedding_scaled_kernel(const unsigned int* token_ids,
                                        const float* embedding_table,
                                        float* output,
                                        int hidden_dim,
                                        int total_tokens,
                                        float scale) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // Look up embedding and multiply by scale before storing to output[gid].
    }
}

// ============================================================================
// Exercise 4: Fused Token + Positional Embedding Addition
// output[pos, d] = token_table[token_id * D + d] + pos_table[token_pos * D + d]
// ============================================================================
__global__ void fused_token_pos_embedding_kernel(const unsigned int* token_ids,
                                                 const float* token_table,
                                                 const float* pos_table,
                                                 float* output,
                                                 int hidden_dim,
                                                 int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // Compute sum of token embedding and positional embedding.
    }
}

// ============================================================================
// Exercise 5: BFloat16 Embedding Forward
// Matching embedding_forward.metal with __nv_bfloat16
// ============================================================================
__global__ void embedding_bf16_kernel(const unsigned int* token_ids,
                                      const __nv_bfloat16* embedding_table,
                                      __nv_bfloat16* output,
                                      int hidden_dim,
                                      int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // output[gid] = embedding_table[token_id * hidden_dim + d];
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
    const int hidden_dim = 64;
    const int vocab_size = 100;
    const int total_elem = total_tokens * hidden_dim;

    std::vector<unsigned int> h_tokens = {5, 12, 0, 99};
    std::vector<float> h_table(vocab_size * hidden_dim);
    for (int v = 0; v < vocab_size; ++v) {
        for (int d = 0; d < hidden_dim; ++d) {
            h_table[v * hidden_dim + d] = static_cast<float>(v * 100 + d + 1);
        }
    }

    // --- Test 1: Basic Embedding Lookup ---
    {
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_forward_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float exp = h_table[tok * hidden_dim + d];
                float act = h_out[t * hidden_dim + d];
                if (exp != act) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 1 Passed: Basic embedding forward lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Basic embedding forward lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 2: Bounds Checked Lookup ---
    {
        std::vector<unsigned int> h_bad_tokens = {5, 105, 0, 99}; // 105 >= vocab_size (100)
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_bad_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, -1, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_bounds_checked_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens, vocab_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Token 1 (idx 105) should be zeroed out
        for (int d = 0; d < hidden_dim; ++d) {
            if (h_out[1 * hidden_dim + d] != 0.0f) { ok = false; break; }
        }
        // Token 0 should match token 5
        if (h_out[0] != h_table[5 * hidden_dim + 0]) ok = false;

        if (ok) {
            std::cout << "Test 2 Passed: Bounds-checked embedding lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Bounds-checked embedding lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 3: Scaled Embedding Lookup ---
    {
        float scale = 8.0f;
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_scaled_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens, scale);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float exp = h_table[tok * hidden_dim + d] * scale;
                float act = h_out[t * hidden_dim + d];
                if (std::abs(exp - act) > 1e-3f) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 3 Passed: Scaled embedding lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Scaled embedding lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 4: Fused Token + Positional Embedding ---
    {
        std::vector<float> h_pos(total_tokens * hidden_dim, 10.0f);
        unsigned int* d_tokens;
        float *d_table, *d_pos, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, total_tokens * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), total_tokens * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        fused_token_pos_embedding_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_pos, d_out, hidden_dim, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int t = 0; t < total_tokens; ++t) {
            unsigned int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float exp = h_table[tok * hidden_dim + d] + 10.0f;
                float act = h_out[t * hidden_dim + d];
                if (std::abs(exp - act) > 1e-3f) { ok = false; break; }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 4 Passed: Fused token + positional embedding." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Fused token + positional embedding." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_pos); cudaFree(d_out);
    }

    // --- Test 5: BF16 Embedding Lookup ---
    {
        std::vector<__nv_bfloat16> h_bf_table(vocab_size * hidden_dim);
        for (int i = 0; i < vocab_size * hidden_dim; ++i) {
            h_bf_table[i] = __float2bfloat16(h_table[i]);
        }

        unsigned int* d_tokens;
        __nv_bfloat16 *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_bf_table.data(), vocab_size * hidden_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(__nv_bfloat16)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_bf16_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_dim, total_tokens);
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
            std::cout << "Test 5 Passed: BF16 embedding forward lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: BF16 embedding forward lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
