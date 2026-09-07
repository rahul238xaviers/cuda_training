// ============================================================================
// Module 7.2: Embedding Lookup - Intermediate Workbook
// Kernels Covered:
//   - embedding_vec4: 128-bit float4 vectorized gather
//   - block_per_token_embedding: Threadblock cooperative embedding copy
//   - embedding_backward_atomic: Sparse gradient accumulation into embedding weights
//   - embedding_dropout: Embedding lookup with inverted dropout application
//   - embedding_2d_batch_seq: 2D batch & sequence coordinate mapping
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

// ============================================================================
// Exercise 1: 128-bit Vectorized Embedding Lookup (float4)
// num_vec = (total_tokens * hidden_dim) / 4.
// hidden_vec = hidden_dim / 4.
// ============================================================================
__global__ void embedding_vec4_kernel(const unsigned int* token_ids,
                                      const float4* embedding_table,
                                      float4* output,
                                      int hidden_vec,
                                      int total_tokens) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_vec) {
        // TODO:
        // int token_pos = gid / hidden_vec;
        // int vd = gid % hidden_vec;
        // unsigned int token_id = token_ids[token_pos];
        // output[gid] = embedding_table[token_id * hidden_vec + vd];
    }
}

// ============================================================================
// Exercise 2: Block-per-Token Cooperative Embedding Copy
// gridDim.x = total_tokens. blockDim.x = 256.
// Thread 0 reads token ID into shared memory.
// Threads in block cooperatively copy the embedding row from global table to output.
// ============================================================================
__global__ void block_per_token_embedding_kernel(const unsigned int* token_ids,
                                                 const float* embedding_table,
                                                 float* output,
                                                 int hidden_dim) {
    __shared__ unsigned int s_token_id;
    int token_pos = blockIdx.x;
    int tid = threadIdx.x;

    // TODO:
    // 1. If tid == 0, s_token_id = token_ids[token_pos];
    // 2. __syncthreads();
    // 3. Loop d = tid; d < hidden_dim; d += blockDim.x:
    //    output[token_pos * hidden_dim + d] = embedding_table[s_token_id * hidden_dim + d];
}

// ============================================================================
// Exercise 3: Embedding Backward Pass (Sparse Atomic Accumulation)
// In training, gradients from multiple token positions with the SAME token ID
// must be summed into the embedding table weights:
// grad_table[token_id * hidden_dim + d] += grad_out[pos * hidden_dim + d]
// ============================================================================
__global__ void embedding_backward_atomic_kernel(const unsigned int* token_ids,
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
// Exercise 4: Embedding Lookup with Inverted Dropout
// If mask[gid] == 1: output[gid] = val * (1.0f / (1.0f - drop_prob))
// If mask[gid] == 0: output[gid] = 0.0f
// ============================================================================
__global__ void embedding_dropout_kernel(const unsigned int* token_ids,
                                         const float* embedding_table,
                                         const unsigned char* mask,
                                         float* output,
                                         int hidden_dim,
                                         int total_tokens,
                                         float drop_prob) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_tokens * hidden_dim) {
        // TODO:
        // int token_pos = gid / hidden_dim;
        // int d = gid % hidden_dim;
        // unsigned int token_id = token_ids[token_pos];
        // float val = embedding_table[token_id * hidden_dim + d];
        // float scale = 1.0f / (1.0f - drop_prob);
        // output[gid] = (mask[gid] != 0) ? (val * scale) : 0.0f;
    }
}

// ============================================================================
// Exercise 5: Explicit 2D [Batch, Seq] Embedding Mapping
// token_ids has shape [Batch, Seq]
// output has shape [Batch, Seq, Hidden]
// ============================================================================
__global__ void embedding_2d_batch_seq_kernel(const unsigned int* token_ids,
                                              const float* embedding_table,
                                              float* output,
                                              int batch, int seq_len, int hidden_dim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_dim;
    if (gid < total) {
        // TODO:
        // int d = gid % hidden_dim;
        // int s = (gid / hidden_dim) % seq_len;
        // int b = gid / (seq_len * hidden_dim);
        // unsigned int token_id = token_ids[b * seq_len + s];
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

    std::vector<unsigned int> h_tokens = {3, 7, 3, 15}; // notice token 3 repeated!
    std::vector<float> h_table(vocab_size * hidden_dim);
    for (int v = 0; v < vocab_size; ++v) {
        for (int d = 0; d < hidden_dim; ++d) {
            h_table[v * hidden_dim + d] = static_cast<float>(v + 1) * 0.5f;
        }
    }

    // --- Test 1: 128-bit Vectorized Embedding Lookup ---
    {
        int hidden_vec = hidden_dim / 4;
        int total_vec = total_tokens * hidden_vec;

        unsigned int* d_tokens;
        float4 *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_vec * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_out, total_vec * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_vec * sizeof(float4)));

        int blocks = (total_vec + blockSize - 1) / blockSize;
        embedding_vec4_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, hidden_vec, total_tokens);
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
            std::cout << "Test 1 Passed: 128-bit vectorized embedding lookup." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: 128-bit vectorized embedding lookup." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 2: Block-per-Token Embedding Copy ---
    {
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        block_per_token_embedding_kernel<<<total_tokens, 64>>>(d_tokens, d_table, d_out, hidden_dim);
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
            std::cout << "Test 2 Passed: Block-per-token cooperative embedding copy." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Block-per-token cooperative embedding copy." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // --- Test 3: Embedding Backward Atomic Accumulation ---
    {
        std::vector<float> h_dy(total_elem, 2.0f);
        unsigned int* d_tokens;
        float *d_dy, *d_grad_table;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_dy, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total_elem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_grad_table, 0, vocab_size * hidden_dim * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_backward_atomic_kernel<<<blocks, blockSize>>>(d_tokens, d_dy, d_grad_table, hidden_dim, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_grad_table(vocab_size * hidden_dim);
        CUDA_CHECK(cudaMemcpy(h_grad_table.data(), d_grad_table, vocab_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        // Token 3 appears twice (pos 0 and pos 2), so grad for token 3 should be 2.0 + 2.0 = 4.0f
        for (int d = 0; d < hidden_dim; ++d) {
            if (h_grad_table[3 * hidden_dim + d] != 4.0f) { ok = false; break; }
            if (h_grad_table[7 * hidden_dim + d] != 2.0f) { ok = false; break; }
            if (h_grad_table[15 * hidden_dim + d] != 2.0f) { ok = false; break; }
            if (h_grad_table[0 * hidden_dim + d] != 0.0f) { ok = false; break; }
        }
        if (ok) {
            std::cout << "Test 3 Passed: Embedding backward sparse atomic accumulation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Embedding backward sparse atomic accumulation." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_dy); cudaFree(d_grad_table);
    }

    // --- Test 4: Embedding Dropout ---
    {
        float drop_prob = 0.5f;
        std::vector<unsigned char> h_mask(total_elem);
        for (int i = 0; i < total_elem; ++i) h_mask[i] = (i % 2 == 0) ? 1 : 0;

        unsigned int* d_tokens;
        float *d_table, *d_out;
        unsigned char* d_mask;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, total_elem * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), total_elem * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_dropout_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_mask, d_out, hidden_dim, total_tokens, drop_prob);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        float scale = 1.0f / (1.0f - drop_prob); // 2.0f
        for (int i = 0; i < total_elem; ++i) {
            int t = i / hidden_dim;
            int d = i % hidden_dim;
            float val = h_table[h_tokens[t] * hidden_dim + d];
            float exp = (h_mask[i] != 0) ? (val * scale) : 0.0f;
            if (std::abs(h_out[i] - exp) > 1e-4f) { ok = false; break; }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 4 Passed: Embedding lookup with inverted dropout." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Embedding lookup with inverted dropout." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_mask); cudaFree(d_out);
    }

    // --- Test 5: 2D [Batch, Seq] Mapping ---
    {
        const int B = 2, S = 2; // total_tokens = 4
        unsigned int* d_tokens;
        float *d_table, *d_out;
        CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, total_elem * sizeof(float)));

        int blocks = (total_elem + blockSize - 1) / blockSize;
        embedding_2d_batch_seq_kernel<<<blocks, blockSize>>>(d_tokens, d_table, d_out, B, S, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(total_elem);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elem * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                unsigned int tok = h_tokens[b * S + s];
                for (int d = 0; d < hidden_dim; ++d) {
                    float exp = h_table[tok * hidden_dim + d];
                    float act = h_out[(b * S + s) * hidden_dim + d];
                    if (exp != act) { ok = false; break; }
                }
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 5 Passed: 2D [Batch, Seq] embedding mapping." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: 2D [Batch, Seq] embedding mapping." << std::endl;
        }

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
