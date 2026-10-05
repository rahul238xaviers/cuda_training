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
// PROBLEM 1: Vectorized Token Embedding Gather (float4 128-bit)
// -----------------------------------------------------------------------------
__global__ void kernel_embedding_gather_float4(
    const int * __restrict__ token_ids,
    const float * __restrict__ embedding_table,
    float *out_embeddings,
    int hidden_dim, // divisible by 4
    int num_tokens
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_dim4 = hidden_dim / 4;
    int total_vectors = num_tokens * hidden_dim4;

    // TODO: 1. Decompose gid into token_pos and vec_d
    // if (gid < total_vectors) {
    //     int token_pos = gid / hidden_dim4;
    //     int vec_d = gid % hidden_dim4;
    //     int token_id = __ldg(&token_ids[token_pos]);
    //     const float4 *table4 = reinterpret_cast<const float4*>(embedding_table);
    //     float4 *out4 = reinterpret_cast<float4*>(out_embeddings);
    //     out4[gid] = table4[token_id * hidden_dim4 + vec_d];
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 4D Tensor Permute [B, S, H, D] -> [B, H, S, D] with Coalesced Shared Tiles
// -----------------------------------------------------------------------------
__global__ void kernel_permute_4d_coalesced(
    const float * __restrict__ in,
    float *out,
    int B, int S, int H, int D
) {
    // Shared memory tile to exchange S and H coordinates coalesced
    __shared__ float s_tile[16][17];
    // TODO: Coalesced tile loading and transposed writing for head/seq dimensions
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Coalesced Transpose-B Dot Product (Rows of A with Rows of B)
// C = A * B^T. Notice B is stored row-major, so reading rows of B is naturally coalesced!
// -----------------------------------------------------------------------------
__global__ void kernel_coalesced_trans_b_dot(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float *C,
    int K
) {
    int row_a = blockIdx.y;
    int row_b = blockIdx.x;
    // Each thread in warp computes partial dot product of A[row_a] and B[row_b]
    // with contiguous coalesced loads from both A and B!
    // TODO: Implement coalesced dot product and warp reduction
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Padded Memory Alignment Buffer Copy
// -----------------------------------------------------------------------------
__global__ void kernel_aligned_padded_copy(
    const float * __restrict__ in,
    float *out,
    int width,
    int height,
    size_t in_pitch_f,
    size_t out_pitch_f
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO: if (row < height && col < width)
    // out[row * out_pitch_f + col] = in[row * in_pitch_f + col] * 2.0f;
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Coalesced Multi-Head Rotary Stride Deinterleaver
// -----------------------------------------------------------------------------
__global__ void kernel_coalesced_rope_split(
    const float * __restrict__ in,
    float *out_first_half,
    float *out_second_half,
    int half_dim,
    int total_heads
) {
    // In LLaMA / Mistral RoPE: head dimension D is split into [0..D/2-1] and [D/2..D-1]
    // TODO: Coalesced 128-bit streaming into two separate half-buffers
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.1 Coalescing & Read-Only Cache (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: float4 Embedding Gather
    {
        int vocab_size = 50, hidden_dim = 64, num_tokens = 4;
        std::vector<float> h_table(vocab_size * hidden_dim);
        for (size_t i = 0; i < h_table.size(); ++i) h_table[i] = (float)i * 0.25f;
        std::vector<int> h_tokens = {10, 4, 0, 49};
        std::vector<float> h_out(num_tokens * hidden_dim, 0.0f);

        int *d_tokens;
        float *d_table, *d_out;
        cudaMalloc(&d_tokens, num_tokens * sizeof(int));
        cudaMalloc(&d_table, h_table.size() * sizeof(float));
        cudaMalloc(&d_out, h_out.size() * sizeof(float));

        cudaMemcpy(d_tokens, h_tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(float), cudaMemcpyHostToDevice);

        int total_vectors = num_tokens * (hidden_dim / 4);
        kernel_embedding_gather_float4<<<(total_vectors + 63) / 64, 64>>>(d_tokens, d_table, d_out, hidden_dim, num_tokens);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int t = 0; t < num_tokens; ++t) {
            int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float exp_val = h_table[tok * hidden_dim + d];
                if (std::fabs(h_out[t * hidden_dim + d] - exp_val) > 1e-4f) { ok = false; break; }
            }
        }
        reportStatus("Problem 1: Vectorized Token Embedding Gather (float4 128-bit)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // Test 2: 4D Tensor Permute
    {
        int B = 1, S = 16, H = 16, D = 16;
        int n = B * S * H * D;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        kernel_permute_4d_coalesced<<<1, block>>>(d_in, d_out, B, S, H, D);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int s = 0; s < S; ++s) {
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < D; ++d) {
                    int in_idx = s * (H * D) + h * D + d;
                    int out_idx = h * (S * D) + s * D + d;
                    if (h_out[out_idx] != h_in[in_idx]) { ok = false; break; }
                }
            }
        }
        reportStatus("Problem 2: 4D Tensor Permute [B, S, H, D] -> [B, H, S, D]", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Transpose-B Dot Product
    {
        int K = 128;
        std::vector<float> h_A(K, 2.0f), h_B(K, 3.0f);
        float h_C = 0.0f;
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, K * sizeof(float));
        cudaMalloc(&d_B, K * sizeof(float));
        cudaMalloc(&d_C, sizeof(float));
        cudaMemcpy(d_A, h_A.data(), K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid(1, 1);
        kernel_coalesced_trans_b_dot<<<grid, 32>>>(d_A, d_B, d_C, K);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_C - (K * 6.0f)) < 1e-3f);
        reportStatus("Problem 3: Coalesced Transpose-B Dot Product (Rows of A & B)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 4: Padded Memory Alignment Copy
    {
        int width = 30, height = 20;
        size_t in_pitch = 32, out_pitch = 64;
        std::vector<float> h_in(height * in_pitch, 4.0f), h_out(height * out_pitch, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, height * in_pitch * sizeof(float));
        cudaMalloc(&d_out, height * out_pitch * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), height * in_pitch * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        kernel_aligned_padded_copy<<<grid, block>>>(d_in, d_out, width, height, in_pitch, out_pitch);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, height * out_pitch * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                if (h_out[r * out_pitch + c] != 8.0f) { ok = false; break; }
            }
        }
        reportStatus("Problem 4: Padded Memory Alignment Buffer Copy", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: RoPE Split
    {
        int half_dim = 32, total_heads = 4;
        int full_len = total_heads * (half_dim * 2);
        std::vector<float> h_in(full_len);
        for (int i = 0; i < full_len; ++i) h_in[i] = (float)i;
        std::vector<float> h_first(total_heads * half_dim, 0.0f);
        std::vector<float> h_second(total_heads * half_dim, 0.0f);

        float *d_in, *d_first, *d_second;
        cudaMalloc(&d_in, full_len * sizeof(float));
        cudaMalloc(&d_first, h_first.size() * sizeof(float));
        cudaMalloc(&d_second, h_second.size() * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), full_len * sizeof(float), cudaMemcpyHostToDevice);

        kernel_coalesced_rope_split<<<total_heads, half_dim>>>(d_in, d_first, d_second, half_dim, total_heads);
        cudaDeviceSynchronize();

        cudaMemcpy(h_first.data(), d_first, h_first.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_second.data(), d_second, h_second.size() * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = true;
        for (int h = 0; h < total_heads; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                if (h_first[h * half_dim + i] != h_in[h * (half_dim * 2) + i]) ok = false;
                if (h_second[h * half_dim + i] != h_in[h * (half_dim * 2) + half_dim + i]) ok = false;
            }
        }
        reportStatus("Problem 5: Coalesced Multi-Head Rotary Stride Deinterleaver", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_first); cudaFree(d_second);
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
