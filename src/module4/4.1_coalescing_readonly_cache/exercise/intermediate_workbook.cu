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
// PROBLEM 1: Embedding Gather via Read-Only Cache (__restrict__)
// -----------------------------------------------------------------------------
__global__ void kernel_embedding_gather(
    const int * __restrict__ token_ids,
    const float * __restrict__ embedding_table,
    float *out_embeddings,
    int hidden_dim,
    int num_tokens
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_dim;
    // TODO: Decompose gid into token_idx and dim_idx
    // if (gid < total_elements) {
    //     int token_idx = gid / hidden_dim;
    //     int d = gid % hidden_dim;
    //     int token_id = __ldg(&token_ids[token_idx]);
    //     out_embeddings[gid] = __ldg(&embedding_table[token_id * hidden_dim + d]);
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Coalesced Column-Wise Reduction via Shared Memory Transposition
// Reading down a column in DRAM is uncoalesced. Load a square tile coalesced,
// transpose in shared memory, then reduce!
// -----------------------------------------------------------------------------
__global__ void kernel_coalesced_col_sum(const float *in, float *out_col_sums, int rows, int cols) {
    __shared__ float s_tile[32][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // TODO: Coalesced row-major load into s_tile[ty][tx]
    // TODO: __syncthreads();
    // TODO: Transpose read s_tile[tx][ty] to accumulate col sum
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Vectorized Interleaved Merge (float2 to float)
// -----------------------------------------------------------------------------
__global__ void kernel_interleaved_merge(const float2 *in_pairs, float *out, int num_pairs) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Load in_pairs[gid], write out[2 * gid] = val.x, out[2 * gid + 1] = val.y
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Non-Aligned Memory Boundary Offset Alignment
// -----------------------------------------------------------------------------
__global__ void kernel_unaligned_offset_fix(const float *in, float *out, int byte_offset, int n) {
    // Input pointer may be offset by an arbitrary byte count (e.g. +4 or +8 bytes).
    // Safely load and process without alignment faults.
    // TODO: Compute scalar float offset and copy with scale
}

// -----------------------------------------------------------------------------
// PROBLEM 5: 3x3 2D Constant Memory Filter Convolution
// -----------------------------------------------------------------------------
__constant__ float c_sobel_3x3[9];

__global__ void kernel_constant_filter_2d(const float *in, float *out, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO: Compute 3x3 convolution using c_sobel_3x3 from constant memory
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.1 Coalescing & Read-Only Cache (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Embedding Gather
    {
        int vocab_size = 100, hidden_dim = 64, num_tokens = 4;
        std::vector<float> h_table(vocab_size * hidden_dim);
        for (size_t i = 0; i < h_table.size(); ++i) h_table[i] = (float)i * 0.1f;
        std::vector<int> h_tokens = {5, 20, 0, 99};
        std::vector<float> h_out(num_tokens * hidden_dim, 0.0f);

        int *d_tokens;
        float *d_table, *d_out;
        cudaMalloc(&d_tokens, num_tokens * sizeof(int));
        cudaMalloc(&d_table, h_table.size() * sizeof(float));
        cudaMalloc(&d_out, h_out.size() * sizeof(float));

        cudaMemcpy(d_tokens, h_tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(float), cudaMemcpyHostToDevice);

        int total_len = num_tokens * hidden_dim;
        kernel_embedding_gather<<<(total_len + 127) / 128, 128>>>(d_tokens, d_table, d_out, hidden_dim, num_tokens);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int t = 0; t < num_tokens; ++t) {
            int tok = h_tokens[t];
            for (int d = 0; d < hidden_dim; ++d) {
                float expected = h_table[tok * hidden_dim + d];
                if (std::fabs(h_out[t * hidden_dim + d] - expected) > 1e-4f) { ok = false; break; }
            }
        }
        reportStatus("Problem 1: Embedding Gather via Read-Only Cache (__restrict__)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_tokens); cudaFree(d_table); cudaFree(d_out);
    }

    // Test 2: Coalesced Column-Wise Reduction
    {
        int rows = 32, cols = 32;
        std::vector<float> h_in(rows * cols, 1.0f), h_out(cols, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, rows * cols * sizeof(float));
        cudaMalloc(&d_out, cols * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 32);
        kernel_coalesced_col_sum<<<1, block>>>(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, cols * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int c = 0; c < cols; ++c) {
            if (std::fabs(h_out[c] - 32.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: Coalesced Column-Wise Reduction via Shared Transposition", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Interleaved Merge
    {
        int num_pairs = 128;
        std::vector<float2> h_pairs(num_pairs);
        for (int i = 0; i < num_pairs; ++i) {
            h_pairs[i].x = (float)(i * 2);
            h_pairs[i].y = (float)(i * 2 + 1);
        }
        std::vector<float> h_out(num_pairs * 2, 0.0f);
        float2 *d_pairs;
        float *d_out;
        cudaMalloc(&d_pairs, num_pairs * sizeof(float2));
        cudaMalloc(&d_out, num_pairs * 2 * sizeof(float));
        cudaMemcpy(d_pairs, h_pairs.data(), num_pairs * sizeof(float2), cudaMemcpyHostToDevice);

        kernel_interleaved_merge<<<(num_pairs + 63) / 64, 64>>>(d_pairs, d_out, num_pairs);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, num_pairs * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < num_pairs * 2; ++i) {
            if (h_out[i] != (float)i) { ok = false; break; }
        }
        reportStatus("Problem 3: Vectorized Interleaved Merge (float2 to float)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_pairs); cudaFree(d_out);
    }

    // Test 4: Unaligned Offset Fix
    {
        int n = 256;
        std::vector<float> h_in(n + 8);
        for (int i = 0; i < n + 8; ++i) h_in[i] = (float)i;
        std::vector<float> h_out(n, 0.0f);

        float *d_in, *d_out;
        cudaMalloc(&d_in, (n + 8) * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), (n + 8) * sizeof(float), cudaMemcpyHostToDevice);

        kernel_unaligned_offset_fix<<<1, 256>>>(d_in + 3, d_out, 3, n); // Offset by 3 floats (12 bytes, not 16-aligned)
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != h_in[i + 3] * 2.0f) { ok = false; break; }
        }
        reportStatus("Problem 4: Non-Aligned Memory Boundary Offset Alignment", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: 3x3 Constant Filter
    {
        int width = 16, height = 16;
        float h_sobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        cudaMemcpyToSymbol(c_sobel_3x3, h_sobel, 9 * sizeof(float));

        std::vector<float> h_in(width * height, 0.0f), h_out(width * height, 0.0f);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                h_in[r * width + c] = (c < 8) ? 10.0f : 0.0f; // Step edge at col 8
            }
        }
        float *d_in, *d_out;
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        kernel_constant_filter_2d<<<1, block>>>(d_in, d_out, width, height);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // At c = 7, left neighbors are 10, right neighbors are 0, Sobel response is 10*(1+2+1) = 40.0f
        for (int r = 1; r < height - 1; ++r) {
            if (std::fabs(h_out[r * width + 7] - 40.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 5: 3x3 2D Constant Memory Filter Convolution", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
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
