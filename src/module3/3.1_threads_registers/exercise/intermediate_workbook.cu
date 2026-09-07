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
// PROBLEM 1: 128-bit Vectorized Load/Store (float4) with Boundary Epilogue
// -----------------------------------------------------------------------------
__global__ void kernel_vectorized_float4(const float *in, float *out, float alpha, int n) {
    // TODO: Process elements in chunks of 4 using float4
    // int n4 = n / 4;
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gid < n4) {
    //     float4 v = reinterpret_cast<const float4*>(in)[gid];
    //     v.x *= alpha; v.y *= alpha; v.z *= alpha; v.w *= alpha;
    //     reinterpret_cast<float4*>(out)[gid] = v;
    // }
    // Handle remaining elements (epilogue) in single thread or dedicated logic:
    // int rem_idx = n4 * 4 + gid;
    // if (rem_idx < n) { out[rem_idx] = in[rem_idx] * alpha; }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Fast GELU Activation in Thread Registers
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------
__global__ void kernel_gelu_registers(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO: Load in[gid] into register
        // TODO: Compute GELU using register operations and tanhf()
        // TODO: Store result into out[gid]
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Row-Strided Pitch Indexing for Padded 2D Tensors
// -----------------------------------------------------------------------------
__global__ void kernel_padded_2d(const float *in, float *out, int width, int height, size_t pitch_elements) {
    // TODO: Calculate col and row
    // TODO: if (col < width && row < height)
    // TODO: offset = row * pitch_elements + col;
    // TODO: out[offset] = in[offset] * 2.0f;
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Thread Register Multi-Accumulator (4x Unrolled Inner Loop)
// -----------------------------------------------------------------------------
__global__ void kernel_unrolled_dot(const float *a, const float *b, float *out, int k) {
    // Single thread block or 1 thread computing dot product of length k
    // k is divisible by 4
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // TODO: Maintain 4 independent register accumulators (acc0, acc1, acc2, acc3)
        // TODO: Unroll loop by 4: acc0 += a[i]*b[i], acc1 += a[i+1]*b[i+1], etc.
        // TODO: Combine accumulators into *out
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: In-Register Pairwise Dimension Reversal
// -----------------------------------------------------------------------------
__global__ void kernel_pairwise_reverse(const float *in, float *out, int num_pairs) {
    // Each thread reads a pair (in[2*i], in[2*i+1]), reverses their order in registers,
    // and writes to out[2*i] = in[2*i+1], out[2*i+1] = in[2*i].
    // TODO: Calculate thread index
    // TODO: Load pair into registers
    // TODO: Swap in registers and write to out
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.1 Threads & Registers (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Vectorized float4 with epilogue
    {
        int n = 1027; // Not a multiple of 4
        std::vector<float> h_in(n, 3.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        kernel_vectorized_float4<<<numBlocks, blockSize>>>(d_in, d_out, 2.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 6.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 1: 128-bit Vectorized float4 with Epilogue", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: GELU in registers
    {
        int n = 100;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = -2.0f + (float)i * 0.04f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_gelu_registers<<<1, 128>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        const float SQRT_2_OVER_PI = 0.7978845608f;
        for (int i = 0; i < n; ++i) {
            float x = h_in[i];
            float expected = 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
            if (std::fabs(h_out[i] - expected) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: Fast GELU Activation in Thread Registers", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Padded 2D Matrix
    {
        int width = 30, height = 20;
        size_t pitch_elements = 32; // padded to multiple of 32
        std::vector<float> h_in(height * pitch_elements, 1.5f), h_out(height * pitch_elements, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, height * pitch_elements * sizeof(float));
        cudaMalloc(&d_out, height * pitch_elements * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), height * pitch_elements * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        kernel_padded_2d<<<grid, block>>>(d_in, d_out, width, height, pitch_elements);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, height * pitch_elements * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                if (std::fabs(h_out[r * pitch_elements + c] - 3.0f) > 1e-4f) { ok = false; break; }
            }
        }
        reportStatus("Problem 3: Row-Strided Pitch Indexing for Padded 2D Tensors", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Unrolled Dot Product
    {
        int k = 256;
        std::vector<float> h_a(k, 1.0f), h_b(k, 2.0f);
        float h_out = 0.0f;
        float *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, k * sizeof(float));
        cudaMalloc(&d_b, k * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_a, h_a.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), k * sizeof(float), cudaMemcpyHostToDevice);

        kernel_unrolled_dot<<<1, 1>>>(d_a, d_b, d_out, k);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 512.0f) < 1e-4f);
        reportStatus("Problem 4: Thread Register Multi-Accumulator (4x Unrolled)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    // Test 5: Pairwise Reversal
    {
        int num_pairs = 128;
        std::vector<float> h_in(num_pairs * 2), h_out(num_pairs * 2, 0.0f);
        for (int i = 0; i < num_pairs * 2; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, num_pairs * 2 * sizeof(float));
        cudaMalloc(&d_out, num_pairs * 2 * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), num_pairs * 2 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_pairwise_reverse<<<(num_pairs + 127) / 128, 128>>>(d_in, d_out, num_pairs);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, num_pairs * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < num_pairs; ++i) {
            if (h_out[2 * i] != h_in[2 * i + 1] || h_out[2 * i + 1] != h_in[2 * i]) {
                ok = false; break;
            }
        }
        reportStatus("Problem 5: In-Register Pairwise Dimension Reversal", ok);
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
