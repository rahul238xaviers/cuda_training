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
// PROBLEM 1: 1D Global Thread Indexing with Bounds Check
// -----------------------------------------------------------------------------
__global__ void kernel_scale_1d(const float *in, float *out, float scale, int n) {
    // TODO: Calculate 1D global thread index 'gid'
    // TODO: Add bounds check (gid < n)
    // TODO: out[gid] = in[gid] * scale;
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 2D Matrix Coordinate Indexing to Flattened Buffer
// -----------------------------------------------------------------------------
__global__ void kernel_add_matrix_2d(const float *a, const float *b, float *c, int rows, int cols) {
    // TODO: Calculate 2D global coordinates 'col' and 'row'
    //       col = blockIdx.x * blockDim.x + threadIdx.x
    //       row = blockIdx.y * blockDim.y + threadIdx.y
    // TODO: Bounds check: if (row < rows && col < cols)
    // TODO: Calculate flat index: idx = row * cols + col
    // TODO: c[idx] = a[idx] + b[idx];
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Register Accumulation (Fused Scale + Bias + ReLU)
// -----------------------------------------------------------------------------
__global__ void kernel_fused_relu(const float *in, float *out, float scale, float bias, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO: Read in[gid] into a thread register variable 'val'
        // TODO: Perform val = val * scale + bias;
        // TODO: Perform ReLU: val = (val > 0.0f) ? val : 0.0f;
        // TODO: Store 'val' to out[gid]
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Vectorized Memory Access with float2 (64-bit Loads)
// -----------------------------------------------------------------------------
__global__ void kernel_vector_add_float2(const float2 *a, const float2 *b, float2 *c, int num_pairs) {
    // TODO: Calculate 1D thread index
    // TODO: Check bounds against 'num_pairs'
    // TODO: Load pair from 'a' and 'b' into thread registers
    // TODO: Compute c[idx].x = a[idx].x + b[idx].x and c[idx].y = a[idx].y + b[idx].y
}

// -----------------------------------------------------------------------------
// PROBLEM 5: 3D Grid Indexing for Tensor Head/Channel Mapping
// -----------------------------------------------------------------------------
__global__ void kernel_tensor_3d(const float *in, float *out, int B, int H, int D) {
    // Shape: [Batch, Heads, HiddenDim]
    // TODO: Map threadIdx/blockIdx x->D, y->H, z->B
    // TODO: Perform bounds checks
    // TODO: Compute flat offset: flat_idx = b * (H * D) + h * D + d
    // TODO: out[flat_idx] = in[flat_idx] + (float)h;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.1 Threads & Registers (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: 1D Scale
    {
        int n = 1024;
        std::vector<float> h_in(n, 2.5f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        kernel_scale_1d<<<numBlocks, blockSize>>>(d_in, d_out, 3.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 7.5f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 1: 1D Global Thread Indexing with Bounds Check", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in);
        cudaFree(d_out);
    }

    // Test 2: 2D Matrix Add
    {
        int rows = 32, cols = 64, n = rows * cols;
        std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n, 0.0f);
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        kernel_add_matrix_2d<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_c[i] - 3.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: 2D Matrix Coordinate Indexing", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    // Test 3: Fused ReLU
    {
        int n = 512;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)(i - 256);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_fused_relu<<<(n + 255) / 256, 256>>>(d_in, d_out, 0.5f, -10.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            float expected = (h_in[i] * 0.5f - 10.0f > 0.0f) ? (h_in[i] * 0.5f - 10.0f) : 0.0f;
            if (std::fabs(h_out[i] - expected) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: Register Accumulation (Fused Scale + Bias + ReLU)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: float2 Vectorized Add
    {
        int num_pairs = 512;
        int n = num_pairs * 2;
        std::vector<float> h_a(n, 1.5f), h_b(n, 2.5f), h_c(n, 0.0f);
        float2 *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, num_pairs * sizeof(float2));
        cudaMalloc(&d_b, num_pairs * sizeof(float2));
        cudaMalloc(&d_c, num_pairs * sizeof(float2));
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_vector_add_float2<<<(num_pairs + 255) / 256, 256>>>(d_a, d_b, d_c, num_pairs);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_c[i] - 4.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 4: Vectorized float2 64-bit Memory Access", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    // Test 5: 3D Grid Indexing
    {
        int B = 2, H = 4, D = 16, n = B * H * D;
        std::vector<float> h_in(n, 10.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(8, 4, 1);
        dim3 grid((D + block.x - 1) / block.x, (H + block.y - 1) / block.y, B);
        kernel_tensor_3d<<<grid, block>>>(d_in, d_out, B, H, D);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < D; ++d) {
                    int idx = b * (H * D) + h * D + d;
                    float expected = 10.0f + (float)h;
                    if (std::fabs(h_out[idx] - expected) > 1e-4f) { ok = false; break; }
                }
            }
        }
        reportStatus("Problem 5: 3D Grid Indexing for Multi-Head Tensors", ok);
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
