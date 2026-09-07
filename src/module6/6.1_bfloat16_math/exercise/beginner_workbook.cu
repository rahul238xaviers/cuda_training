// ============================================================================
// Module 6.1: BFloat16 Math - Beginner Workbook
//
// Concepts Covered:
//   - __nv_bfloat16 data type
//   - __float2bfloat16 and __bfloat162float conversions
//   - Scalar and elementwise vector operations in BF16
//   - Mixed-precision arithmetic (FP32 compute, BF16 storage)
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
// Exercise 1: Float to BF16 Conversion and Round-Trip
// Convert an input array of FP32 floats to __nv_bfloat16, and then back to FP32.
// ============================================================================
__global__ void float_to_bf16_and_back_kernel(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Convert in[idx] to __nv_bfloat16 using __float2bfloat16
        // Then convert it back to float using __bfloat162float and write to out[idx]
    }
}

// ============================================================================
// Exercise 2: Vector Scaling in BF16
// Compute y[i] = alpha * x[i] where alpha is float, x and y are __nv_bfloat16.
// Compute the multiplication in float (mixed-precision) and write back as BF16.
// ============================================================================
__global__ void bf16_scale_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Load x[idx], convert to float, multiply by alpha,
        // convert result to __nv_bfloat16, and store in y[idx]
    }
}

// ============================================================================
// Exercise 3: Elementwise BF16 FMA (Fused Multiply-Add)
// Compute out[i] = a[i] * b[i] + c[i] where all arrays are __nv_bfloat16.
// Perform the intermediate multiply-add in FP32 precision to avoid precision loss.
// ============================================================================
__global__ void bf16_fma_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                               const __nv_bfloat16* c, __nv_bfloat16* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Load a[idx], b[idx], c[idx]
        // Compute (a_f * b_f + c_f) in float
        // Convert to __nv_bfloat16 and write to out[idx]
    }
}

// ============================================================================
// Exercise 4: BF16 ReLU Activation
// Compute out[i] = max(0, in[i]) for __nv_bfloat16 elements.
// ============================================================================
__global__ void bf16_relu_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Load in[idx], apply ReLU (fmaxf(val_f, 0.0f)),
        // convert to __nv_bfloat16 and write to out[idx]
    }
}

// ============================================================================
// Exercise 5: BF16 Array Quantization
// Given a raw float array, quantize it into a __nv_bfloat16 array on device.
// ============================================================================
__global__ void fp32_to_bf16_quantize_kernel(const float* in, __nv_bfloat16* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Convert in[idx] to __nv_bfloat16 and store in out[idx]
    }
}

// ============================================================================
// Verification Harness
// ============================================================================
int main() {
    int passed = 0;
    const int total = 5;
    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // --- Test 1: Float to BF16 Round-Trip ---
    {
        std::vector<float> h_in(N), h_out(N);
        for (int i = 0; i < N; ++i) h_in[i] = 1.0f + static_cast<float>(i) * 0.0625f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        float_to_bf16_and_back_kernel<<<numBlocks, blockSize>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            // BF16 has 7 bits of mantissa, ~1% precision
            if (std::abs(h_in[i] - h_out[i]) > 0.05f * std::abs(h_in[i])) {
                ok = false;
                break;
            }
        }
        if (ok && h_out[0] != 0.0f) {
            std::cout << "Test 1 Passed: Float to BF16 round-trip accurate within BF16 tolerance." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Float to BF16 round-trip mismatch." << std::endl;
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }

    // --- Test 2: BF16 Vector Scaling ---
    {
        std::vector<__nv_bfloat16> h_x(N), h_y(N);
        float alpha = 2.5f;
        for (int i = 0; i < N; ++i) h_x[i] = __float2bfloat16(static_cast<float>(i + 1) * 0.5f);

        __nv_bfloat16 *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(__nv_bfloat16)));

        bf16_scale_kernel<<<numBlocks, blockSize>>>(d_x, d_y, alpha, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float expected = __bfloat162float(h_x[i]) * alpha;
            float actual = __bfloat162float(h_y[i]);
            if (std::abs(expected - actual) > 0.05f * std::abs(expected)) {
                ok = false;
                break;
            }
        }
        if (ok && __bfloat162float(h_y[0]) != 0.0f) {
            std::cout << "Test 2 Passed: BF16 vector scaling." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: BF16 vector scaling." << std::endl;
        }

        cudaFree(d_x);
        cudaFree(d_y);
    }

    // --- Test 3: Elementwise BF16 FMA ---
    {
        std::vector<__nv_bfloat16> h_a(N), h_b(N), h_c(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_a[i] = __float2bfloat16(1.5f);
            h_b[i] = __float2bfloat16(2.0f);
            h_c[i] = __float2bfloat16(0.5f);
        }

        __nv_bfloat16 *d_a, *d_b, *d_c, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        bf16_fma_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float expected = 1.5f * 2.0f + 0.5f; // 3.5f
            float actual = __bfloat162float(h_out[i]);
            if (std::abs(expected - actual) > 0.05f) {
                ok = false;
                break;
            }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 3 Passed: Elementwise BF16 FMA." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Elementwise BF16 FMA." << std::endl;
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_out);
    }

    // --- Test 4: BF16 ReLU ---
    {
        std::vector<__nv_bfloat16> h_in(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            float val = static_cast<float>(i - N / 2) * 0.1f;
            h_in[i] = __float2bfloat16(val);
        }

        __nv_bfloat16 *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        bf16_relu_kernel<<<numBlocks, blockSize>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float in_f = __bfloat162float(h_in[i]);
            float expected = (in_f > 0.0f) ? in_f : 0.0f;
            float actual = __bfloat162float(h_out[i]);
            if (std::abs(expected - actual) > 0.05f) {
                ok = false;
                break;
            }
        }
        if (ok && __bfloat162float(h_out[N - 1]) > 0.0f) {
            std::cout << "Test 4 Passed: BF16 ReLU activation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: BF16 ReLU activation." << std::endl;
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }

    // --- Test 5: FP32 to BF16 Quantize ---
    {
        std::vector<float> h_in(N);
        std::vector<__nv_bfloat16> h_out(N);
        for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i + 1) * 3.14159f;

        float* d_in;
        __nv_bfloat16* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        fp32_to_bf16_quantize_kernel<<<numBlocks, blockSize>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float actual = __bfloat162float(h_out[i]);
            if (std::abs(h_in[i] - actual) > 0.05f * std::abs(h_in[i])) {
                ok = false;
                break;
            }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 5 Passed: FP32 to BF16 quantization." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: FP32 to BF16 quantization." << std::endl;
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
