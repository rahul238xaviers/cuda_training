// ============================================================================
// Module 6.1: BFloat16 Math - Intermediate Workbook
//
// Concepts Covered:
//   - Packed vectorization using __nv_bfloat162
//   - Dual-issue SIMD arithmetic
//   - Handling non-multiple-of-2 array sizes (tail cleanup)
//   - Warp reduction in mixed precision (BF16 input, FP32 accumulator)
//   - BF16 GELU activation function
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

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ============================================================================
// Exercise 1: Packed Vector Addition with __nv_bfloat162
// Cast __nv_bfloat16* to __nv_bfloat162* and add pairs of BF16 numbers in parallel.
// N is assumed to be even. Total pairs = N / 2.
// ============================================================================
__global__ void packed_bf162_add_kernel(const __nv_bfloat162* a2,
                                        const __nv_bfloat162* b2,
                                        __nv_bfloat162* out2,
                                        int num_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        // TODO: Load pair from a2[idx] and b2[idx]
        // Compute sum using float conversions for intermediate math:
        //   float2 a_f = __bfloat1622float2(a_val);
        //   float2 b_f = __bfloat1622float2(b_val);
        //   float2 sum = make_float2(a_f.x + b_f.x, a_f.y + b_f.y);
        //   out2[idx] = __float22bfloat162_rn(sum);
    }
}

// ============================================================================
// Exercise 2: Packed FMA: out = a * b + c
// Compute elementwise FMA on pairs using __nv_bfloat162.
// ============================================================================
__global__ void packed_bf162_fma_kernel(const __nv_bfloat162* a2,
                                        const __nv_bfloat162* b2,
                                        const __nv_bfloat162* c2,
                                        __nv_bfloat162* out2,
                                        int num_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        // TODO: Unpack a2[idx], b2[idx], c2[idx] to float2
        // Compute fma: res.x = a.x * b.x + c.x, res.y = a.y * b.y + c.y
        // Pack back to __nv_bfloat162 and store in out2[idx]
    }
}

// ============================================================================
// Exercise 3: Packed BF16 Processing with Odd Array Length Cleanup
// An array has length N (which may be odd). Process pairs up to N/2,
// and if N is odd, handle the last element gracefully.
// ============================================================================
__global__ void bf16_vector_scale_odd_kernel(const __nv_bfloat16* x,
                                             __nv_bfloat16* y,
                                             float scale,
                                             int N) {
    int num_pairs = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO:
    // 1. If idx < num_pairs, cast to __nv_bfloat162*, load pair, multiply both by scale, write to y as pair.
    // 2. If N is odd and idx == num_pairs (or thread 0 when idx == 0), handle the tail element x[N - 1].
}

// ============================================================================
// Exercise 4: Mixed-Precision Dot Product (Warp Shuffle Reduction)
// Compute the dot product of two BF16 vectors of length 32 (one warp).
// Accumulate products in float (FP32) and use __shfl_down_sync.
// Output the scalar float result to out_dot[0].
// ============================================================================
__global__ void warp_bf16_dot_kernel(const __nv_bfloat16* a,
                                     const __nv_bfloat16* b,
                                     float* out_dot) {
    int lane = threadIdx.x;
    // TODO:
    // 1. Load a[lane] and b[lane], convert to float, compute product.
    // 2. Perform warp reduction across all 32 lanes using __shfl_down_sync(FULL_MASK, val, offset).
    // 3. Lane 0 writes final sum to *out_dot.
}

// ============================================================================
// Exercise 5: BF16 GELU Activation (Approximate)
// GELU(x) = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)))
// Compute activation with FP32 math, convert output back to __nv_bfloat16.
// ============================================================================
__global__ void bf16_gelu_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Load in[idx], convert to float x.
        // Compute GELU in float precision.
        // Convert to __nv_bfloat16 and write to out[idx].
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

    // --- Test 1: Packed Vector Addition ---
    {
        int num_pairs = N / 2;
        std::vector<__nv_bfloat16> h_a(N), h_b(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_a[i] = __float2bfloat16(static_cast<float>(i) * 0.5f);
            h_b[i] = __float2bfloat16(10.0f);
        }

        __nv_bfloat16 *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (num_pairs + blockSize - 1) / blockSize;
        packed_bf162_add_kernel<<<blocks, blockSize>>>(
            reinterpret_cast<const __nv_bfloat162*>(d_a),
            reinterpret_cast<const __nv_bfloat162*>(d_b),
            reinterpret_cast<__nv_bfloat162*>(d_out),
            num_pairs);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = static_cast<float>(i) * 0.5f + 10.0f;
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp - act) > 0.05f * exp) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 1 Passed: Packed __nv_bfloat162 vector addition." << std::endl;
            passed++;
        } else {
            std::cout << "Test 1 Failed: Packed __nv_bfloat162 vector addition." << std::endl;
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    // --- Test 2: Packed FMA ---
    {
        int num_pairs = N / 2;
        std::vector<__nv_bfloat16> h_a(N), h_b(N), h_c(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            h_a[i] = __float2bfloat16(2.0f);
            h_b[i] = __float2bfloat16(3.5f);
            h_c[i] = __float2bfloat16(1.0f);
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

        int blocks = (num_pairs + blockSize - 1) / blockSize;
        packed_bf162_fma_kernel<<<blocks, blockSize>>>(
            reinterpret_cast<const __nv_bfloat162*>(d_a),
            reinterpret_cast<const __nv_bfloat162*>(d_b),
            reinterpret_cast<const __nv_bfloat162*>(d_c),
            reinterpret_cast<__nv_bfloat162*>(d_out),
            num_pairs);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = 2.0f * 3.5f + 1.0f; // 8.0f
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp - act) > 0.05f) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[0]) != 0.0f) {
            std::cout << "Test 2 Passed: Packed __nv_bfloat162 FMA." << std::endl;
            passed++;
        } else {
            std::cout << "Test 2 Failed: Packed __nv_bfloat162 FMA." << std::endl;
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    }

    // --- Test 3: Odd Array Length Tail Cleanup ---
    {
        const int odd_N = 1025; // odd!
        std::vector<__nv_bfloat16> h_x(odd_N), h_y(odd_N);
        float scale = 3.0f;
        for (int i = 0; i < odd_N; ++i) h_x[i] = __float2bfloat16(static_cast<float>(i + 1));

        __nv_bfloat16 *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, odd_N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_y, odd_N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), odd_N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0, odd_N * sizeof(__nv_bfloat16)));

        int blocks = (odd_N + blockSize - 1) / blockSize;
        bf16_vector_scale_odd_kernel<<<blocks, blockSize>>>(d_x, d_y, scale, odd_N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, odd_N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < odd_N; ++i) {
            float exp = static_cast<float>(i + 1) * scale;
            float act = __bfloat162float(h_y[i]);
            if (std::abs(exp - act) > 0.05f * exp) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_y[odd_N - 1]) != 0.0f) {
            std::cout << "Test 3 Passed: Odd array length packed processing." << std::endl;
            passed++;
        } else {
            std::cout << "Test 3 Failed: Odd array length packed processing." << std::endl;
        }

        cudaFree(d_x); cudaFree(d_y);
    }

    // --- Test 4: Mixed-Precision Dot Product (Warp Shuffle) ---
    {
        std::vector<__nv_bfloat16> h_a(32), h_b(32);
        float expected_dot = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float va = static_cast<float>(i + 1) * 0.5f;
            float vb = 2.0f;
            h_a[i] = __float2bfloat16(va);
            h_b[i] = __float2bfloat16(vb);
            expected_dot += va * vb;
        }

        __nv_bfloat16 *d_a, *d_b;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_a, 32 * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b, 32 * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), 32 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), 32 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

        warp_bf16_dot_kernel<<<1, 32>>>(d_a, d_b, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_out = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        if (std::abs(h_out - expected_dot) < 0.05f * expected_dot && h_out != 0.0f) {
            std::cout << "Test 4 Passed: Mixed-precision warp dot product." << std::endl;
            passed++;
        } else {
            std::cout << "Test 4 Failed: Mixed-precision warp dot product (got " << h_out
                      << ", exp " << expected_dot << ")." << std::endl;
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    // --- Test 5: BF16 GELU Activation ---
    {
        std::vector<__nv_bfloat16> h_in(N), h_out(N);
        for (int i = 0; i < N; ++i) {
            float v = static_cast<float>(i - N / 2) * 0.01f;
            h_in[i] = __float2bfloat16(v);
        }

        __nv_bfloat16 *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(__nv_bfloat16)));

        int blocks = (N + blockSize - 1) / blockSize;
        bf16_gelu_kernel<<<blocks, blockSize>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float x = __bfloat162float(h_in[i]);
            float exp_gelu = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
            float act = __bfloat162float(h_out[i]);
            if (std::abs(exp_gelu - act) > 0.05f) { ok = false; break; }
        }
        if (ok && __bfloat162float(h_out[N - 1]) != 0.0f) {
            std::cout << "Test 5 Passed: BF16 GELU activation." << std::endl;
            passed++;
        } else {
            std::cout << "Test 5 Failed: BF16 GELU activation." << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total << " tests." << std::endl;
    return (passed == total) ? 0 : 1;
}
