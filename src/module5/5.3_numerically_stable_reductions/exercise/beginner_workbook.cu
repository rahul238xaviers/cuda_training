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

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// -----------------------------------------------------------------------------
// PROBLEM 1: Safe Max Subtraction
// -----------------------------------------------------------------------------
__global__ void kernel_subtract_max(const float *logits, float *out, float max_val, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < n) out[gid] = logits[gid] - max_val;
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Overflow-Free Exponentiation with Clamping
// -----------------------------------------------------------------------------
__global__ void kernel_safe_exp(const float *shifted_logits, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // shifted_logits are <= 0.0f.
    // Clamp to -80.0f to avoid subnormal float underflow.
    // TODO: if (gid < n) {
    //     float x = fmaxf(shifted_logits[gid], -80.0f);
    //     out[gid] = __expf(x);
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Stable 256-Element Block Softmax
// -----------------------------------------------------------------------------
__global__ void kernel_stable_softmax_256(const float *logits, float *probs, int n) {
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float z = (tid < n) ? logits[tid] : -1e30f;

    // TODO: 1. Find max across 256 threads, store in s_max
    // TODO: 2. Compute exp(z - s_max)
    // TODO: 3. Sum exponentials across 256 threads, store in s_sum_exp
    // TODO: 4. Write probs[tid] = exp_val / s_sum_exp;
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Softmax Cross-Entropy Loss Single-Row
// Loss = -logits[target] + max_val + log(sum(exp(logits - max_val)))
// -----------------------------------------------------------------------------
__global__ void kernel_cross_entropy_row(
    const float *logits,
    int target_class,
    float *out_loss,
    int n
) {
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Compute numerically stable cross entropy loss
    // Thread 0 writes out_loss = -logits[target_class] + s_max + logf(s_sum_exp);
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Temperature-Scaled Softmax (z_i / T)
// -----------------------------------------------------------------------------
__global__ void kernel_temperature_softmax(const float *logits, float *probs, float temperature, int n) {
    __shared__ float s_scratch[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Scale logit by 1.0f / temperature, then compute stable softmax
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.3 Numerically Stable Reductions (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Subtract Max
    {
        int n = 256;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_subtract_max<<<1, 256>>>(d_in, d_out, 255.0f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != (float)(i - 255)) { ok = false; break; }
        }
        reportStatus("Problem 1: Safe Max Subtraction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Safe Exp
    {
        int n = 256;
        std::vector<float> h_in(n, 0.0f), h_out(n, 0.0f);
        h_in[0] = -1000.0f; // Extremely negative, must not crash or produce subnormals
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_safe_exp<<<1, 256>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (h_out[1] == 1.0f && h_out[0] >= 0.0f && !std::isnan(h_out[0]));
        reportStatus("Problem 2: Overflow-Free Exponentiation with Clamping", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Stable Softmax
    {
        int n = 256;
        // Large logits that would overflow standard expf (e.g. 500.0f)
        std::vector<float> h_logits(n, 500.0f), h_probs(n, 0.0f);
        float *d_logits, *d_probs;
        cudaMalloc(&d_logits, n * sizeof(float));
        cudaMalloc(&d_probs, n * sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_stable_softmax_256<<<1, 256>>>(d_logits, d_probs, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_probs.data(), d_probs, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float sum = 0.0f;
        for (float p : h_probs) {
            sum += p;
            if (std::isnan(p) || std::isinf(p)) ok = false;
        }
        if (std::fabs(sum - 1.0f) > 1e-3f) ok = false;
        reportStatus("Problem 3: Stable 256-Element Block Softmax (Huge Logits)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_probs);
    }

    // Test 4: Cross-Entropy Loss
    {
        int n = 256, target = 42;
        std::vector<float> h_logits(n, 10.0f);
        h_logits[target] = 15.0f; // Target has logit 15.0, others 10.0
        float h_loss = 0.0f;
        float *d_logits, *d_loss;
        cudaMalloc(&d_logits, n * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_cross_entropy_row<<<1, 256>>>(d_logits, target, d_loss, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        // exp(15-15) = 1.0; 255 * exp(10-15) = 255 * e^-5 = 1.718
        // sum = 2.718; loss = -15 + 15 + log(2.718) = ~1.0
        bool ok = (!std::isnan(h_loss) && h_loss > 0.0f && h_loss < 5.0f);
        reportStatus("Problem 4: Softmax Cross-Entropy Loss Single-Row", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_loss);
    }

    // Test 5: Temperature-Scaled Softmax
    {
        int n = 256;
        std::vector<float> h_logits(n, 2.0f), h_probs(n, 0.0f);
        float *d_logits, *d_probs;
        cudaMalloc(&d_logits, n * sizeof(float));
        cudaMalloc(&d_probs, n * sizeof(float));
        cudaMemcpy(d_logits, h_logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_temperature_softmax<<<1, 256>>>(d_logits, d_probs, 0.5f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_probs.data(), d_probs, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float sum = 0.0f;
        for (float p : h_probs) sum += p;
        if (std::fabs(sum - 1.0f) > 1e-3f) ok = false;
        reportStatus("Problem 5: Temperature-Scaled Softmax (z_i / T)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_logits); cudaFree(d_probs);
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
