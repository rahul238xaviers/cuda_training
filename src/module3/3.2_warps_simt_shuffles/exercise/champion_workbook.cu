#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

void reportStatus(const std::string &name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 1: In-Warp Numerically Stable Softmax (32 Elements)
// Softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
// -----------------------------------------------------------------------------
__global__ void kernel_warp_softmax(const float *in, float *out) {
    int lane = threadIdx.x % 32;
    float val = in[lane];

    // TODO: Step 1: Compute warp-level maximum using __shfl_down_sync
    // float max_val = val;
    // for (int offset = 16; offset > 0; offset /= 2) {
    //     max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    // }
    // max_val = __shfl_sync(0xffffffff, max_val, 0); // broadcast max to all threads

    // TODO: Step 2: Compute exp(val - max_val) and compute sum across warp
    // float exp_val = __expf(val - max_val);
    // float sum_exp = exp_val;
    // for (int offset = 16; offset > 0; offset /= 2) {
    //     sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    // }
    // sum_exp = __shfl_sync(0xffffffff, sum_exp, 0); // broadcast sum to all threads

    // TODO: Step 3: Write out[lane] = exp_val / sum_exp
}

// -----------------------------------------------------------------------------
// PROBLEM 2: 32-Element Warp Bitonic Sort using __shfl_xor_sync
// -----------------------------------------------------------------------------
__global__ void kernel_warp_bitonic_sort(const float *in, float *out) {
    int lane = threadIdx.x % 32;
    float val = in[lane];

    // TODO: Implement 32-element Bitonic sort network using __shfl_xor_sync
    // for (int size = 2; size <= 32; size *= 2) {
    //     for (int stride = size / 2; stride > 0; stride /= 2) {
    //         float other = __shfl_xor_sync(0xffffffff, val, stride);
    //         bool dir = ((lane & size) != 0);
    //         if (((lane & stride) == 0) == (val > other == dir)) {
    //             val = other;
    //         }
    //     }
    // }
    // out[lane] = val;
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Sub-Warp Independent Reductions (Two 16-thread Groups)
// -----------------------------------------------------------------------------
__global__ void kernel_subwarp_reduce(const float *in, float *out_subsums) {
    int lane = threadIdx.x % 32;
    float val = in[lane];
    // TODO: Reduce lanes [0..15] into lane 0, and lanes [16..31] into lane 16
    // Using delta offsets 8, 4, 2, 1
    // for (int offset = 8; offset > 0; offset /= 2) {
    //     val += __shfl_down_sync(0xffffffff, val, offset);
    // }
    // if (lane == 0) out_subsums[0] = val;
    // if (lane == 16) out_subsums[1] = val;
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Warp Dot Product of Two 32-Element Vectors
// -----------------------------------------------------------------------------
__global__ void kernel_warp_dot_product(const float *a, const float *b, float *out_dot) {
    int lane = threadIdx.x % 32;
    // TODO: Compute local product = a[lane] * b[lane]
    // TODO: Reduce across warp in 5 cycles with __shfl_down_sync
    // TODO: If lane 0, write to *out_dot
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Warp Stream Compaction with __ballot_sync and __popc
// Keep all elements where val > 0.0f and write them contiguously
// -----------------------------------------------------------------------------
__global__ void kernel_warp_compact(const float *in, float *out_compact, int *out_total) {
    int lane = threadIdx.x % 32;
    float val = in[lane];
    bool keep = (val > 0.0f);

    // TODO: 1. Generate active bitmask of kept elements
    // unsigned mask = __ballot_sync(0xffffffff, keep);
    // TODO: 2. Count active elements in lanes before this lane
    // unsigned lower_mask = (1u << lane) - 1;
    // int dest_idx = __popc(mask & lower_mask);
    // TODO: 3. If this lane is kept, write out_compact[dest_idx] = val;
    // TODO: 4. Lane 0 writes total count = __popc(mask) to *out_total
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.2 Warps & Shuffles (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: In-Warp Softmax
    {
        std::vector<float> h_in(32), h_out(32, 0.0f);
        for (int i = 0; i < 32; ++i) h_in[i] = (float)i * 0.5f;
        float *d_in, *d_out;
        cudaMalloc(&d_in, 32 * sizeof(float));
        cudaMalloc(&d_out, 32 * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_softmax<<<1, 32>>>(d_in, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);

        // Ground truth
        float max_val = *std::max_element(h_in.begin(), h_in.end());
        float sum = 0.0f;
        for (float v : h_in) sum += std::exp(v - max_val);
        bool ok = true;
        for (int i = 0; i < 32; ++i) {
            float expected = std::exp(h_in[i] - max_val) / sum;
            if (std::fabs(h_out[i] - expected) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 1: In-Warp Numerically Stable Softmax", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Bitonic Sort
    {
        std::vector<float> h_in = {
            12.f, 3.f, 19.f, 8.f, 25.f, 1.f, 14.f, 6.f,
            30.f, 2.f, 18.f, 9.f, 22.f, 4.f, 15.f, 7.f,
            28.f, 0.f, 17.f, 10.f, 21.f, 5.f, 13.f, 11.f,
            29.f, 16.f, 26.f, 20.f, 23.f, 24.f, 27.f, 31.f
        };
        std::vector<float> h_out(32, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, 32 * sizeof(float));
        cudaMalloc(&d_out, 32 * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_bitonic_sort<<<1, 32>>>(d_in, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 32; ++i) {
            if (h_out[i] != (float)i) { ok = false; break; }
        }
        reportStatus("Problem 2: 32-Element Warp Bitonic Sort with XOR Shuffles", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Sub-Warp Reductions
    {
        std::vector<float> h_in(32, 1.0f);
        for (int i = 16; i < 32; ++i) h_in[i] = 2.0f;
        std::vector<float> h_out(2, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, 32 * sizeof(float));
        cudaMalloc(&d_out, 2 * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_subwarp_reduce<<<1, 32>>>(d_in, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (h_out[0] == 16.0f && h_out[1] == 32.0f);
        reportStatus("Problem 3: Sub-Warp Independent Reductions (2x16 threads)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Warp Dot Product
    {
        std::vector<float> h_a(32, 2.0f), h_b(32, 3.0f);
        float h_out = 0.0f;
        float *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, 32 * sizeof(float));
        cudaMalloc(&d_b, 32 * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_a, h_a.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_dot_product<<<1, 32>>>(d_a, d_b, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_out - 192.0f) < 1e-4f);
        reportStatus("Problem 4: Warp Dot Product of Two 32-Element Vectors", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    // Test 5: Stream Compaction
    {
        std::vector<float> h_in = {
            -1.f, 4.f, -5.f, 2.f, -3.f, -7.f, 9.f, -2.f,
            11.f, -4.f, 6.f, -8.f, 1.f, -9.f, 3.f, -6.f,
            -1.f, 8.f, -5.f, 7.f, -3.f, -7.f, 5.f, -2.f,
            10.f, -4.f, 12.f, -8.f, 15.f, -9.f, 14.f, -6.f
        };
        std::vector<float> expected_positives;
        for (float v : h_in) if (v > 0.0f) expected_positives.push_back(v);

        std::vector<float> h_out(32, 0.0f);
        int h_total = 0;
        float *d_in, *d_out;
        int *d_total;
        cudaMalloc(&d_in, 32 * sizeof(float));
        cudaMalloc(&d_out, 32 * sizeof(float));
        cudaMalloc(&d_total, sizeof(int));
        cudaMemcpy(d_in, h_in.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_compact<<<1, 32>>>(d_in, d_out, d_total);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);

        bool ok = (h_total == (int)expected_positives.size());
        if (ok) {
            for (size_t i = 0; i < expected_positives.size(); ++i) {
                if (h_out[i] != expected_positives[i]) { ok = false; break; }
            }
        }
        reportStatus("Problem 5: Warp Stream Compaction with __ballot_sync and __popc", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_total);
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
