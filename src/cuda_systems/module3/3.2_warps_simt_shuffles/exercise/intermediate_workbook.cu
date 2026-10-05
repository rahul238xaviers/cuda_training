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
// PROBLEM 1: Warp-Level Maximum Reduction (for Softmax Normalizer)
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_max(float val) {
    // TODO: Reduce across warp using fmaxf and __shfl_down_sync (offsets 16, 8, 4, 2, 1)
    // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

__global__ void kernel_warp_max(const float *in, float *out_max, int num_warps) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid];
    float max_val = warp_reduce_max(val);
    int lane = threadIdx.x % 32;
    int warp_idx = gid / 32;
    if (lane == 0 && warp_idx < num_warps) {
        out_max[warp_idx] = max_val;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Inclusive Warp Prefix Scan using __shfl_up_sync
// -----------------------------------------------------------------------------
__device__ inline float warp_inclusive_scan(float val) {
    int lane = threadIdx.x % 32;
    // TODO: Compute prefix sum where thread i gets sum of elements [0..i]
    // for (int offset = 1; offset < 32; offset *= 2) {
    //     float n = __shfl_up_sync(0xffffffff, val, offset);
    //     if (lane >= offset) val += n;
    // }
    return val;
}

__global__ void kernel_inclusive_scan(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float val = in[gid];
        out[gid] = warp_inclusive_scan(val);
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Exclusive Warp Prefix Scan
// -----------------------------------------------------------------------------
__device__ inline float warp_exclusive_scan(float val) {
    // Exclusive scan: element i gets sum of elements [0..i-1], thread 0 gets 0
    // TODO: Perform inclusive scan, then shift right by 1 lane using __shfl_up_sync(..., 1)
    // float inc = warp_inclusive_scan(val);
    // int lane = threadIdx.x % 32;
    // float exc = __shfl_up_sync(0xffffffff, inc, 1);
    // return (lane == 0) ? 0.0f : exc;
    return 0.0f;
}

__global__ void kernel_exclusive_scan(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = warp_exclusive_scan(in[gid]);
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Warp Ballot & Population Count (__ballot_sync, __popc)
// -----------------------------------------------------------------------------
__global__ void kernel_count_positives(const float *in, int *out_count, int num_warps) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid];
    // TODO: Create a bitmask of threads where val > 0 using __ballot_sync(0xffffffff, predicate)
    // unsigned mask = __ballot_sync(0xffffffff, val > 0.0f);
    // int count = __popc(mask); // Count number of 1-bits
    // int lane = threadIdx.x % 32;
    // if (lane == 0) out_count[gid / 32] = count;
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Vectorized Pair Shuffle (2 floats in float2)
// -----------------------------------------------------------------------------
__global__ void kernel_pair_shuffle(const float2 *in, float2 *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float2 val = in[gid];
        // TODO: Shuffle float2 across adjacent lanes using __shfl_xor_sync with mask 1
        // float2 swapped;
        // swapped.x = __shfl_xor_sync(0xffffffff, val.x, 1);
        // swapped.y = __shfl_xor_sync(0xffffffff, val.y, 1);
        // out[gid] = swapped;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.2 Warps & Shuffles (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Warp Max
    {
        int num_warps = 2;
        int n = num_warps * 32;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 32);
        h_in[15] = 999.0f; // max in warp 0
        h_in[45] = 555.0f; // max in warp 1

        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, num_warps * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_max<<<1, 64>>>(d_in, d_out, num_warps);
        cudaDeviceSynchronize();

        std::vector<float> h_out(num_warps);
        cudaMemcpy(h_out.data(), d_out, num_warps * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (h_out[0] == 999.0f && h_out[1] == 555.0f);
        reportStatus("Problem 1: Warp-Level Maximum Reduction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 2: Inclusive Scan
    {
        int n = 32;
        std::vector<float> h_in(n, 1.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_inclusive_scan<<<1, 32>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (float)(i + 1)) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 2: Inclusive Warp Prefix Scan", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Exclusive Scan
    {
        int n = 32;
        std::vector<float> h_in(n, 1.0f), h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_exclusive_scan<<<1, 32>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (float)i) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: Exclusive Warp Prefix Scan", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: Warp Ballot & Popcount
    {
        int num_warps = 1;
        int n = 32;
        std::vector<float> h_in(n);
        int expected_count = 0;
        for (int i = 0; i < n; ++i) {
            h_in[i] = (i % 3 == 0) ? 5.0f : -1.0f;
            if (h_in[i] > 0.0f) expected_count++;
        }
        float *d_in;
        int *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_count_positives<<<1, 32>>>(d_in, d_out, num_warps);
        cudaDeviceSynchronize();

        int h_count = 0;
        cudaMemcpy(&h_count, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_count == expected_count);
        reportStatus("Problem 4: Warp Ballot & Population Count", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: Pair Shuffle
    {
        int n = 32;
        std::vector<float2> h_in(n), h_out(n);
        for (int i = 0; i < n; ++i) {
            h_in[i].x = (float)i;
            h_in[i].y = (float)(i * 10);
        }
        float2 *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float2));
        cudaMalloc(&d_out, n * sizeof(float2));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float2), cudaMemcpyHostToDevice);

        kernel_pair_shuffle<<<1, 32>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float2), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            int target_idx = i ^ 1;
            if (h_out[i].x != h_in[target_idx].x || h_out[i].y != h_in[target_idx].y) {
                ok = false; break;
            }
        }
        reportStatus("Problem 5: Vectorized Pair Shuffle", ok);
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
