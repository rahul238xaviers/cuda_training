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
// PROBLEM 1: Lane ID and Warp ID Identification
// -----------------------------------------------------------------------------
__global__ void kernel_lane_warp_ids(int *out_lane, int *out_warp, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // TODO: Compute lane_id (0..31) and warp_id
        // out_lane[gid] = threadIdx.x % 32;
        // out_warp[gid] = threadIdx.x / 32;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Warp Broadcast with __shfl_sync
// -----------------------------------------------------------------------------
__global__ void kernel_warp_broadcast(const float *in_lane0, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float val = in_lane0[gid];
        // TODO: Broadcast the value held by lane 0 of each warp to all other lanes
        // float bcast = __shfl_sync(0xffffffff, val, 0);
        // out[gid] = bcast;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Neighbor Pair Sum with __shfl_down_sync
// -----------------------------------------------------------------------------
__global__ void kernel_neighbor_sum(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float val = in[gid];
        // TODO: Use __shfl_down_sync(0xffffffff, val, 1) to get neighbor's value
        // TODO: Compute out[gid] = val + neighbor
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Complete 32-Lane Warp Sum Reduction
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    // TODO: Implement 5-step reduction tree with offsets 16, 8, 4, 2, 1
    // val += __shfl_down_sync(0xffffffff, val, 16);
    // val += __shfl_down_sync(0xffffffff, val, 8);
    // val += __shfl_down_sync(0xffffffff, val, 4);
    // val += __shfl_down_sync(0xffffffff, val, 2);
    // val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void kernel_warp_reduce(const float *in, float *out_warp_sums, int num_warps) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid];
    float sum = warp_reduce_sum(val);
    int lane = threadIdx.x % 32;
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane == 0 && warp_idx < num_warps) {
        out_warp_sums[warp_idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: All-Reduce across Warp with __shfl_xor_sync
// -----------------------------------------------------------------------------
__global__ void kernel_warp_allreduce(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float val = in[gid];
        // TODO: Butterfly reduction using XOR: offsets 16, 8, 4, 2, 1
        // val += __shfl_xor_sync(0xffffffff, val, 16);
        // val += __shfl_xor_sync(0xffffffff, val, 8);
        // val += __shfl_xor_sync(0xffffffff, val, 4);
        // val += __shfl_xor_sync(0xffffffff, val, 2);
        // val += __shfl_xor_sync(0xffffffff, val, 1);
        // out[gid] = val; // Every lane now holds the sum!
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 3.2 Warps & Shuffles (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Lane and Warp IDs
    {
        int n = 128;
        std::vector<int> h_lane(n, 0), h_warp(n, 0);
        int *d_lane, *d_warp;
        cudaMalloc(&d_lane, n * sizeof(int));
        cudaMalloc(&d_warp, n * sizeof(int));

        kernel_lane_warp_ids<<<1, 128>>>(d_lane, d_warp, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_lane.data(), d_lane, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_warp.data(), d_warp, n * sizeof(int), cudaMemcpyDeviceToHost);

        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (h_lane[i] != (i % 32) || h_warp[i] != (i / 32)) { ok = false; break; }
        }
        reportStatus("Problem 1: Lane ID and Warp ID Identification", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_lane); cudaFree(d_warp);
    }

    // Test 2: Warp Broadcast
    {
        int n = 64; // 2 warps
        std::vector<float> h_in(n, 0.0f), h_out(n, 0.0f);
        h_in[0] = 42.0f; // lane 0 of warp 0
        h_in[32] = 99.0f; // lane 0 of warp 1
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_broadcast<<<1, 64>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 32; ++i) if (h_out[i] != 42.0f) ok = false;
        for (int i = 32; i < 64; ++i) if (h_out[i] != 99.0f) ok = false;

        reportStatus("Problem 2: Warp Broadcast with __shfl_sync", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 3: Neighbor Pair Sum
    {
        int n = 32;
        std::vector<float> h_in(n), h_out(n, 0.0f);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_neighbor_sum<<<1, 32>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 31; ++i) {
            if (std::fabs(h_out[i] - (h_in[i] + h_in[i + 1])) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 3: Neighbor Pair Sum with __shfl_down_sync", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 4: 32-Lane Warp Sum Reduction
    {
        int num_warps = 4;
        int n = num_warps * 32;
        std::vector<float> h_in(n, 1.0f);
        std::vector<float> h_out(num_warps, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, num_warps * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_reduce<<<1, 128>>>(d_in, d_out, num_warps);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, num_warps * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int w = 0; w < num_warps; ++w) {
            if (std::fabs(h_out[w] - 32.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 4: Complete 32-Lane Warp Sum Reduction", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_out);
    }

    // Test 5: All-Reduce with XOR Butterfly
    {
        int n = 32;
        std::vector<float> h_in(n, 2.0f);
        std::vector<float> h_out(n, 0.0f);
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_allreduce<<<1, 32>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - 64.0f) > 1e-4f) { ok = false; break; }
        }
        reportStatus("Problem 5: All-Reduce across Warp with __shfl_xor_sync", ok);
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
