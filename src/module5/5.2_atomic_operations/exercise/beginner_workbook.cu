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
// PROBLEM 1: Global Integer Counter with atomicAdd
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_counter(int *counter, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: If gid < n, increment counter by 1 using atomicAdd
    // if (gid < n) atomicAdd(counter, 1);
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Global Floating-Point Loss Accumulation
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_loss_sum(const float *losses, float *total_loss, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: If gid < n, atomicAdd(total_loss, losses[gid]);
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Integer Global Min & Max Tracking
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_min_max(const int *in, int *out_min, int *out_max, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < n) {
    //     atomicMin(out_min, in[gid]);
    //     atomicMax(out_max, in[gid]);
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Atomic Exchange for Flag Grabbing
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_flag_grab(int *flag, int *grabber_thread_id) {
    // Exactly one thread will successfully swap flag from 0 to 1 and get old value 0!
    // TODO: int old_val = atomicExch(flag, 1);
    // if (old_val == 0) *grabber_thread_id = threadIdx.x;
}

// -----------------------------------------------------------------------------
// PROBLEM 5: 10-Class Histogram Binning with atomicAdd
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_histogram(const int *class_labels, int *hist_bins, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < n) {
    //     int c = class_labels[gid];
    //     atomicAdd(&hist_bins[c], 1);
    // }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.2 Atomic Operations (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Atomic Counter
    {
        int n = 10000;
        int *d_counter;
        cudaMalloc(&d_counter, sizeof(int));
        cudaMemset(d_counter, 0, sizeof(int));

        kernel_atomic_counter<<<(n + 255) / 256, 256>>>(d_counter, n);
        cudaDeviceSynchronize();

        int h_counter = 0;
        cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_counter == n);
        reportStatus("Problem 1: Global Integer Counter with atomicAdd", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_counter);
    }

    // Test 2: Atomic Loss Sum
    {
        int n = 1000;
        std::vector<float> h_losses(n, 0.5f);
        float *d_losses, *d_total;
        cudaMalloc(&d_losses, n * sizeof(float));
        cudaMalloc(&d_total, sizeof(float));
        cudaMemcpy(d_losses, h_losses.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_total, 0, sizeof(float));

        kernel_atomic_loss_sum<<<(n + 255) / 256, 256>>>(d_losses, d_total, n);
        cudaDeviceSynchronize();

        float h_total = 0.0f;
        cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_total - 500.0f) < 1e-2f);
        reportStatus("Problem 2: Global Floating-Point Loss Accumulation", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_losses); cudaFree(d_total);
    }

    // Test 3: Atomic Min Max
    {
        int n = 500;
        std::vector<int> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = i - 250; // min is -250, max is 249
        int *d_in, *d_min, *d_max;
        cudaMalloc(&d_in, n * sizeof(int));
        cudaMalloc(&d_min, sizeof(int));
        cudaMalloc(&d_max, sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        int init_min = 1000000, init_max = -1000000;
        cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice);

        kernel_atomic_min_max<<<(n + 255) / 256, 256>>>(d_in, d_min, d_max, n);
        cudaDeviceSynchronize();

        int h_min = 0, h_max = 0;
        cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

        bool ok = (h_min == -250 && h_max == 249);
        reportStatus("Problem 3: Integer Global Min & Max Tracking", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_min); cudaFree(d_max);
    }

    // Test 4: Atomic Exchange Flag
    {
        int *d_flag, *d_grabber;
        cudaMalloc(&d_flag, sizeof(int));
        cudaMalloc(&d_grabber, sizeof(int));
        cudaMemset(d_flag, 0, sizeof(int));
        int init_grabber = -1;
        cudaMemcpy(d_grabber, &init_grabber, sizeof(int), cudaMemcpyHostToDevice);

        kernel_atomic_flag_grab<<<1, 64>>>(d_flag, d_grabber);
        cudaDeviceSynchronize();

        int h_grabber = -1;
        cudaMemcpy(&h_grabber, d_grabber, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_grabber >= 0 && h_grabber < 64);
        reportStatus("Problem 4: Atomic Exchange for Flag Grabbing", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_flag); cudaFree(d_grabber);
    }

    // Test 5: Histogram Binning
    {
        int n = 1000, num_classes = 10;
        std::vector<int> h_labels(n);
        for (int i = 0; i < n; ++i) h_labels[i] = i % num_classes; // 100 per class
        int *d_labels, *d_bins;
        cudaMalloc(&d_labels, n * sizeof(int));
        cudaMalloc(&d_bins, num_classes * sizeof(int));
        cudaMemcpy(d_labels, h_labels.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_bins, 0, num_classes * sizeof(int));

        kernel_atomic_histogram<<<(n + 255) / 256, 256>>>(d_labels, d_bins, n);
        cudaDeviceSynchronize();

        std::vector<int> h_bins(num_classes);
        cudaMemcpy(h_bins.data(), d_bins, num_classes * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int c = 0; c < num_classes; ++c) {
            if (h_bins[c] != 100) { ok = false; break; }
        }
        reportStatus("Problem 5: 10-Class Histogram Binning with atomicAdd", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_labels); cudaFree(d_bins);
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
