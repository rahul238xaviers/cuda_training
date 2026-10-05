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

// -----------------------------------------------------------------------------
// PROBLEM 1: Contention-Mitigated Loss Aggregator (1 atomicAdd per Block)
// -----------------------------------------------------------------------------
__global__ void kernel_contention_free_loss(const float *losses, float *global_loss, int n) {
    __shared__ float s_scratch[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float local_sum = 0.0f;
    // TODO: 1. Grid-stride loop accumulation into local_sum
    // TODO: 2. Warp reduction
    // TODO: 3. Scratchpad reduction
    // TODO: 4. Exactly one thread per block executes: atomicAdd(global_loss, block_sum)
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Shared-Memory Pre-Accumulated Histogram
// -----------------------------------------------------------------------------
__global__ void kernel_shared_preacc_hist(const int *in, int *out_hist, int num_bins, int n) {
    extern __shared__ int s_bins[];
    int tid = threadIdx.x;

    // TODO: 1. Clear s_bins to 0
    // TODO: 2. __syncthreads();
    // TODO: 3. Accumulate into s_bins[bin] using atomicAdd on shared memory
    // TODO: 4. __syncthreads();
    // TODO: 5. Write s_bins[tid] to global out_hist[tid] with atomicAdd
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Custom Floating-Point atomicMax using atomicCAS
// -----------------------------------------------------------------------------
__device__ inline float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    // TODO: while loop using atomicCAS
    // do {
    //     assumed = old;
    //     if (__int_as_float(assumed) >= val) break;
    //     old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    // } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_float_atomic_max(const float *in, float *global_max, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < n) atomicMaxFloat(global_max, in[gid]);
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Atomic Spin-Lock Critical Section
// -----------------------------------------------------------------------------
__global__ void kernel_spinlock_increment(int *shared_counter, int *mutex, int n) {
    // Each thread enters critical section, increments shared_counter, and exits
    // TODO: while (atomicExch(mutex, 1) != 0); // acquire lock
    // (*shared_counter)++;
    // atomicExch(mutex, 0); // release lock
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Scattered Token Gradient Row Accumulator
// -----------------------------------------------------------------------------
__global__ void kernel_scatter_token_grads(
    const int *token_ids,
    const float *grad_in,
    float *grad_embedding_table,
    int hidden_dim,
    int num_tokens
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // For every token t and dim d:
    // TODO: atomicAdd(&grad_embedding_table[token_id * hidden_dim + d], grad_in[t * hidden_dim + d]);
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.2 Atomic Operations (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Contention-Free Loss
    {
        int n = 50000;
        std::vector<float> h_losses(n, 0.25f);
        float *d_losses, *d_global_loss;
        cudaMalloc(&d_losses, n * sizeof(float));
        cudaMalloc(&d_global_loss, sizeof(float));
        cudaMemcpy(d_losses, h_losses.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_global_loss, 0, sizeof(float));

        kernel_contention_free_loss<<<8, 256>>>(d_losses, d_global_loss, n);
        cudaDeviceSynchronize();

        float h_loss = 0.0f;
        cudaMemcpy(&h_loss, d_global_loss, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_loss - (50000 * 0.25f)) < 1.0f);
        reportStatus("Problem 1: Contention-Mitigated Loss Aggregator", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_losses); cudaFree(d_global_loss);
    }

    // Test 2: Shared Pre-Accumulated Histogram
    {
        int n = 2048, num_bins = 16;
        std::vector<int> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = i % num_bins;
        int *d_in, *d_hist;
        cudaMalloc(&d_in, n * sizeof(int));
        cudaMalloc(&d_hist, num_bins * sizeof(int));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_hist, 0, num_bins * sizeof(int));

        kernel_shared_preacc_hist<<<2, 256, num_bins * sizeof(int)>>>(d_in, d_hist, num_bins, n);
        cudaDeviceSynchronize();

        std::vector<int> h_hist(num_bins);
        cudaMemcpy(h_hist.data(), d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int b = 0; b < num_bins; ++b) {
            if (h_hist[b] != (n / num_bins)) { ok = false; break; }
        }
        reportStatus("Problem 2: Shared-Memory Pre-Accumulated Histogram", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_hist);
    }

    // Test 3: Float atomicMax
    {
        int n = 500;
        std::vector<float> h_in(n);
        for (int i = 0; i < n; ++i) h_in[i] = (float)i * 0.5f; // max is 249.5
        float *d_in, *d_max;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        float init_val = -1000.0f;
        cudaMemcpy(d_max, &init_val, sizeof(float), cudaMemcpyHostToDevice);

        kernel_float_atomic_max<<<(n + 255) / 256, 256>>>(d_in, d_max, n);
        cudaDeviceSynchronize();

        float h_max = 0.0f;
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_max - 249.5f) < 1e-3f);
        reportStatus("Problem 3: Custom Floating-Point atomicMax using atomicCAS", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_max);
    }

    // Test 4: Spin-Lock
    {
        int n = 64; // Small thread group to test spinlock safely
        int *d_counter, *d_mutex;
        cudaMalloc(&d_counter, sizeof(int));
        cudaMalloc(&d_mutex, sizeof(int));
        cudaMemset(d_counter, 0, sizeof(int));
        cudaMemset(d_mutex, 0, sizeof(int));

        kernel_spinlock_increment<<<1, n>>>(d_counter, d_mutex, n);
        cudaDeviceSynchronize();

        int h_counter = 0;
        cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_counter == n);
        reportStatus("Problem 4: Atomic Spin-Lock Critical Section", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_counter); cudaFree(d_mutex);
    }

    // Test 5: Scattered Token Grads
    {
        int vocab_size = 10, hidden_dim = 16, num_tokens = 4;
        std::vector<int> h_tokens = {2, 5, 2, 8}; // Token 2 appears twice!
        std::vector<float> h_grads(num_tokens * hidden_dim, 1.0f);
        std::vector<float> h_table(vocab_size * hidden_dim, 0.0f);

        int *d_tokens;
        float *d_grads, *d_table;
        cudaMalloc(&d_tokens, num_tokens * sizeof(int));
        cudaMalloc(&d_grads, h_grads.size() * sizeof(float));
        cudaMalloc(&d_table, h_table.size() * sizeof(float));

        cudaMemcpy(d_tokens, h_tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grads, h_grads.data(), h_grads.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_table, 0, h_table.size() * sizeof(float));

        kernel_scatter_token_grads<<<1, num_tokens * hidden_dim>>>(d_tokens, d_grads, d_table, hidden_dim, num_tokens);
        cudaDeviceSynchronize();

        cudaMemcpy(h_table.data(), d_table, h_table.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        // Token 2 must accumulate 2.0f for all dimensions
        for (int d = 0; d < hidden_dim; ++d) {
            if (std::fabs(h_table[2 * hidden_dim + d] - 2.0f) > 1e-3f) ok = false;
            if (std::fabs(h_table[5 * hidden_dim + d] - 1.0f) > 1e-3f) ok = false;
        }
        reportStatus("Problem 5: Scattered Token Gradient Row Accumulator", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_tokens); cudaFree(d_grads); cudaFree(d_table);
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
