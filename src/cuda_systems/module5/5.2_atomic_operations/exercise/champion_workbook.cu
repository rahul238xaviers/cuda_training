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
// PROBLEM 1: Lock-Free Atomic Ring Queue (Pushing work indices)
// -----------------------------------------------------------------------------
__global__ void kernel_atomic_ring_enqueue(
    const int *in_items,
    int *queue_buffer,
    int *head_idx,
    int queue_capacity,
    int num_items
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < num_items) {
    //     int slot = atomicAdd(head_idx, 1) % queue_capacity;
    //     queue_buffer[slot] = in_items[gid];
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 2: In-Place AdamW Optimizer Parameter Step
// W[i] = W[i] - lr * (weight_decay * W[i] + m_hat / (sqrt(v_hat) + eps))
// -----------------------------------------------------------------------------
__global__ void kernel_adamw_step(
    float *param,
    const float *grad,
    float *exp_avg,     // m
    float *exp_avg_sq,  // v
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float bias_correction1,
    float bias_correction2,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Implement vectorized/scalar AdamW update in registers
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Warp-Aggregated Atomics using __match_any_sync or Intra-Warp Reductions
// When multiple threads in a warp target the SAME index, reduce within warp first
// so only 1 thread issues the atomic instruction!
// -----------------------------------------------------------------------------
__global__ void kernel_warp_aggregated_atomic_add(
    const int *keys,
    const float *values,
    float *table,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Implement warp-aggregated atomic add
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Sparse Matrix Coordinate (COO) Vector Product Accumulator
// y[row[k]] += val[k] * x[col[k]]
// -----------------------------------------------------------------------------
__global__ void kernel_spmv_coo_atomic(
    const int *row_indices,
    const int *col_indices,
    const float *values,
    const float *x,
    float *y,
    int nnz
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < nnz) {
    //     float prod = values[gid] * x[col_indices[gid]];
    //     atomicAdd(&y[row_indices[gid]], prod);
    // }
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Exact Reproducible Sum via 64-bit Fixed-Point Atomics
// -----------------------------------------------------------------------------
__global__ void kernel_fixed_point_atomic_sum(const float *in, long long *fixed_sum, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Scale float by 1,000,000 to fixed-point integer, and atomicAdd to long long
    // TODO: if (gid < n) {
    //     long long int fixed_val = (long long int)(in[gid] * 1000000.0f);
    //     atomicAdd((unsigned long long int*)fixed_sum, (unsigned long long int)fixed_val);
    // }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 5.2 Atomic Operations (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Atomic Ring Queue
    {
        int num_items = 512, capacity = 1024;
        std::vector<int> h_items(num_items);
        for (int i = 0; i < num_items; ++i) h_items[i] = i + 100;

        int *d_items, *d_queue, *d_head;
        cudaMalloc(&d_items, num_items * sizeof(int));
        cudaMalloc(&d_queue, capacity * sizeof(int));
        cudaMalloc(&d_head, sizeof(int));

        cudaMemcpy(d_items, h_items.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_head, 0, sizeof(int));
        cudaMemset(d_queue, 0, capacity * sizeof(int));

        kernel_atomic_ring_enqueue<<<2, 256>>>(d_items, d_queue, d_head, capacity, num_items);
        cudaDeviceSynchronize();

        int h_head = 0;
        cudaMemcpy(&h_head, d_head, sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (h_head == num_items);
        reportStatus("Problem 1: Lock-Free Atomic Ring Queue", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_items); cudaFree(d_queue); cudaFree(d_head);
    }

    // Test 2: AdamW Step
    {
        int n = 256;
        std::vector<float> h_param(n, 1.0f), h_grad(n, 0.1f), h_m(n, 0.0f), h_v(n, 0.0f);
        float *d_param, *d_grad, *d_m, *d_v;
        cudaMalloc(&d_param, n * sizeof(float));
        cudaMalloc(&d_grad, n * sizeof(float));
        cudaMalloc(&d_m, n * sizeof(float));
        cudaMalloc(&d_v, n * sizeof(float));

        cudaMemcpy(d_param, h_param.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad, h_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_m, 0, n * sizeof(float));
        cudaMemset(d_v, 0, n * sizeof(float));

        kernel_adamw_step<<<1, 256>>>(d_param, d_grad, d_m, d_v, 0.9f, 0.999f, 1e-3f, 0.01f, 1e-8f, 0.1f, 0.001f, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_param.data(), d_param, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            // Parameter must decrease from 1.0f
            if (h_param[i] >= 1.0f || std::isnan(h_param[i])) { ok = false; break; }
        }
        reportStatus("Problem 2: In-Place AdamW Optimizer Parameter Step", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_param); cudaFree(d_grad); cudaFree(d_m); cudaFree(d_v);
    }

    // Test 3: Warp Aggregated Atomics
    {
        int n = 512;
        std::vector<int> h_keys(n, 0); // All 512 threads target key 0!
        std::vector<float> h_vals(n, 1.0f);
        float *d_table;
        int *d_keys;
        float *d_vals;
        cudaMalloc(&d_table, 10 * sizeof(float));
        cudaMalloc(&d_keys, n * sizeof(int));
        cudaMalloc(&d_vals, n * sizeof(float));

        cudaMemset(d_table, 0, 10 * sizeof(float));
        cudaMemcpy(d_keys, h_keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, h_vals.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_warp_aggregated_atomic_add<<<2, 256>>>(d_keys, d_vals, d_table, n);
        cudaDeviceSynchronize();

        float h_res = 0.0f;
        cudaMemcpy(&h_res, d_table, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (std::fabs(h_res - 512.0f) < 1.0f);
        reportStatus("Problem 3: Warp-Aggregated Atomics (Intra-Warp Coalescing)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_table); cudaFree(d_keys); cudaFree(d_vals);
    }

    // Test 4: SpMV COO
    {
        int nnz = 100, num_rows = 10;
        std::vector<int> h_rows(nnz), h_cols(nnz);
        std::vector<float> h_vals(nnz, 2.0f), h_x(num_rows, 3.0f);
        for (int i = 0; i < nnz; ++i) {
            h_rows[i] = i % num_rows;
            h_cols[i] = (i * 3) % num_rows;
        }
        std::vector<float> h_y(num_rows, 0.0f);

        int *d_r, *d_c;
        float *d_v, *d_x, *d_y;
        cudaMalloc(&d_r, nnz * sizeof(int));
        cudaMalloc(&d_c, nnz * sizeof(int));
        cudaMalloc(&d_v, nnz * sizeof(float));
        cudaMalloc(&d_x, num_rows * sizeof(float));
        cudaMalloc(&d_y, num_rows * sizeof(float));

        cudaMemcpy(d_r, h_rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, num_rows * sizeof(float));

        kernel_spmv_coo_atomic<<<1, 128>>>(d_r, d_c, d_v, d_x, d_y, nnz);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float total_y = 0.0f;
        for (float v : h_y) total_y += v;
        // Total sum must be nnz * (2.0 * 3.0) = 600.0f
        if (std::fabs(total_y - 600.0f) > 1e-2f) ok = false;
        reportStatus("Problem 4: Sparse Matrix Coordinate (COO) Vector Accumulator", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_r); cudaFree(d_c); cudaFree(d_v); cudaFree(d_x); cudaFree(d_y);
    }

    // Test 5: Fixed Point Atomic
    {
        int n = 1000;
        std::vector<float> h_in(n, 1.25f); // 1.25 * 1000 = 1250.0
        long long *d_fixed;
        float *d_in;
        cudaMalloc(&d_fixed, sizeof(long long));
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMemset(d_fixed, 0, sizeof(long long));
        cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        kernel_fixed_point_atomic_sum<<<4, 256>>>(d_in, d_fixed, n);
        cudaDeviceSynchronize();

        long long h_fixed = 0;
        cudaMemcpy(&h_fixed, d_fixed, sizeof(long long), cudaMemcpyDeviceToHost);
        double reconstructed = (double)h_fixed / 1000000.0;
        bool ok = (std::fabs(reconstructed - 1250.0) < 1e-3);
        reportStatus("Problem 5: Exact Reproducible Sum via 64-bit Fixed-Point Atomics", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_fixed); cudaFree(d_in);
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
