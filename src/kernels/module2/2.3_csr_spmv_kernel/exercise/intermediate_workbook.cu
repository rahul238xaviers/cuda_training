// 2.3_csr_spmv_kernel - CSR / COO Sparse Matrix-Vector Multiply (SpMV)
// Exercise Workbook (Intermediate Tier: Shared Memory & Warp-Tiled Implementation): Implement the kernel under evaluation.
// Compile: nvcc -O3 -std=c++17 --extended-lambda workbook.cu -o wb && ./wb

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// [KERNEL SPECIFICATION & PROBLEM OBJECTIVE]
// -----------------------------------------------------------------------------
// Title: CSR / COO Sparse Matrix-Vector Multiply (SpMV) (Sparse Computations)
// Mathematical Equation: y = A_sparse * x where A is stored in CSR (row_ptr, col_indices, values)
// Hardware Goal: Compressed Sparse Row format, row-balancing, and vector dot-products
//
// [EVALUATION CRITERIA]:
// 1. Numerical Correctness: Max absolute error vs CPU ground truth must be < 1e-4.
// 2. Hardware Saturation: Achieved memory bandwidth and TFLOPs reported.
//
// [PITFALLS TO AVOID]:
// - Ensure coalesced 128-byte DRAM transactions.
// - Avoid warp divergence across the 32 threads in each lockstep warp.
// - Avoid SRAM bank conflicts on shared memory indices.
// =============================================================================

// --- CPU Ground-Truth Reference Implementation ---
void cpu_reference(const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y, int num_rows) {
  for (int r = 0; r < num_rows; ++r) {
    float sum = 0.0f;
    for (int i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
      sum += values[i] * x[col_indices[i]];
    }
    y[r] = sum;
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void csr_spmv_kernel(const int* __restrict__ row_ptr, const int* __restrict__ col_indices, const float* __restrict__ values, const float* __restrict__ x, float* __restrict__ y, int num_rows) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: CSR / COO Sparse Matrix-Vector Multiply (SpMV)" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int num_rows = 1024;
  dim3 block(256);
  dim3 grid((num_rows + 255) / 256);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
