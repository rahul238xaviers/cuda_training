// 1.1_dense_gemv - Fused GEMV Matrix-Vector Dot Product
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
// Title: Fused GEMV Matrix-Vector Dot Product (Dense Linear)
// Mathematical Equation: y = alpha * A * x + beta * y where A is (M x K) and x is (K x 1)
// Hardware Goal: Coalesced 128-byte transactions, vectorized float4 loads, warp shuffle reductions
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
void cpu_reference(const float* A, const float* x, float* y, int M, int K, float alpha, float beta) {
  for (int r = 0; r < M; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < K; ++c) {
      sum += A[r * K + c] * x[c];
    }
    y[r] = alpha * sum + beta * y[r];
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void fused_gemv_kernel(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int K, float alpha, float beta) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Fused GEMV Matrix-Vector Dot Product" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int M = 1024;
  const int K = 2048;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  dim3 block(32, 8);
  dim3 grid((M + block.y - 1) / block.y);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
