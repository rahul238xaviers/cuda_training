// 4.4_numerically_stable_softmax - Numerically Stable Block-Wide Online Softmax
// Exercise Workbook (Beginner Tier: Direct Naive / Baseline Implementation): Implement the kernel under evaluation.
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
// Title: Numerically Stable Block-Wide Online Softmax (Activations)
// Mathematical Equation: p_i = exp(x_i - max(x)) / sum(exp(x - max(x))) in 1 unified block pass
// Hardware Goal: Single-pass streaming online softmax without materializing global maximum
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
void cpu_reference(const float* x, float* p, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float m = -1e9f;
    for (int c = 0; c < cols; ++c) if (x[r * cols + c] > m) m = x[r * cols + c];
    float l = 0.0f;
    for (int c = 0; c < cols; ++c) l += std::exp(x[r * cols + c] - m);
    for (int c = 0; c < cols; ++c) p[r * cols + c] = std::exp(x[r * cols + c] - m) / l;
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void online_softmax_kernel(const float* __restrict__ x, float* __restrict__ p, int rows, int cols) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Numerically Stable Block-Wide Online Softmax" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int rows = 512;
  const int cols = 2048;
  dim3 block(256);
  dim3 grid(rows);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
