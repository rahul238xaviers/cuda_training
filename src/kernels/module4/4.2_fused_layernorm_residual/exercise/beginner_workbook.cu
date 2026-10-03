// 4.2_fused_layernorm_residual - Fused LayerNorm with Residual Add & Bias
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
// Title: Fused LayerNorm with Residual Add & Bias (Normalizations)
// Mathematical Equation: x_res = x + res; y = ((x_res - mean) / sqrt(var + eps)) * gamma + beta
// Hardware Goal: Three-in-one kernel: residual addition, mean/variance calculation, and normalization
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
void cpu_reference(float* x, const float* res, const float* gamma, const float* beta, float* y, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
      x[r * cols + c] += res[r * cols + c];
      sum += x[r * cols + c];
      sum_sq += x[r * cols + c] * x[r * cols + c];
    }
    float mean = sum / cols;
    float var = std::max(0.0f, (sum_sq / cols) - (mean * mean));
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) y[r * cols + c] = (x[r * cols + c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void fused_layernorm_residual_kernel(float* __restrict__ x, const float* __restrict__ res, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ y, int rows, int cols, float eps) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Fused LayerNorm with Residual Add & Bias" << std::endl;
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
  const float eps = 1e-5f;
  dim3 block(256);
  dim3 grid(rows);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
