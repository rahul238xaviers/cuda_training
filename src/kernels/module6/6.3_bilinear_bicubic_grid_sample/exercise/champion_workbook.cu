// 6.3_bilinear_bicubic_grid_sample - Bilinear & Bicubic Grid Sampling for Feature Maps
// Exercise Workbook (Champion Tier: Production Vectorized & Register-Blocked Implementation): Implement the kernel under evaluation.
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
// Title: Bilinear & Bicubic Grid Sampling for Feature Maps (Vision & Video)
// Mathematical Equation: y[r, c] = sum_{i, j} w_ij * x[floor(r_in) + i, floor(c_in) + j]
// Hardware Goal: Differentiable spatial transformations, optical flow warping, and video frame interpolation
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
void cpu_reference(const float* input, const float* grid, float* output, int height, int width, int out_h, int out_w) {
  for (int r = 0; r < out_h; ++r) {
    for (int c = 0; c < out_w; ++c) {
      float u = grid[(r * out_w + c) * 2 + 0];
      float v = grid[(r * out_w + c) * 2 + 1];
      float in_x = ((u + 1.0f) * 0.5f) * (width - 1);
      float in_y = ((v + 1.0f) * 0.5f) * (height - 1);
      int x0 = std::floor(in_x), y0 = std::floor(in_y);
      int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
      x0 = std::max(0, x0); y0 = std::max(0, y0);
      float dx = in_x - x0, dy = in_y - y0;
      float p00 = input[y0 * width + x0], p01 = input[y0 * width + x1];
      float p10 = input[y1 * width + x0], p11 = input[y1 * width + x1];
      output[r * out_w + c] = (1.0f - dx) * (1.0f - dy) * p00 + dx * (1.0f - dy) * p01 + (1.0f - dx) * dy * p10 + dx * dy * p11;
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void bilinear_grid_sample_kernel(const float* __restrict__ input, const float* __restrict__ grid_coords, float* __restrict__ output, int height, int width, int out_h, int out_w) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Bilinear & Bicubic Grid Sampling for Feature Maps" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int height = 128;
  const int width = 128;
  const int out_h = 128;
  const int out_w = 128;
  dim3 block(16, 16);
  dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
