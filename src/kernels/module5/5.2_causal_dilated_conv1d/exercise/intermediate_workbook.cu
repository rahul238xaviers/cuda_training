// 5.2_causal_dilated_conv1d - 1D Causal Dilated Convolution for Waveforms
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
// Title: 1D Causal Dilated Convolution for Waveforms (Audio Processing)
// Mathematical Equation: y[t] = sum_k(weight[k] * x[t - k * dilation]) with zero future leakage
// Hardware Goal: WaveNet / AudioGen causal convolutions with exponential dilation and residual connections
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
void cpu_reference(const float* x, const float* weight, float* y, int seq_len, int kernel_size, int dilation) {
  for (int t = 0; t < seq_len; ++t) {
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
      int pt = t - k * dilation;
      if (pt >= 0) sum += weight[k] * x[pt];
    }
    y[t] = sum;
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void causal_dilated_conv1d_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y, int seq_len, int kernel_size, int dilation) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: 1D Causal Dilated Convolution for Waveforms" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int seq_len = 1024 * 16;
  const int kernel_size = 3;
  const int dilation = 4;
  dim3 block(256);
  dim3 grid((seq_len + 255) / 256);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
