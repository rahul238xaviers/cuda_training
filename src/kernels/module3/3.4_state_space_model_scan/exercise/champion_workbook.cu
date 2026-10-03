// 3.4_state_space_model_scan - Linear Attention & State-Space Model (SSM) Scan
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
// Title: Linear Attention & State-Space Model (SSM) Scan (Sequence Models)
// Mathematical Equation: h_t = a_t * h_{t-1} + b_t * x_t using parallel prefix associative scan
// Hardware Goal: Mamba-style associative parallel scan over recurrent hidden state
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
void cpu_reference(const float* a, const float* b, const float* x, float* h, int seq_len, int d_state) {
  for (int s = 0; s < d_state; ++s) {
    float prev = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      float val = a[t * d_state + s] * prev + b[t * d_state + s] * x[t * d_state + s];
      h[t * d_state + s] = val;
      prev = val;
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void ssm_scan_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ x, float* __restrict__ h, int seq_len, int d_state) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Linear Attention & State-Space Model (SSM) Scan" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int seq_len = 256;
  const int d_state = 64;
  dim3 block(64);
  dim3 grid((d_state + 63) / 64);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
