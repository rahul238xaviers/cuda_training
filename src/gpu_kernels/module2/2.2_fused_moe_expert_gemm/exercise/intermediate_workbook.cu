// 2.2_fused_moe_expert_gemm - Fused MoE SwiGLU Expert Matrix Multiply
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
// Title: Fused MoE SwiGLU Expert Matrix Multiply (Mixture of Experts)
// Mathematical Equation: Expert_e(x) = (W_gate * x * sigmoid(W_gate * x)) * (W_up * x)
// Hardware Goal: Grouped GEMM across dynamic expert tokens without global scatter/gather copying
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
void cpu_reference(const float* x, const float* W_gate, const float* W_up, float* out, int num_tokens, int d_in, int d_out) {
  for (int t = 0; t < num_tokens; ++t) {
    for (int c = 0; c < d_out; ++c) {
      float gate = 0.0f, up = 0.0f;
      for (int i = 0; i < d_in; ++i) {
        gate += x[t * d_in + i] * W_gate[i * d_out + c];
        up += x[t * d_in + i] * W_up[i * d_out + c];
      }
      float silu = gate / (1.0f + std::exp(-gate));
      out[t * d_out + c] = silu * up;
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void fused_moe_expert_kernel(const float* __restrict__ x, const float* __restrict__ W_gate, const float* __restrict__ W_up, float* __restrict__ out, int num_tokens, int d_in, int d_out) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: Fused MoE SwiGLU Expert Matrix Multiply" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int num_tokens = 64;
  const int d_in = 512;
  const int d_out = 512;
  dim3 block(64);
  dim3 grid((d_out + 63) / 64, num_tokens);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
