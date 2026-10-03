// 3.1_flash_attention_forward - FlashAttention-2 Forward Kernel
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
// Title: FlashAttention-2 Forward Kernel (Attention Mechanisms)
// Mathematical Equation: O = softmax(Q * K^T / sqrt(d)) * V using online streaming max and sum-exp in SRAM
// Hardware Goal: Online softmax, SRAM tiling of Q, K, V blocks, and zero DRAM N x N materialization
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
void cpu_reference(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_head, float scale) {
  for (int q = 0; q < seq_len; ++q) {
    std::vector<float> scores(seq_len);
    float m = -1e9f;
    for (int k = 0; k < seq_len; ++k) {
      float s = 0.0f;
      for (int d = 0; d < d_head; ++d) s += Q[q * d_head + d] * K[k * d_head + d];
      s *= scale;
      scores[k] = s;
      if (s > m) m = s;
    }
    float l = 0.0f;
    for (int k = 0; k < seq_len; ++k) { scores[k] = std::exp(scores[k] - m); l += scores[k]; }
    for (int d = 0; d < d_head; ++d) {
      float acc = 0.0f;
      for (int k = 0; k < seq_len; ++k) acc += (scores[k] / l) * V[k * d_head + d];
      O[q * d_head + d] = acc;
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void flash_attention_forward_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O, int seq_len, int d_head, float scale) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: FlashAttention-2 Forward Kernel" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int seq_len = 128;
  const int d_head = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  dim3 block(64);
  dim3 grid((seq_len + 63) / 64);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
