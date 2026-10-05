// 2.1_moe_topk_gating - MoE Top-K Gating & Routing Softmax
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
// Title: MoE Top-K Gating & Routing Softmax (Mixture of Experts)
// Mathematical Equation: top_k_indices, top_k_weights = top_k(softmax(logits), k=2)
// Hardware Goal: Argmax Top-K selection, gating softmax normalization, and token-to-expert sorting
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
void cpu_reference(const float* logits, int* indices, float* weights, int num_tokens, int num_experts, int k) {
  for (int t = 0; t < num_tokens; ++t) {
    const float* row = logits + t * num_experts;
    int t1 = 0, t2 = 0;
    float m1 = -1e9f, m2 = -1e9f;
    for (int e = 0; e < num_experts; ++e) {
      if (row[e] > m1) { m2 = m1; t2 = t1; m1 = row[e]; t1 = e; }
      else if (row[e] > m2) { m2 = row[e]; t2 = e; }
    }
    float s1 = std::exp(m1 - m1), s2 = std::exp(m2 - m1);
    indices[t * 2 + 0] = t1; indices[t * 2 + 1] = t2;
    weights[t * 2 + 0] = s1 / (s1 + s2); weights[t * 2 + 1] = s2 / (s1 + s2);
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void moe_topk_gating_kernel(const float* __restrict__ logits, int* __restrict__ indices, float* __restrict__ weights, int num_tokens, int num_experts, int k) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: MoE Top-K Gating & Routing Softmax" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int num_tokens = 512;
  const int num_experts = 8;
  const int k = 2;
  dim3 block(256);
  dim3 grid((num_tokens + 255) / 256);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
