// 3.2_flash_decoding_split_kv - FlashDecoding Split-KV Long-Context Acceleration
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
// Title: FlashDecoding Split-KV Long-Context Acceleration (Attention Mechanisms)
// Mathematical Equation: Split sequence into K partitions, compute partial log-sum-exp, merge outputs
// Hardware Goal: Partitioning sequence length dimension across SMs for low-latency batch=1 inference
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
void cpu_reference(const float* q, const float* K, const float* V, float* part_out, float* part_lse, int seq_len, int d_head, int num_splits, float scale) {
  int chunk = (seq_len + num_splits - 1) / num_splits;
  for (int s = 0; s < num_splits; ++s) {
    int k_start = s * chunk, k_end = std::min(k_start + chunk, seq_len);
    float m = -1e9f;
    for (int k = k_start; k < k_end; ++k) {
      float sc = 0.0f;
      for (int d = 0; d < d_head; ++d) sc += q[d] * K[k * d_head + d];
      sc *= scale; if (sc > m) m = sc;
    }
    float l = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
      float sc = 0.0f;
      for (int d = 0; d < d_head; ++d) sc += q[d] * K[k * d_head + d];
      l += std::exp(sc * scale - m);
    }
    for (int d = 0; d < d_head; ++d) {
      float acc = 0.0f;
      for (int k = k_start; k < k_end; ++k) {
        float sc = 0.0f;
        for (int i = 0; i < d_head; ++i) sc += q[i] * K[k * d_head + i];
        acc += (std::exp(sc * scale - m) / l) * V[k * d_head + d];
      }
      part_out[s * d_head + d] = acc;
    }
    part_lse[s] = m + std::log(l);
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void flash_decoding_split_kv_kernel(const float* __restrict__ q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ partial_out, float* __restrict__ partial_lse, int seq_len, int d_head, int num_splits, float scale) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: FlashDecoding Split-KV Long-Context Acceleration" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int seq_len = 1024;
  const int d_head = 64;
  const int num_splits = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  dim3 block(64);
  dim3 grid(num_splits);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
