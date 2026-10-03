// 7.1_fp8_int4_gemv_dequant - FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant
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
// Title: FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant (Quantization)
// Mathematical Equation: y = sum_k(dequant(W_int4[row, k], scale[row]) * x[k]) with 75% DRAM traffic reduction
// Hardware Goal: Packing 2 INT4 nibbles per byte, on-the-fly register unpacking, and scale-factor multiply
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
void cpu_reference(const unsigned char* W_pack, const float* sc, const float* x, float* y, int M, int K) {
  for (int r = 0; r < M; ++r) {
    float scale = sc[r];
    float sum = 0.0f;
    for (int cp = 0; cp < K / 2; ++cp) {
      unsigned char b = W_pack[r * (K / 2) + cp];
      float w0 = static_cast<float>(static_cast<int>(b & 0x0F) - 8) * scale;
      float w1 = static_cast<float>(static_cast<int>((b >> 4) & 0x0F) - 8) * scale;
      sum += w0 * x[cp * 2 + 0] + w1 * x[cp * 2 + 1];
    }
    y[r] = sum;
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void int4_gemv_dequant_kernel(const unsigned char* __restrict__ W_packed, const float* __restrict__ scales, const float* __restrict__ x, float* __restrict__ y, int M, int K) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int M = 512;
  const int K = 2048;
  dim3 block(32, 8);
  dim3 grid((M + 7) / 8);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
