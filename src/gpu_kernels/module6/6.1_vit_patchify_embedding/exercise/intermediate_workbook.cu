// 6.1_vit_patchify_embedding - ViT Patchify & 3D Spatial-Temporal Embedding
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
// Title: ViT Patchify & 3D Spatial-Temporal Embedding (Vision & Video)
// Mathematical Equation: patch_token[p, d] = sum_{c, h, w}(pixel[p, c, h, w] * proj_weight[c, h, w, d])
// Hardware Goal: Transforming raw (B, C, T, H, W) video tensors into linear token projections
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
void cpu_reference(const float* images, const float* proj_w, float* tokens, int num_patches, int patch_sq, int d_model) {
  for (int p = 0; p < num_patches; ++p) {
    for (int d = 0; d < d_model; ++d) {
      float sum = 0.0f;
      for (int i = 0; i < patch_sq; ++i) sum += images[p * patch_sq + i] * proj_w[i * d_model + d];
      tokens[p * d_model + d] = sum;
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void vit_patchify_kernel(const float* __restrict__ images, const float* __restrict__ proj_weights, float* __restrict__ tokens, int num_patches, int patch_size_sq, int d_model) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: ViT Patchify & 3D Spatial-Temporal Embedding" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int num_patches = 196;
  const int patch_size_sq = 16 * 16 * 3;
  const int d_model = 768;
  dim3 block(64);
  dim3 grid((d_model + 63) / 64, num_patches);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
