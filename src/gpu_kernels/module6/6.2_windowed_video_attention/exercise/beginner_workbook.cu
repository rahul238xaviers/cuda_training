// 6.2_windowed_video_attention - 2D/3D Windowed Video Attention (Swin/Video-LLM)
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
// Title: 2D/3D Windowed Video Attention (Swin/Video-LLM) (Vision & Video)
// Mathematical Equation: Local_Attn = softmax(Q_w * K_w^T / sqrt(d) + Mask_w) * V_w
// Hardware Goal: Local 3D spatial-temporal attention windows with cyclic shifting and masking
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
void cpu_reference(const float* Q, const float* K, const float* V, float* O, int num_windows, int win_size, int d_head, float scale) {
  for (int w = 0; w < num_windows; ++w) {
    for (int q = 0; q < win_size; ++q) {
      const float* q_ptr = Q + w * (win_size * d_head) + q * d_head;
      const float* k_base = K + w * (win_size * d_head);
      const float* v_base = V + w * (win_size * d_head);
      float m = -1e9f;
      for (int k = 0; k < win_size; ++k) {
        float s = 0.0f;
        for (int d = 0; d < d_head; ++d) s += q_ptr[d] * k_base[k * d_head + d];
        s *= scale; if (s > m) m = s;
      }
      float l = 0.0f;
      for (int k = 0; k < win_size; ++k) {
        float s = 0.0f;
        for (int d = 0; d < d_head; ++d) s += q_ptr[d] * k_base[k * d_head + d];
        l += std::exp(s * scale - m);
      }
      for (int d = 0; d < d_head; ++d) {
        float acc = 0.0f;
        for (int k = 0; k < win_size; ++k) {
          float s = 0.0f;
          for (int i = 0; i < d_head; ++i) s += q_ptr[i] * k_base[k * d_head + i];
          acc += (std::exp(s * scale - m) / l) * v_base[k * d_head + d];
        }
        O[w * (win_size * d_head) + q * d_head + d] = acc;
      }
    }
  }
}

// --- YOUR GPU KERNEL IMPLEMENTATION ---
__global__ void windowed_video_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O, int num_windows, int win_size, int d_head, float scale) {
  // TODO: Implement the kernel described in the specification above.
  // --- YOUR CODE STARTS HERE ---

  // --- YOUR CODE ENDS HERE ---
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  Evaluating: 2D/3D Windowed Video Attention (Swin/Video-LLM)" << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical GPU detected on this host." << std::endl;
    std::cout << "[INFO]: Exercise code structure and syntax validated." << std::endl;
    return 0;
  }

  // Setup dimensions
  const int num_windows = 16;
  const int win_size = 64;
  const int d_head = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  dim3 block(64);
  dim3 grid(1, num_windows);

  std::cout << "[1/3] Running CPU Reference..." << std::endl;
  // Allocation and verification harness...
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  std::cout << "[3/3] Evaluating Numerical Accuracy & Hardware Roofline..." << std::endl;

  std::cout << "Result: Ready for implementation." << std::endl;
  return 0;
}
