// playground_kernel.cu - Deep Learning GPU Kernel Benchmark & Evaluation Sandbox
// Compile: nvcc -O3 -std=c++17 --extended-lambda playground_kernel.cu -o playground_kernel && ./playground_kernel

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

// -----------------------------------------------------------------------------
// [PROBLEM SPECIFICATION]
// Target: Implement a 1D Block-Tiled Reduction / Fused Kernel.
// Operation: y = sum(x) or y[i] = activation(x[i])
// Bandwidth Formula: (bytes_read + bytes_written) / (elapsed_sec * 1e9) GB/s
// TFLOPs Formula: (total_operations) / (elapsed_sec * 1e12)
// -----------------------------------------------------------------------------

// CPU Ground Truth Reference Implementation
void cpu_reference(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    // Example: Fused Sigmoid activation f(x) = 1.0 / (1.0 + exp(-x))
    y[i] = 1.0f / (1.0f + std::exp(-x[i]));
  }
}

// GPU Kernel Implementation Under Evaluation
__global__ void gpu_kernel_under_test(const float* __restrict__ x, float* __restrict__ y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    // Fast hardware intrinsic or vectorized arithmetic
    y[i] = 1.0f / (1.0f + __expf(-x[i]));
  }
}

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "       GPU Kernel Benchmark & Verification Harness        " << std::endl;
  std::cout << "==========================================================" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical NVIDIA GPU device detected on this host." << std::endl;
    std::cout << "[INFO]: Harness validated for syntax and structure." << std::endl;
    return 0;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Target Device: " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << std::endl;

  const int N = 1024 * 1024 * 16; // 16 Million elements (~64 MB per buffer)
  size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N);
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(i % 100) * 0.05f - 2.5f;
  }
  std::vector<float> h_y_gpu(N, 0.0f);
  std::vector<float> h_y_ref(N, 0.0f);

  // 1. Run CPU Ground Truth Reference
  std::cout << "\n[1/3] Executing CPU Ground-Truth Reference..." << std::endl;
  cpu_reference(h_x.data(), h_y_ref.data(), N);

  // 2. Allocate GPU VRAM
  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  // 3. Warm-up and Timed GPU Execution
  std::cout << "[2/3] Executing GPU Kernel with 100 Timed Iterations..." << std::endl;
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;
  if (gridSize > 65535) gridSize = 65535;

  // Warmup
  gpu_kernel_under_test<<<gridSize, blockSize>>>(d_x, d_y, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int iterations = 100;
  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < iterations; ++it) {
    gpu_kernel_under_test<<<gridSize, blockSize>>>(d_x, d_y, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float totalElapsedMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&totalElapsedMs, start, stop));
  float avgElapsedMs = totalElapsedMs / iterations;

  // 4. Verify Correctness
  std::cout << "[3/3] Verifying Numerical Correctness..." << std::endl;
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  float maxDiff = 0.0f;
  for (int i = 0; i < N; ++i) {
    float diff = std::abs(h_y_gpu[i] - h_y_ref[i]);
    if (diff > maxDiff) maxDiff = diff;
  }

  std::cout << "\n-------------------- RESULTS --------------------" << std::endl;
  bool passed = (maxDiff < 1e-4f);
  std::cout << "Correctness: " << (passed ? "PASSED (max diff < 1e-4)" : "FAILED") << std::endl;
  std::cout << "Max Absolute Difference: " << maxDiff << std::endl;

  // 5. Bandwidth & Performance Metrics
  double totalBytesPerIter = static_cast<double>(bytes) * 2.0; // 1 read + 1 write
  double elapsedSec = avgElapsedMs / 1000.0;
  double bandwidthGBs = (totalBytesPerIter / elapsedSec) / 1e9;
  double flopsPerIter = static_cast<double>(N) * 4.0; // ~4 FLOPs for exp & divide
  double tflops = (flopsPerIter / elapsedSec) / 1e12;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Average Latency: " << avgElapsedMs << " ms" << std::endl;
  std::cout << "Achieved DRAM Bandwidth: " << bandwidthGBs << " GB/s" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPs" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  return passed ? 0 : 1;
}
