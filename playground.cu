// playground.cu - Interactive CUDA Kernel Scratchpad
// Compile: nvcc -O3 -std=c++17 --extended-lambda playground.cu -o playground_cuda && ./playground_cuda

#include <iostream>
#include <vector>
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

// Simple vector addition kernel with grid-stride loop
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  std::cout << "=== CUDA GPU Playground Sandbox ===" << std::endl;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "[INFO]: No physical NVIDIA GPU device detected on this host." << std::endl;
    std::cout << "[INFO]: This CUDA scratchpad is ready for syntax checking and compilation." << std::endl;
    return 0;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Active GPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
  std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;

  const int N = 1024;
  size_t bytes = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;
  vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool correct = true;
  for (int i = 0; i < N; ++i) {
    if (std::abs(h_c[i] - 3.0f) > 1e-5f) {
      correct = false;
      break;
    }
  }

  std::cout << "Verification: " << (correct ? "PASSED (c[i] == 3.0)" : "FAILED") << std::endl;

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
