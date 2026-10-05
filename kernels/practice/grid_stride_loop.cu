#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void addHalfKernel(const float *A, float *out, const int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (index < N) {
    for (int i = index; i < N; i += stride) {
      out[i] = A[i] + 0.5f;
    }
  }
}

int main() {
  const int N = 1 << 28;
  float *A;
  float *d_A;
  float *out;
  float *d_out;
  const size_t sizeInBytes = N * sizeof(float);

  A = (float *)malloc(sizeInBytes);
  out = (float *)malloc(sizeInBytes);

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMemcpy(d_A, A, sizeInBytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, sizeInBytes));

  int threadsPerBlock = 1024;
  int block = 16;

  std::cout << "The total number of blocks are " << block << std::endl;

  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  addHalfKernel<<<block, threadsPerBlock>>>(d_A, d_out, N);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK((cudaEventSynchronize(stop)));
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::cout << "Total time taken by the GPU kernel addHalf to process " << N
            << " is " << ms << "ms" << std::endl;

  CUDA_CHECK(cudaMemcpy(out, d_out, sizeInBytes, cudaMemcpyDeviceToHost));

  int countOfMismatch = 0;
  auto cpuStart = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    if (std::fabs((A[i] + 0.5f) - out[i]) >= 1e-4f) {
      ++countOfMismatch;
    }
  }
  auto cpuStop = std::chrono::high_resolution_clock::now();
  if (countOfMismatch > 0) {
    std::cout
        << "There are mismatch in the host calculated values of the matrix and "
           "GPU based kernel. Total elements that mismatched are "

        << countOfMismatch << std::endl;
  }
  double cpuMs = std::chrono::duration_cast<std::chrono::duration<double>>(
                     cpuStop - cpuStart)
                     .count() *
                 1000.0;

  std::cout << "Speedup: " << cpuMs / ms << "x" << std::endl;
  return 0;
}