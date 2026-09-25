
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void reduction_kernel(const float *A, float *out, const int N) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (index < N) {
    for (int i = index; i < N; i += stride) {
    }
  }
}

int main() {
  int N = 1 << 20;

  float *A, *d_A, *out;

  const size_t sizeInBytes = N * sizeof(float);

  A = (float *)malloc(sizeInBytes);

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&out, sizeof(float)));

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  // Copy the A,B from CPU RAM to d_A, d_B DRAM
  CUDA_CHECK(cudaMemcpy(d_A, A, sizeInBytes, cudaMemcpyHostToDevice));

  int blocks = 256;
  int grids = 256;

  // Release memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(out));

  free(A);
  return 0;
}