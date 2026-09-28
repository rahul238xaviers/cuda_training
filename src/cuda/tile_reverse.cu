#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int TILE = 128;
__global__ void tilekernel(const float *A, float *B, const int N) {
  int base = blockIdx.x * TILE;
  __shared__ float tile[TILE];
  if (N > base) {

    tile[threadIdx.x] = A[base + threadIdx.x];
  }
  __syncthreads();
  B[base + TILE - 1 - threadIdx.x] = tile[threadIdx.x];
}

int main() {

  int N = 4096;
  size_t sizeInBytes = sizeof(float) * 4096;
  float *A, *d_A, *B, *d_B;

  A = (float *)malloc(sizeInBytes);
  B = (float *)malloc(sizeInBytes);

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&d_B, sizeInBytes));

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  CUDA_CHECK(cudaMemcpy(d_A, A, sizeInBytes, cudaMemcpyHostToDevice));

  tilekernel<<<32, 128>>>(d_A, d_B, N);
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(B, d_B, sizeInBytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (A[i] != B[(i / TILE) * TILE + (TILE - 1) - (i % TILE)]) {
      std::cout << "The test failed at index " << i << std::endl;
      cudaFree(d_A);
      cudaFree(d_B);
      free(A);
      free(B);
      return 1;
    }
  }

  std::cout << "The test passed" << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  free(A);
  free(B);

  return 0;
}