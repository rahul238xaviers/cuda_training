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

  __shared__ float blockSum[256];
  float threadLocalSum = 0.0f;

  if (index < N) {
    for (int i = index; i < N; i += stride) {
      threadLocalSum += A[i];
    }
  }

  blockSum[threadIdx.x] = threadLocalSum;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(out, blockSum[0]);
  }
}

int main() {
  int N = 1 << 20;

  float *A, *d_A, *out, *h_out;

  const size_t sizeInBytes = N * sizeof(float);

  A = (float *)malloc(sizeInBytes);
  h_out = (float *)malloc(sizeof(float));

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&out, sizeof(float)));

  CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  // Copy the A,B from CPU RAM to d_A, d_B DRAM
  CUDA_CHECK(cudaMemcpy(d_A, A, sizeInBytes, cudaMemcpyHostToDevice));

  int blocks = 256;
  int grids = 256;

  reduction_kernel<<<grids, blocks>>>(d_A, out, N);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaMemcpy(h_out, out, sizeof(float), cudaMemcpyDeviceToHost));

  double cpu = 0.0;

  for (int i = 0; i < N; i++)
    cpu += A[i];

  double rel = std::fabs(*h_out - cpu) / std::fabs(cpu);
  printf("GPU: %f  CPU: %f  rel err: %g\n", *h_out, cpu, rel);
  printf(rel < 1e-4 ? "[PASSED]\n" : "[FAILED]\n");

  // Release memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(out));

  free(A);
  free(h_out);
  return 0;
}