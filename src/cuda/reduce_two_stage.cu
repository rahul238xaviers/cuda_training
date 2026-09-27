#include <cmath>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
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

__global__ void partialKernel(const float *A, float *scratch, const int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float threadLocalSum = 0.0f;
  __shared__ float blockSum[256];

  if (index < N) {

    for (int i = index; i < N; i += stride) {
      threadLocalSum += A[i];
    }

    blockSum[threadIdx.x] = threadLocalSum;
  }
  __syncthreads();

  if (index < N) {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {

        blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    scratch[blockIdx.x] = blockSum[0];
  }
}

__global__ void final_kernel(const float *scratch, float *out, const int N) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float threadLocalSum = 0.0f;
  __shared__ float blockSum[256];

  if (index < N) {
    for (int i = index; i < N; i += stride) {
      threadLocalSum += scratch[i];
    }
    blockSum[threadIdx.x] = threadLocalSum;
    __syncthreads();

    if (index < N) {
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
        }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0) {
      *out = blockSum[0];
    }
  }
}
// main block
int main() {

  float *A, *d_A, *out, *d_out, *d_scratch;
  int N = 1 << 23;
  size_t sizeInBytes = N * sizeof(float);

  int totalBlocks = 512;
  int totalThreadsInBlock = 256;

  A = (float *)malloc(sizeInBytes);
  out = (float *)malloc(sizeof(float));

  for (int i = 0; i < N; i++) {
    A[i] = i;
  }

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float) * totalBlocks));

  CUDA_CHECK(cudaMemcpy(d_A, A, sizeInBytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_out, 0.0f, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_scratch, 0.0f, sizeof(float)));

  partialKernel<<<totalBlocks, totalThreadsInBlock>>>(d_A, d_scratch, N);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  final_kernel<<<1, 256>>>(d_scratch, d_out, 512);

  CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "The summation of the total number " << N << " is " << *out
            << std::endl;

  double cpu = 0.0f;

  for (int i = 0; i < N; i++) {

    cpu += A[i];
  }

  double relDiff = std::fabs(*out - cpu) / cpu;
  std::cout << "rel diff = " << relDiff
            << (relDiff <= 1e-4 ? "  PASS" : "  FAIL") << std::endl;

  free(A);
  free(out);
  cudaFree(d_A);
  cudaFree(d_out);
  cudaFree(d_scratch);
  return 0;
}