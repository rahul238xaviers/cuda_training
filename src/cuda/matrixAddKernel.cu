#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
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

// Helper function to initialize matrix elements
void initMatrix(float *mat, int rows, int cols, float baseVal = 1.0f) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // Row-major indexing: mat[r * cols + c]
      mat[r * cols + c] =
          baseVal + static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }
}

// TODO: Write your matrix addition kernel here
__global__ void matrixAddKernel(const float *A, const float *B, float *C,
                                int cols, int rows) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (rows > row && cols > col) {
    int flat = row * cols + col;
    C[flat] = A[flat] + B[flat];
  }
}

int main() {
  // 1. Matrix Dimensions (Rows x Columns)
  const int numRows = 1 << 14;
  const int numCols = 1 << 14;
  const size_t numElements = static_cast<size_t>(numRows) * numCols;
  const size_t sizeInBytes = numElements * sizeof(float);

  std::cout << "Matrix Dimensions: " << numRows << " x " << numCols << " ("
            << numElements << " elements)" << std::endl;

  // 2. Allocate Unified Memory (accessible by CPU and GPU)
  float *h_A = nullptr;
  float *h_B = nullptr;
  float *h_C = nullptr;

  // Host side allocation of the variables
  h_A = (float *)malloc(sizeInBytes);
  h_B = (float *)malloc(sizeInBytes);
  h_C = (float *)malloc(sizeInBytes);

  // 3. Initialize Matrices A and B
  srand(42); // Seed for reproducible pseudo-random numbers
  initMatrix(h_A, numRows, numCols, 1.0f);
  initMatrix(h_B, numRows, numCols, 2.0f);

  std::cout << "Matrices successfully initialized!" << std::endl;

  dim3 block(32, 32, 1);
  dim3 grid((numCols + block.x - 1) / block.x,
            (numRows + block.y - 1) / block.y, 1);

  std::cout << "Launching CUDA kernel with total threads in a block "
            << block.x * block.y << "... and total blocks in the grid "
            << grid.x * grid.y << std::endl;

  // Migrate the arrays to the VRAM with cudaMemPrefetchAsync method.

  float *d_A, *d_B, *d_C;

  CUDA_CHECK(cudaMalloc(&d_A, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&d_B, sizeInBytes));
  CUDA_CHECK(cudaMalloc(&d_C, sizeInBytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeInBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeInBytes, cudaMemcpyHostToDevice));

  // Creating event to measure the time taken by the GPU
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  // Execute the kernel
  matrixAddKernel<<<grid, block>>>(d_A, d_B, d_C, numCols, numRows);
  CUDA_CHECK((cudaEventRecord(stop, 0)));
  CUDA_CHECK((cudaEventSynchronize(stop)));

  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());

  // Measuring time taken by the kernel
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::cout << "Total time taken by the GPU kernel to process " << numElements
            << " is " << ms << "ms" << std::endl;

  CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeInBytes, cudaMemcpyDeviceToHost));

  // verifying the results calculated by the kernel and the CPU
  int countOfMismatch = 0;
  auto cpuStart = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if (std::fabs(h_C[r * numCols + c] -
                    (h_A[r * numCols + c] + h_B[r * numCols + c])) > 1e-5f) {
        countOfMismatch = countOfMismatch + 1;
      }
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

  // 4. Free Memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
