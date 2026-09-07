#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

// 1. THE GPU KERNEL (This runs in parallel across thousands of GPU cores)
__global__ void vectorAddKernel(const float* x, const float* y, float* out, int n) {
    // Calculate the unique global index for this specific GPU thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds memory access
    if (index < n) {
        out[index] = x[index] + y[index];
    }
}

int main() {
    // Define a massive vector size (50 Million elements)
    const int N = 2 << 28;
    size_t size = N * sizeof(float);

    std::cout << "Allocating memory for " << N << " elements..." << std::endl;

    int device;
    cudaGetDevice(&device);

    // Pointers for our arrays
    float *x, *y, *out;

    // Allocate CUDA Unified Memory – accessible by both CPU and GPU automatically
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    cudaMallocManaged(&out, size);

    // Initialize the host vectors with dummy data
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 2. DEFINE GPU EXECUTION CONFIGURATION
    // 256 threads per block is a well-optimised standard for modern GPUs
    int threadsPerBlock = 256;
    // Calculate enough blocks to cover all N elements (rounding up)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks..." << std::endl;

    // 3. LAUNCH THE KERNEL on the GPU
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, out, N);

    // Wait for the GPU to finish before the CPU accesses the results
    cudaDeviceSynchronize();

    // 4. VERIFY RESULTS
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = std::max(maxError, std::abs(out[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;
    if (maxError < 1e-5) {
        std::cout << "✅ Success! The GPU added massive numbers flawlessly." << std::endl;
    } else {
        std::cout << "❌ Error: Math does not match!" << std::endl;
    }

    // Free the allocated GPU Unified memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    return 0;
}
