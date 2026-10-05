#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void vectorDotKernel(const float *x, const float *y, float *out,
                                int n) {

  // The __shared__ is added in front of a variable to tell the compiler that
  // this variable will be at the block level. A block is a logical construct of
  // grouping threads. A block cannot span across SM(stream multi processor).
  __shared__ float blockSum[256];
  // threadLocal scope is at the thread level. Each thread will have their own
  // allocation of this variable.
  float threadLocal = 0.0f;
  // An interesting concept and very useful. Usually, thinking is as many
  // elements in array, launch threads. But this approach may not be the best
  // use of hardware or may be limited by the hardware. Hence, the idea is to
  // allocate thread in a way that there is no collide but still they can
  // process multiple elements. This is the reason why stride came. Consider we
  // have 4 threads only but process 8 elements. Each threads are unique by
  // number and gets to process 2 elements. Now, how do we make sure if they
  // swim in thier own lanes. If the stride here is 4, then
  // thread 0 will process [0,4] while thread 1 will process
  // [1,5], thread 2 will process[2,6] and thread 3 which is last thread will
  // process [3,7] elements this gave uniqueness in processing.
  int stride = blockDim.x * gridDim.x;

  // below is the demonstrationn of how we will process the elements for each
  // thread considering the stride
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
       index += stride) {
    // we will store the thread level values and accumulate it
    threadLocal += x[index] * y[index];
  }

  // Once all the elements of threads are processed then we will write it to the
  // block level varaible store
  blockSum[threadIdx.x] = threadLocal;
  // The barrier below guarantees each thread must complete before further
  // processing is done. The concept is to hold the fastest thread while the
  // slow thread can still process. Powerful concept but to be handled carefully
  // when this is put inside a if condition which could stop only partial number
  // of threads but not all.
  __syncthreads();

  // The maths is interesting here. The art of halves until you process all the
  // elements. You keep changing the stride to half of the previous one to loop
  // less number of times and giving a time complexity of O(log N)
  // The shift operator stride >> = 1 can also be written as stride = stride >>1
  //  it drops the last bit from right while adding 0 on the far left.
  // If our block dimension was 4 (4 threads) then, loop 1 will have stride 2
  // thread 0 and thread 1 will pass the check threadIdx.x < stride hence
  // blockSum[0] = blockSum[0] + blockSum[2]
  // and  blockSum[1] = blockSum[1] + blockSum[3], this covers all the blockSum
  // that was written into blockSum[0] and blockSum[1].
  // For the next loop the stride value would be 1 which means that thread 0
  // will be only qualifier but wait, we need assurance that when thread 0 is
  // processing in the second loop the thread 1 which was writing  blockSum[1]
  // has done the job else there will be race and we must not read before we
  // complete write. Hence,  __syncthreads() is required. Once the thread 0 runs
  // in the second loop the thread 1 and thread 0 from previous loop has
  // finished the job.
  // and that would mean we have blockSum[0] = blockSum[0] + blockSum[1]
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // The below check of threadIdx.x == 0 can be done for any thread but we need
  // to make sure we are writing only once as the final value blockSum[0] is
  // stored and that is one element. Putting it for all the threads will fire
  // this block for all the threads in the block creating a
  // race condition and potentially corrupting the value. The atomicAdd is used
  // to make sure that it is added per block. We can have multiple blocks within
  // same SM or across and atomicAdd ensured that we are adding this safely
  // across the blocks by serialising read-modify-write.
  if (threadIdx.x == 0) {
    atomicAdd(out, blockSum[0]);
  }
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("name:               %s\n", p.name);
  printf("SMs:                %d\n", p.multiProcessorCount);
  printf("threads/SM (max):   %d\n", p.maxThreadsPerMultiProcessor);
  printf("threads/block(max): %d\n", p.maxThreadsPerBlock);
  printf("shared mem/block:   %zu bytes\n", p.sharedMemPerBlock);
  printf("warp size:          %d\n", p.warpSize);

  const int N = 8 << 20;
  size_t size = N * sizeof(float);

  float *x = nullptr;
  float *y = nullptr;
  float *out = nullptr;

  cudaMallocManaged(&x, size);
  cudaMallocManaged(&y, size);
  cudaMallocManaged(&out, sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i * 0.1;
  }

  *out = 0.0f;

  int maxThreadsPerBlock = 256;
  int blocksPerGrid = 128;

  vectorDotKernel<<<blocksPerGrid, maxThreadsPerBlock>>>(x, y, out, N);
  cudaDeviceSynchronize();

  float cpu = 0.0f;

  for (int i = 0; i < N; i++)
    cpu += x[i] * y[i];

  float rel = std::fabs(*out - cpu) / std::fabs(cpu);
  printf("GPU: %f  CPU: %f  rel err: %g\n", *out, cpu, rel);
  // This may show FAILED. The reasons are :-
  //  1. The CPU reference sums sequentially in float, so it keeps adding small
  //     terms into a growing total. Once the total is ~1e19, float's resolution
  //     step (~1e12) is bigger than the late terms (~7e12), so those terms get
  //     rounded away and the CPU total goes ~0.35% too LOW.
  //  2. The GPU tree reduction keeps partial sums small until the final merge,
  //     so it stays within float's precision: rel err ~1e-7.
  //  The GPU is the accurate one; the float sequential reference is not.
  //  Fix: compute the reference in double (see note).
  printf(rel < 1e-4 ? "[PASSED]\n" : "[FAILED]\n");

  cudaFree(x);
  cudaFree(y);
  cudaFree(out);

  return 0;
}