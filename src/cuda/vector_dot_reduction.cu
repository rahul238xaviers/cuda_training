# include<cuda_runtime.h>
# include<stdio.h>

__global__ void vectorDotKernel(const float *x,const float *y, float *out, int n){

     int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < n)
            {
               out[index] = x[index] * y[index];
                
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
    return 0;
}