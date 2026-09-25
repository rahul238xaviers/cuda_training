# include <stdio.h>
# include <iostream>
# include<cuda_runtime.h>
# include <cuda_runtime_api.h>

__global__ void register_pressure_check(float *x, float *y, float *out, int N){

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N){
        
        float vals[64];
        for(int i = 0; i < 64; i++){
            if(i > 0) {
                 vals[i] = vals[i-1] * 2 + i ;
            } 
            else{
                 vals[i] = x[index];    
            }                  
    }  
     __syncthreads();

    if(index == 0){
        for(int i = 0; i < 64; i++){
        atomicAdd(out, vals[i]);
    }  
    }

    
    }
}


int main(){

    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *x, *y, *out;

    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    cudaMallocManaged(&out, sizeof(float));

    std::cout << "Allocating memory for " << N << " elements..." << std::endl;

    for(int i=0; i<= N; i++){

        x[i] = i;
        y[i] = i * 0.01f;  
    }

    *out = 0.0f;

    std::cout << "Value of out " << std::endl;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel..." << std::endl;

    register_pressure_check<<<blocksPerGrid, threadsPerBlock>>>( x, y, out, N);

    cudaDeviceSynchronize();

    std::cout << "Value of out " << *out << std::endl;


    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    return 0;
}