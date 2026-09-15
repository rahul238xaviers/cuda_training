#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Compile: nvcc -O2 -arch=sm_75 src/cuda/gpu_check.cu -o output/gpu_check

#define GB(x) ((double)(x) / (1024.0 * 1024.0 * 1024.0))

int main() {
    cudaDeviceProp p;
    int deviceCount = 0;
    int runtimeVersion = 0;
    int driverVersion = 0;
    int clockRate = 0;
    int memClockRate = 0;

    cudaGetDeviceCount(&deviceCount);
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);

    printf("CUDA runtime version:  %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("CUDA driver version:   %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("Device count:          %d\n", deviceCount);
    printf("\n");

    cudaGetDeviceProperties(&p, 0);
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);

    printf("name:                  %s\n", p.name);
    printf("compute capability:    %d.%d\n", p.major, p.minor);
    printf("total global mem:      %.2f GB\n", GB(p.totalGlobalMem));
    printf("L2 cache:              %.2f MB\n", (double)p.l2CacheSize / (1024.0 * 1024.0));
    printf("integrated:            %s\n", p.integrated ? "yes" : "no");
    printf("\n");

    printf("SMs:                   %d\n", p.multiProcessorCount);
    printf("registers/SM:          %d\n", p.regsPerMultiprocessor);
    printf("max threads/SM:        %d\n", p.maxThreadsPerMultiProcessor);
    printf("max threads/block:     %d\n", p.maxThreadsPerBlock);
    printf("max block dims:        (%d, %d, %d)\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("max grid dims:         (%d, %d, %d)\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("shared mem/block:      %zu bytes\n", p.sharedMemPerBlock);
    printf("shared mem/SM:         %zu bytes\n", p.sharedMemPerMultiprocessor);
    printf("warp size:             %d\n", p.warpSize);
    printf("max blocks/SM:         %d\n", p.maxBlocksPerMultiProcessor);
    printf("\n");

    double clockMHz = clockRate / 1000.0;
    double memClockMHz = memClockRate / 1000.0;
    double memBusBytes = p.memoryBusWidth / 8.0;
    double bandwidth = 2.0 * memClockMHz * memBusBytes / 1000.0;

    printf("core clock:            %.0f MHz\n", clockMHz);
    printf("memory clock:          %.0f MHz\n", memClockMHz);
    printf("memory bus width:      %d bits\n", p.memoryBusWidth);
    printf("memory bandwidth:      %.0f GB/s\n", bandwidth);
    printf("\n");

    printf("PCI location:          %04x:%02x:%02x\n", p.pciDomainID, p.pciBusID, p.pciDeviceID);
    printf("async engines:         %d\n", p.asyncEngineCount);
    printf("concurrent kernels:    %s\n", p.concurrentKernels ? "yes" : "no");
    printf("host memory mapped:    %s\n", p.canMapHostMemory ? "yes" : "no");
    printf("managed memory:        %s\n", p.managedMemory ? "yes" : "no");
    printf("cooperative launch:    %s\n", p.cooperativeLaunch ? "yes" : "no");

    return 0;
}