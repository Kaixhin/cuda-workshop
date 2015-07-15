#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void reduceKernel(int *input, int *output, int N) {
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  extern __shared__ int sdata[];
  sdata[tid] = 0.0f;

  if (i < N) {
    sdata[tid] = input[i];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s*= 2) {
      if (tid % (2*s) == 0) sdata[tid] += sdata[tid + s];
      __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
  }
}

int nextPow2(int x) {
  --x;

  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;

  return ++x;
}

int main() {
  // Size of array
  int N = 50000000;

  // Get device properties
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  int numThreads = deviceProperties.maxThreadsPerBlock;

  // Set grid and block sizes
  dim3 block(numThreads, 1, 1);
  dim3 grid((N+numThreads-1)/numThreads, 1, 1);

  // Allocate host memory
  int *data_h;
  data_h = (int*)malloc(N*sizeof(int));

  // Generate random data
  srand(time(NULL));
  for (int i = 0; i < N; i++) data_h[i] = rand() % 10;

  // Start timer
  float timeCPU, timeGPU;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Sum on CPU
  int sumCPU = 0;
  for (int i = 0; i < N; i++) sumCPU += data_h[i];
  
  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeCPU, start, stop);

  // Allocate device memory and copy host memory to device
  int *data_d;
  int *blockSum_d;

  cudaMalloc((void **)&data_d, N * sizeof(int));
  cudaMalloc((void **)&blockSum_d, grid.x * sizeof(int));
  
  cudaMemcpy(data_d, data_h, N * sizeof(int), cudaMemcpyHostToDevice);
  
  // Start timer
  cudaEventRecord(start, 0);

  // Run kernel
  reduceKernel<<<grid, block, block.x*sizeof(int)>>>(data_d, blockSum_d, N);
  // Launch recursively until final sum stored in first index of blockSum_d
  int remainingElements = grid.x;
  int level = 1;
  while (remainingElements > 1) {
    int numThreads = (remainingElements < block.x) ? nextPow2(remainingElements) : block.x;
    int numBlocks = (remainingElements + numThreads - 1) / numThreads;
    printf("Level %i kernel summing %i elements with %i blocks of %i threads\n", level, remainingElements, numBlocks, numThreads);
    reduceKernel<<<numBlocks, numThreads, numThreads*sizeof(int)>>>(blockSum_d, blockSum_d, remainingElements);
    remainingElements = numBlocks;
    level++;
  }

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeGPU, start, stop);

  // Copy back result and print
  int sumGPU;
  cudaMemcpy(&sumGPU, blockSum_d, sizeof(int), cudaMemcpyDeviceToHost);
  printf("CPU result: %i  processing time: %f ms\n", sumCPU, timeCPU);
  printf("GPU result: %i  processing time: %f ms\n", sumGPU, timeGPU);

  return 0;
}
