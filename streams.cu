#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N_THREADS 1024

// f(x) = 1 / (1 + e^-x)
__global__ void sigmoidKernel(int *a, int *c, int N) {
  int tdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tdx < N) { // Boundary condition
    c[tdx] = 1 / (1 + __expf(a[tdx]));
  }
}

int main() {
  int N = 4096000; // Array size

  // Host pointers
  int *a_h[2], *b_h[2];
  // Device pointers
  int *a_d[2], *b_d[2];

  cudaStream_t stream[2];
  for (int i = 0; i < 2; ++i) {
    cudaStreamCreate(&stream[i]); // Stream creation

    // Allocate pinned memory 
    cudaMallocHost((void**)&a_h[i], (N/2)*sizeof(int));
    cudaMallocHost((void**)&b_h[i], (N/2)*sizeof(int));

    // Allocate device memory
    cudaMalloc((void**)&a_d[i], (N/2)*sizeof(int));
    cudaMalloc((void**)&b_d[i], (N/2)*sizeof(int));
  }

  // Load (split) input array with numbers
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < N/2; j++) {
      a_h[i][j] = i * N/2 + j;
    }
  }

  // Create timer
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime;
  // Start timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Streams
  for (int i = 0; i < 2; i++) {
    dim3 grid(N/2 / N_THREADS, 1, 1);
    dim3 block(N_THREADS, 1, 1);
    cudaMemcpyAsync(a_d[i], a_h[i], (N/2)*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
    sigmoidKernel<<<grid, block, 0, stream[i]>>>(a_d[0], b_d[0], N);
    cudaMemcpyAsync(b_h[i], b_d[i], (N/2)*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
  }

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  // Print execution time
  printf("Time to calculate results: %f ms\n", elapsedTime);

  // Clean up
  for (int i = 0; i < 2; i++) {
    cudaStreamDestroy(stream[i]);
    cudaFreeHost(a_h[i]);
    cudaFreeHost(b_h[i]);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceReset();

  return 0;
}
