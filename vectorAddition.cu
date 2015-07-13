#include <stdio.h>
#include <cuda.h>
#define N 4096 // Define size of array
#define THREADS_PER_BLOCK 1024

__global__ void vectorAddKernel(int *a, int *b, int *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {
  // Initialise grid and block variables
  dim3 grid(N / THREADS_PER_BLOCK, 1, 1);
  dim3 block(THREADS_PER_BLOCK, 1, 1);

  // Initialise host arrays and device pointers
  int a_h[N];
  int b_h[N];
  int c_h[N];
  int *a_d;
  int *b_d;
  int *c_d;

  // Load host arrays with data
  for (int i = 0; i < N; i++) {
    a_h[i] = i;
    b_h[i] = i;
  }

  // Allocate device memory
  cudaMalloc((void**)&a_d, N*sizeof(int));
  cudaMalloc((void**)&b_d, N*sizeof(int));
  cudaMalloc((void**)&c_d, N*sizeof(int));

  // Copy host memory to device memory
  cudaMemcpy(a_d, a_h, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, N*sizeof(int), cudaMemcpyHostToDevice);

  // Create timer
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime;

  // Start timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch kernel
  vectorAddKernel<<<grid, block>>>(a_d, b_d, c_d);

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Copy results to device and print
  cudaMemcpy(c_h, c_d, N*sizeof(int), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < N; i++) {
    printf("%i+%i = %i\n", a_h[i], b_h[i], c_h[i]);
  }

  // Print execution time
  printf("Time to calculate results: %f ms\n", elapsedTime);

  // Free memory
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
