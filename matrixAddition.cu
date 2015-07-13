#include <stdio.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 1024

void matrixAdd(int *a, int *b, int *c, int N) {
  int index;
  for (int col = 0; col < N; col++) {
    for (int row = 0; row < N; row++) {
      c[index] = a[index] + b[index];
    }
  }
}

__global__ void matrixAddKernel(int *a, int *b, int *c, int N) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int index = row * N + col;
  c[index] = a[index] + b[index];
}

int main() {
  int N = 4096; // Define size of 1 side of square matrix
  // Initialise grid and block variables
  dim3 grid(N / THREADS_PER_BLOCK, 1, 1);
  dim3 block(THREADS_PER_BLOCK, 1, 1);

  // Initialise host pointers (dynamically allocated memory) and device pointers
  int *a_h;
  int *b_h;
  int *c_h; // GPU results
  int *d_h; // CPU results
  int *a_d;
  int *b_d;
  int *c_d;

  int size; // Number of bytes required by arrays

  // Create timer
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime;

  // Print out information about blocks and threads
  printf("Number of threads: %i (%ix%i)\n", block.x*block.y, block.x, block.y);
  printf("Number of blocks: %i (%ix%i)\n", grid.x*grid.y, grid.x, grid.y);

  // Dynamically allocate host memory
  size = N * N * sizeof(int);
  
  a_h = (int*) malloc(size);
  b_h = (int*) malloc(size);
  c_h = (int*) malloc(size);
  d_h = (int*) malloc(size);

  // Load host arrays with data
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a_h[i * N + j] = i;
      b_h[i * N + j] = i;
    }
  }

  // Allocate device memory
  cudaMalloc((void**)&a_d, size);
  cudaMalloc((void**)&b_d, size);
  cudaMalloc((void**)&c_d, size);

  // Copy host memory to device memory
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

  // Start timer for GPU
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch kernel
  matrixAddKernel<<<grid, block>>>(a_d, b_d, c_d, N);

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Print execution time
  printf("Time to calculate results on GPU: %f ms\n", elapsedTime);

  // Copy results to device
  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
  
  // Start timer for CPU
  cudaEventRecord(start, 0);

  // Launch CPU code
  matrixAdd(a_h, b_h, d_h, N);

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Print execution time
  printf("Time to calculate results on CPU: %f ms\n", elapsedTime);

  // Compare results
  for (int i = 0; i < N*N; i++) {
    if (c_h[i] != d_h[i]) {
      printf("Error: CPU and GPU results do not match\n");
      break;
    }
  }

  // Free memory
  free(a_h);
  free(b_h);
  free(c_h);
  free(d_h);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
