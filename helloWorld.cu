#include <stdio.h>
__global__ void helloWorld(float f) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * blockDim.y + blockDim.x + threadIdx.x + threadIdx.y * blockDim.x;
  printf("Hello block %i (x %i, y %i) running thread %i (x %i, y %i), f=%f\n", blockId, blockIdx.x, blockIdx.y, threadId, threadIdx.x, threadIdx.y, f);
}
int main() {
  dim3 grid(2, 2, 1);
  dim3 block(2, 2, 1);
  helloWorld<<<grid, block>>>(1.2345f); cudaDeviceReset();
  return 0;
}
