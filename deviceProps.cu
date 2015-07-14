#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main() {
  // Get number of GPUs
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of GPU devices: %i\n", deviceCount);

  // Get CUDA driver and runtime version
  int driverVersion;
  int runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);

  // Get device properties
  cudaDeviceProp deviceProperties;
  for (int i = 0; i < deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProperties, i);
    printf("Name: %s\n", deviceProperties.name);
  }

  return 0;
}
