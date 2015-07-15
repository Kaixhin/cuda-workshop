/* Compile with -lcublas flag */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define N 275 // Matrix size

/*
static void simple_sgemm(int n, float alpha, const float *A, const float *B, float beta, float *C) {
  int i, j, k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;
      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}
*/

int main() {
  // Declare variables
  float *h_A, *h_B, *h_C;
  float *d_A = 0, *d_B = 0, *d_C = 0;
  float alpha = 1.0f, beta = 0.0f;
  int n2 = N * N;
  int i;

  cublasHandle_t handle;
  cublasStatus_t status;

  // Initialise cuBLAS
  printf("cuBLAS test running...\n");
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS initialisation error!\n");
    return EXIT_FAILURE;
  }

  // Allocate host memory
  h_A = (float*)malloc(n2 * sizeof(h_A[0]));
  h_B = (float*)malloc(n2 * sizeof(h_B[0]));
  h_C = (float*)malloc(n2 * sizeof(h_C[0]));

  // Fill matrices with test data
  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = rand() / (float)RAND_MAX;
  }

  // Allocate device memory
  cudaMalloc((void**)&d_A, n2 * sizeof(d_A[0]));
  cudaMalloc((void**)&d_B, n2 * sizeof(d_B[0]));
  cudaMalloc((void**)&d_C, n2 * sizeof(d_C[0]));

  // Initialise device matrices with host matrices
  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Device access error (write B)\n");
    return EXIT_FAILURE;
  }
  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Device access error (write C)\n");
    return EXIT_FAILURE;
  }

  // Perform sgemm
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Kernel execution error\n");
    return EXIT_FAILURE;
  }

  // Read back result
  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Device access error (read C)\n");
    return EXIT_FAILURE;
  }

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS shutdown error!\n");
    return EXIT_FAILURE;
  }

  return 0;
}
