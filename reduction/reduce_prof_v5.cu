#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void reduceKernel(int *input, int *output, int N)
{
	int tid = threadIdx.x;
	int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	extern __shared__ int sdata[];
	sdata[tid] = 0;

	//perform first level of reduction, reading from global memory, writing to shared memory
	int sum = (i < N) ? input[i] : 0;

	if (i + blockDim.x < N) sum += input[i+blockDim.x];
	sdata[tid] = sum;

	//synchronise threads in this block before manipulating with the data
	__syncthreads();

	//do reduction in shared memory
	for (int s=blockDim.x/2; s>32; s>>=1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

    if (tid < 32)
    {
        //now that we are using warp-synchronous programming (below) we need to declare our shared memory
        //volatile so that the compiler doesn't reorder stores to it and induce incorrect behavior
        volatile int* smem = sdata;
        if (N >=  64) { smem[tid] += smem[tid + 32]; }
        if (N >=  32) { smem[tid] += smem[tid + 16]; }
        if (N >=  16) { smem[tid] += smem[tid +  8]; }
        if (N >=   8) { smem[tid] += smem[tid +  4]; }
        if (N >=   4) { smem[tid] += smem[tid +  2]; }
        if (N >=   2) { smem[tid] += smem[tid +  1]; }
    }

	//write result for this block to global mem
	if(tid == 0) output[blockIdx.x] = sdata[0];
}

int nextPow2(int x)
{
    --x;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return ++x;
}

int main(int argc, char **argv)
{
	//number of elements in the array
	int N = 4000000;

	//set the number of threads
	int maxThreads = 128;

	//grid and block sizes
	int threads = (N < maxThreads*2) ? nextPow2((N + 1)/ 2) : maxThreads;
    int blocks = (N + (threads * 2 - 1)) / (threads * 2);	
	dim3 grid(blocks, 1, 1);
	dim3 block(threads, 1, 1);

	//print the number of elements
	printf("\n======================\n");
	printf("Parallel reduction sum\n");
	printf("======================\n\n");
	printf("Total number of elements to sum: %i\n", N);
	printf("Kernel launch configuration: %i blocks of %i threads\n", grid.x, block.x);

	//host memory pointer
	int *data_h;

	//allocate host memory
	data_h = (int*)malloc(N*sizeof(int));

	//initialise random number generator seed based on current time
	srand(time(NULL));

	//generate data
	for (int i=0; i<N; i++) data_h[i] = 1;

	//device memory pointers
	int *data_d;
	int *blockSum_d;
	
	//allocate device memory
	cudaMalloc((void **)&data_d, N * sizeof(int));
	cudaMalloc((void **)&blockSum_d, grid.x * sizeof(int));

	//copy memory to device
	cudaMemcpy(data_d, data_h, N * sizeof(int), cudaMemcpyHostToDevice);

	//calculate sums on device
	float timeGPU;
	cudaEvent_t start;     
	cudaEvent_t stop;
	cudaEventCreate(&start);     		
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//level 0
	printf("Level 0 kernel summing %i elements with %i blocks of %i threads...\n", N, grid.x, block.x);
	reduceKernel<<<grid, block, block.x*sizeof(int)>>>(data_d, blockSum_d, N);
	//level 1+
	int remainingElements = grid.x;
	int level = 1;
	while(remainingElements > 1)
	{
		threads = (remainingElements < maxThreads*2) ? nextPow2((remainingElements + 1)/ 2) : maxThreads;
		blocks = (remainingElements + (threads * 2 - 1)) / (threads * 2);	

		printf("Level %i kernel summing %i elements with %i blocks of %i threads...\n", level, remainingElements, blocks, threads);
		reduceKernel<<<blocks, threads, threads*sizeof(int)>>>(blockSum_d, blockSum_d, remainingElements);
		remainingElements = blocks;

		level++;
	}
	cudaEventRecord(stop, 0);     		
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeGPU, start, stop);

	//copy results back to host
	int sumGPU;
	cudaMemcpy(&sumGPU, blockSum_d, sizeof(int), cudaMemcpyDeviceToHost);

	//print result
	printf("result: %i   time: %f ms   throughput: %.4f GB/s\n", sumGPU, timeGPU, 1.0e-9 * ((double)N*sizeof(int))/(timeGPU/1000));

	//cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces
    cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	else return 0;
}