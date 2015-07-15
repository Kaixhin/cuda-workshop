#include <stdio.h>
#include <assert.h>
 
//implement the fast kernel using shared memory
__global__ void reverseArrayBlock(int *out_d, int *in_d)
{
    extern __shared__ int s_data[];
 
    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;
 
    //load one element per thread from device memory and store it 
    // *in reversed order* into temporary shared memory
    s_data[blockDim.x - 1 - threadIdx.x] = in_d[in];
 
    //block until all threads in the block have written their data to shared mem
    __syncthreads();
 
    //write the data from shared memory in forward order, 
    //but to the reversed block offset as before
 
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
 
    int out = outOffset + threadIdx.x;
    out_d[out] = s_data[threadIdx.x];
}

//program main
int main(int argc, char** argv) 
{
    //pointer for host memory and size
    int *a_h;
    int dimA = 256 * 1024; //256K elements (1MB total)
 
    //pointer to device memory
    int *b_d;
	int *a_d;
 
    //define grid and block size
    int numThreads = 256;
 
    //compute number of blocks needed based on array size and desired block size
    int numBlocks = dimA / numThreads;  
 
    //part 1 of 2: Compute the number of bytes of shared memory needed
    //this is used in the kernel invocation below
    int sharedMemSize = numThreads * sizeof(int);
 
    //allocate host and device memory
    size_t memSize = numBlocks * numThreads * sizeof(int);
    a_h = (int *) malloc(memSize);
    cudaMalloc((void **) &a_d, memSize);
    cudaMalloc((void **) &b_d, memSize);
 
    //initialise input array on host
    for(int i = 0; i < dimA; ++i)
    {
        a_h[i] = i;
    }
 
    //copy host array to device array
    cudaMemcpy(a_d, a_h, memSize, cudaMemcpyHostToDevice);
 
    //launch kernel
    reverseArrayBlock<<<numBlocks, numThreads, sharedMemSize>>>(b_d, a_d);

    //device to host copy
    cudaMemcpy(a_h, b_d, memSize, cudaMemcpyDeviceToHost);
 
    //verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++)
    {
        assert(a_h[i] == dimA - 1 - i );
    }
 
    //free device memory
    cudaFree(a_d);
    cudaFree(b_d);
 
    //free host memory
    free(a_h);
 
    //if the program makes it this far, then the results are correct and there are no run-time errors
    printf("Correct!\n");
 
    return 0;
}