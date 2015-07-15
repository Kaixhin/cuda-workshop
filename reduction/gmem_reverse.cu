#include <stdio.h>
#include <assert.h>
 
//implement the kernel
__global__ void reverseArrayBlock(int *out_d, int *in_d)
{
    int inOffset  = blockDim.x * blockIdx.x;
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int in  = inOffset + threadIdx.x;
    int out = outOffset + (blockDim.x - 1 - threadIdx.x);
    out_d[out] = in_d[in];
}

//program main
int main(int argc, char** argv) 
{
    //pointer to host memory and size
    int *a_h;
    int dimA = 256 * 1024; // 256K elements (1MB total)
 
    //pointer to device memory
    int *b_d;
	int *a_d;
 
    //define grid and block size
    int numThreads = 256;
    //compute number of blocks needed based on array size and desired block size
    int numBlocks = dimA / numThreads;  

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
    reverseArrayBlock<<<numBlocks, numThreads>>>(b_d, a_d);
 
    //device to host copy
    cudaMemcpy(a_h, b_d, memSize, cudaMemcpyDeviceToHost );
 
    //verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++)
    {
        assert(a_h[i] == dimA - 1 - i);
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