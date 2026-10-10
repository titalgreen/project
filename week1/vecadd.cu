#include <cuda.h>
#include <stdio.h>

#define VEC_SIZE 1024
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void vecAdd(float* A, float* B, float* C, int vec_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < vec_size)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    float* host_a = NULL;
    float* host_b = NULL;
    float* host_c = NULL;

    float* dev_a = NULL;
    float* dev_b = NULL;
    float* dev_c = NULL;


    host_a = (float*)malloc(sizeof(float) * VEC_SIZE);
    host_b = (float*)malloc(sizeof(float) * VEC_SIZE);
    host_c = (float*)malloc(sizeof(float) * VEC_SIZE);

    //simple way to be able to validate the computation.
    //results should just be i=i*2.
    for(int i = 0; i < VEC_SIZE; i++)
    {
        host_a[i] = (float)i;
        host_b[i] = (float)i;
        printf("%f -- %f\n", host_a[i], host_b[i]);
    }

    cudaMalloc(&dev_a, sizeof(float) * VEC_SIZE);
    cudaMalloc(&dev_b, sizeof(float) * VEC_SIZE);
    cudaMalloc(&dev_c, sizeof(float) * VEC_SIZE);

    cudaMemcpy(dev_a, host_a, sizeof(float) * VEC_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(float) * VEC_SIZE, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = CEIL_DIV(VEC_SIZE, threads);
    vecAdd<<<blocks, threads>>>(dev_a, dev_b, dev_c, VEC_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(host_c, dev_c, sizeof(float) * VEC_SIZE, cudaMemcpyDeviceToHost);

    //simple output to verify results.
    for(int i = 0; i < VEC_SIZE; i++)
    {
        fprintf(stdout, "I: %f\n", host_c[i]);
    }

    free(host_a);
    free(host_b);
    free(host_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
