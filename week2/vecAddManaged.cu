#include <cuda.h>
#include <stdio.h>

#define VEC_SIZE 10240
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
    float* buf_a = NULL;
    float* buf_b = NULL;
    float* buf_c = NULL;

    cudaMallocManaged(&buf_a, sizeof(float) * VEC_SIZE);
    cudaMallocManaged(&buf_b, sizeof(float) * VEC_SIZE);
    cudaMallocManaged(&buf_c, sizeof(float) * VEC_SIZE);

    //simple init of vectors. Results should just be i=i*2
    for(int i = 0; i < VEC_SIZE; i++)
    {
        buf_a[i] = (float)i;
        buf_b[i] = (float)i;
    }

    int threads = 256;
    int blocks = CEIL_DIV(VEC_SIZE, threads);
    vecAdd<<<blocks, threads>>>(buf_a, buf_b, buf_c, VEC_SIZE);

    cudaDeviceSynchronize();

    //simple output to verify results.
    for(int i = 0; i < VEC_SIZE; i++)
    {
        fprintf(stdout, "[%d]: %f\n", i, buf_c[i]);
    }

    cudaFree(buf_a);
    cudaFree(buf_b);
    cudaFree(buf_c);

    buf_a = NULL;
    buf_b = NULL;
    buf_c = NULL;
}
