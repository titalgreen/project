#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_CHUNKS 32
#define START_MSG_SIZE 256
#define MEM_SIZE (1024ULL * 1024 * 64)
#define ITERATIONS 1000

/*
#TODO
    Add warmup iterations.
    Add async memory transfers using streams. 
    This means I need to just make the benchmark code completely generic, take a callback and a void* args.
        This way I can have a single benchmark function and then let the callback deal with the internals.
*/

typedef struct {
    const char* name;
    void* dst;
    const void* src;
    size_t size;
    enum cudaMemcpyKind kind;
} memcpy_args_t;

static double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

static void benchmark(memcpy_args_t* args)
{
    size_t current_msg_size = START_MSG_SIZE;

    while (current_msg_size <= args->size)
    {
        size_t chunked_msg_size = current_msg_size / NUM_CHUNKS;

        CUDA_CHECK(cudaDeviceSynchronize());

        double start = get_time();

        for (int i = 0; i < ITERATIONS; i++)
        {
            for (int j = 0; j < NUM_CHUNKS; j++)
            {
                size_t offset = j * chunked_msg_size;

                size_t bytes = (j == NUM_CHUNKS - 1)
                             ? current_msg_size - offset
                             : chunked_msg_size;

                CUDA_CHECK(cudaMemcpy(
                    (char*)args->dst + offset,
                    (const char*)args->src + offset,
                    bytes,
                    args->kind
                ));
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        double elapsed = get_time() - start;

        printf("[%zu bytes, %d chunks] %s: %.3f us, %.3f GB/s\n",
               current_msg_size,
               NUM_CHUNKS,
               args->name,
               elapsed / ITERATIONS * 1e6,
               (double)current_msg_size * ITERATIONS / elapsed / 1e9);

        current_msg_size *= 2;
    }
}

int main()
{
    void* host_pageable = NULL;
    void* host_pinned = NULL;
    void* device = NULL;

    if (NUM_CHUNKS < 1 || NUM_CHUNKS > START_MSG_SIZE) {
        fprintf(stderr, "NUM_CHUNKS must be between 1 and START_MSG_SIZE\n");
        return EXIT_FAILURE;
    }

    host_pageable = malloc(MEM_SIZE);
    if (host_pageable == NULL) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMallocHost(&host_pinned, MEM_SIZE));
    CUDA_CHECK(cudaMalloc(&device, MEM_SIZE));

    memset(host_pageable, 0, MEM_SIZE);
    memset(host_pinned, 0, MEM_SIZE);

    memcpy_args_t args[] = {
        {
            .name = "Pageable Host to Device",
            .dst = device,
            .src = host_pageable,
            .size = MEM_SIZE,
            .kind = cudaMemcpyHostToDevice
        },
        {
            .name = "Pinned Host to Device",
            .dst = device,
            .src = host_pinned,
            .size = MEM_SIZE,
            .kind = cudaMemcpyHostToDevice
        },
        {
            .name = "Pageable Device to Host",
            .dst = host_pageable,
            .src = device,
            .size = MEM_SIZE,
            .kind = cudaMemcpyDeviceToHost
        },
        {
            .name = "Pinned Device to Host",
            .dst = host_pinned,
            .src = device,
            .size = MEM_SIZE,
            .kind = cudaMemcpyDeviceToHost
        }
    };

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
        benchmark(&args[i]);
    }

    free(host_pageable);
    CUDA_CHECK(cudaFreeHost(host_pinned));
    CUDA_CHECK(cudaFree(device));

    return 0;
}
