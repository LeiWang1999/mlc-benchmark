
#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
// nvcc -gencode arch=compute_80,code=sm_80 -lineinfo -O3 -lcp_lib 2.evaluate_i4_gemm_ladder.cu -o evaluate ; ./evaluate

extern "C" float profile(half* args0, int8_t* args1, half* args2);

int main(){
    const int M = 16384;
    const int N = 16384;
    const int K = 16384;

    half* A = (half *)malloc(M * K * sizeof(half));
    int8_t* B = (int8_t *)malloc(K * N * sizeof(int8_t));
    half* C = (half *)malloc(M * N * sizeof(half));

    for(int i = 0; i < M * K; i++){
        A[i] = __float2half((i % 100)  / 1000.0);
    }

    for(int i = 0; i < K * N; i++){
        B[i] = (i % 100);
    }

    half* d_A, *d_C;
    int8_t* d_B;
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, K * N * sizeof(int8_t));
    cudaMalloc((void**)&d_C, M * N * sizeof(half));
    
    // copy data
    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    printf("Time: %f ms\n", profile(d_A, d_B, d_C));

}