
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_timer_event.h"
#include "cuda_safe_call.h"
#include "index_matrix.h"

typedef SCALAR_TYPE     real;

#ifndef MATRIX_SZ
#error "MATRIX_SZ is undefined"
#endif
#ifndef MATRIX_SZ_EXT
#error "MATRIX_SZ_EXT is undefined"
#endif
#define N MATRIX_SZ
#define M MATRIX_SZ_EXT

template<class T, int Sz, int RhsN>
__device__ T &access_batch_matrix(int batch_sz, T *m, int i, int row)
{
    return m[batch_sz*(Sz + RhsN)*row + i];
}

//ASSERT(Sz+RhsN <= 32)
//ASSERT(Sz+RhsN is power of 2)
template<class T, int Sz, int RhsN>
__global__ void ker_gauss_elim(int batch_sz, T *m_in, T *m_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < batch_sz*(Sz + RhsN))) return;

    T       c[Sz];          //matrix column
    #pragma unroll
    for (int row = 0;row < Sz;++row) c[row] = access_batch_matrix<T,Sz,RhsN>(batch_sz, m_in, i, row);

    //forward elimination
    #pragma unroll
    for (int row = 0;row < Sz;++row) {
        T lead = __shfl(c[row], row, Sz + RhsN);
        c[row] /= lead;
        #pragma unroll
        for (int row1 = 0;row1 < Sz;++row1) {
            if (row1 <= row) continue;
            T mul = __shfl(c[row1], row, Sz + RhsN);
            c[row1] -= c[row]*mul;
        }
    }

    //backward elimination
    #pragma unroll
    for (int row = Sz-1;row >= 0;--row) {
        #pragma unroll
        for (int row1 = Sz-1;row1 >= 0;--row1) {
            if (row1 >= row) continue;
            T mul = __shfl(c[row1], row, Sz + RhsN);
            c[row1] -= c[row]*mul;
        }
    }

    #pragma unroll
    for (int row = 0;row < Sz;++row) access_batch_matrix<T,Sz,RhsN>(batch_sz, m_out, i, row) = c[row];
}

int main(int argc, char **args)
{
    if (argc < 4) {
        std::cout << "USAGE: " << std::string(args[0]) << " dev_num batch_size iters_num" << std::endl;
        return 0;
    }

    if (M < N) {
        std::cout << "Number of columns (M) must be equal or greater than number of rows (N)" << std::endl
                  << "Note that we apply GJ to extened matrix, so M-N is actually is a number if RHSs" << std::endl;
        return 1;
    }

    int batch_size = atoi(args[2]),
        dev_num = atoi(args[1]),
        iters_num = atoi(args[3]);

    batch_size = ((batch_size/256)+1)*256;
    std::cout << "Using rounded batch_size: " << batch_size << std::endl;

    std::cout << "Initializating device number " << dev_num << std::endl;
    cudaSetDevice(dev_num);
    std::cout << "done" << std::endl;

    real    *matrices, *matrices_0,
            *matrices_dev, *matrices_dev_0;

    std::cout << "Allocating memory..." << std::endl;
    CUDA_SAFE_CALL( cudaMallocHost((void**)&matrices, sizeof(real)*batch_size*N*M) );
    CUDA_SAFE_CALL( cudaMallocHost((void**)&matrices_0, sizeof(real)*batch_size*N*M) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&matrices_dev, sizeof(real)*batch_size*N*M) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&matrices_dev_0, sizeof(real)*batch_size*N*M) );
    std::cout << "done" << std::endl;
    
    std::cout << "Preparing matrices on host..." << std::endl;
    for (int s = 0;s < batch_size;++s) {
        for (int ii1 = 0;ii1 < N;++ii1) 
        for (int ii2 = 0;ii2 < M;++ii2) {
            matrices[IM(s,ii1,ii2)] = (ii1 == ii2?2.f:0.f) + real(rand()%100000)/real(100000.f);
            matrices_0[IM(s,ii1,ii2)] = matrices[IM(s,ii1,ii2)];
        }
    }
    std::cout << "done" << std::endl;

    std::cout << "Copying data to gpu..." << std::endl;
    CUDA_SAFE_CALL( cudaMemcpy(matrices_dev_0, matrices, sizeof(real)*batch_size*N*M, cudaMemcpyHostToDevice) );
    std::cout << "done" << std::endl;

    cuda_timer_event    start, end;
    start.init(); end.init();

    std::cout << "Calculation..." << std::endl;
    start.record();

    
    //int iters_num = 10;
    for (int iter = 0;iter < iters_num;++iter) {
        ker_gauss_elim<real,N,M-N><<<M*batch_size/256,256>>>(batch_size, matrices_dev_0, matrices_dev);
    }

    end.record();
    std::cout << "done" << std::endl;

    std::cout << "Elapsed time:           " << end.elapsed_time(start)/1000. << " s" << std::endl;
    std::cout << "Repeat times:           " << iters_num << std::endl;
    std::cout << "Time per iteration:     " << end.elapsed_time(start)/1000./iters_num << " s" << std::endl;

    std::cout << "Copying results back to host..." << std::endl;

    CUDA_SAFE_CALL( cudaMemcpy(matrices, matrices_dev, sizeof(real)*batch_size*N*M, cudaMemcpyDeviceToHost) );

    std::cout << "Calculating residual on cpu..." << std::endl;

    if (M != N) {
        real    norm_C = 0.f;
        for (int s = 0;s < batch_size;++s) {
            for (int rhs_i = N;rhs_i < M;++rhs_i)
            for (int ii1 = 0;ii1 < N;++ii1) {
                real    res = 0.f;
                for (int ii2 = 0;ii2 < N;++ii2) {
                    res += matrices_0[IM(s,ii1,ii2)]*matrices[IM(s,ii2,rhs_i)];
                }
                norm_C = fmax(fabs(res - matrices_0[IM(s,ii1,rhs_i)]), norm_C);
            }
        }
        std::cout << "done" << std::endl;
        std::cout << "Norm_C:                 " << norm_C << std::endl;
    }

    std::cout << "Free memory..." << std::endl;
    CUDA_SAFE_CALL( cudaFreeHost(matrices) );
    CUDA_SAFE_CALL( cudaFreeHost(matrices_0) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev_0) );
    std::cout << "done" << std::endl;

    return 0;
}
