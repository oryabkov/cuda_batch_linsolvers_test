
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_timer_event.h"
#include "cuda_safe_call.h"
#include "index_matrix.h"
#include "ker_shfl_gpu_gauss.cuh"

typedef SCALAR_TYPE     real;

#ifndef MATRIX_SZ
#error "MATRIX_SZ is undefined"
#endif
#ifndef MATRIX_SZ_EXT
#error "MATRIX_SZ_EXT is undefined"
#endif
#define N MATRIX_SZ
#define M MATRIX_SZ_EXT

int main(int argc, char **args)
{
    if (argc < 4) {
        std::cout << "USAGE: " << std::string(args[0]) << " dev_num batch_size repeat_times" << std::endl;
        return 0;
    }

    if (M < N) {
        std::cout << "Number of columns (M) must be equal or greater than number of rows (N)" << std::endl
                  << "Note that we apply GJ to extened matrix, so M-N is actually is a number if RHSs" << std::endl;
        return 1;
    }
    if (M > 32) {
        std::cout << "Number of columns (M) more than 32 is not supported by this solver" << std::endl;
        return 1;
    }

    if (sizeof(real) == sizeof(float))
        std::cout << "Float variant is tested" << std::endl;
    else if (sizeof(real) == sizeof(double))
        std::cout << "Double variant is tested" << std::endl;
    else {
        std::cout << "Real is neither float nor double" << std::endl;
        return 1;
    }

    int batch_size = atoi(args[2]),
        dev_num = atoi(args[1]),
        repeat_times = atoi(args[3]);

    batch_size = ((batch_size/256)+1)*256;
    std::cout << "Using rounded batch_size: " << batch_size << std::endl;

    std::cout << "Initializating device number " << dev_num << std::endl;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_num);
    std::cout << "Device compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    if (deviceProp.major*100 + deviceProp.minor < 305) {
        std::cout << "CC less then 3.5 is not supported by this solver" << std::endl;
        return 1;
    }
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
    
    std::cout << "Preparing random matrices on host..." << std::endl;
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

    for (int iter = 0;iter < repeat_times;++iter) {
        ker_shfl_gpu_gauss<real,N,M-N><<<M*batch_size/256,256>>>(batch_size, matrices_dev_0, matrices_dev);
    }

    end.record();
    std::cout << "done" << std::endl;

    std::cout << "Elapsed time:           " << end.elapsed_time(start)/1000. << " s" << std::endl;
    std::cout << "Repeat times:           " << repeat_times << std::endl;
    std::cout << "Time per iteration:     " << end.elapsed_time(start)/1000./repeat_times << " s" << std::endl;

    std::cout << "Copying results back to host..." << std::endl;

    CUDA_SAFE_CALL( cudaMemcpy(matrices, matrices_dev, sizeof(real)*batch_size*N*M, cudaMemcpyDeviceToHost) );

    if (M != N) {
        std::cout << "Calculating residual on cpu..." << std::endl;
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
        std::cout << "Residual norm_C:        " << norm_C << std::endl;
    } else {
        std::cout << "Note, that you are solving systems without RHSs (N==M)" << std::endl;
    }

    std::cout << "Free memory..." << std::endl;
    CUDA_SAFE_CALL( cudaFreeHost(matrices) );
    CUDA_SAFE_CALL( cudaFreeHost(matrices_0) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev_0) );
    std::cout << "done" << std::endl;

    return 0;
}
