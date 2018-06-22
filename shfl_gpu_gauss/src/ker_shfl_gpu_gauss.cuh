#ifndef __KER_SHFL_GPU_GAUSS_H__
#define __KER_SHFL_GPU_GAUSS_H__

#include <cuda.h>
#include <cuda_runtime.h>

template<class Real, int Sz, int RhsN>
__device__ Real &access_batch_matrix(int batch_sz, Real *m, int i, int row)
{
    return m[batch_sz*(Sz + RhsN)*row + i];
}

//ASSERT(Sz+RhsN <= 32)
//ASSERT(Sz+RhsN is power of 2)
template<class Real, int Sz, int RhsN>
__global__ void ker_shfl_gpu_gauss(int batch_sz, Real *m_in, Real *m_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < batch_sz*(Sz + RhsN))) return;

    Real    c[Sz];          //matrix column
    #pragma unroll
    for (int row = 0;row < Sz;++row) c[row] = access_batch_matrix<Real,Sz,RhsN>(batch_sz, m_in, i, row);

    //forward elimination
    #pragma unroll
    for (int row = 0;row < Sz;++row) {
        Real lead = __shfl(c[row], row, Sz + RhsN);
        c[row] /= lead;
        #pragma unroll
        for (int row1 = 0;row1 < Sz;++row1) {
            if (row1 <= row) continue;
            Real mul = __shfl(c[row1], row, Sz + RhsN);
            c[row1] -= c[row]*mul;
        }
    }

    //backward elimination
    #pragma unroll
    for (int row = Sz-1;row >= 0;--row) {
        #pragma unroll
        for (int row1 = Sz-1;row1 >= 0;--row1) {
            if (row1 >= row) continue;
            Real mul = __shfl(c[row1], row, Sz + RhsN);
            c[row1] -= c[row]*mul;
        }
    }

    #pragma unroll
    for (int row = 0;row < Sz;++row) access_batch_matrix<Real,Sz,RhsN>(batch_sz, m_out, i, row) = c[row];
}

#endif
