#ifndef __SHFL_GPU_GAUSS_H__
#define __SHFL_GPU_GAUSS_H__

#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ker_shfl_gpu_gauss.cuh"

template<class Real>
void shfl_gpu_gauss(int batch_sz, int N, int M, Real *m_in, Real *m_out)
{
    switch (N) {
        case  1: ker_shfl_gpu_gauss<Real, 1, 1><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  2: ker_shfl_gpu_gauss<Real, 2, 2><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  3: ker_shfl_gpu_gauss<Real, 3, 1><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  4: ker_shfl_gpu_gauss<Real, 4, 4><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  5: ker_shfl_gpu_gauss<Real, 5, 3><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  6: ker_shfl_gpu_gauss<Real, 6, 2><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  7: ker_shfl_gpu_gauss<Real, 7, 1><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  8: ker_shfl_gpu_gauss<Real, 8, 8><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case  9: ker_shfl_gpu_gauss<Real, 9, 7><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 10: ker_shfl_gpu_gauss<Real,10, 6><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 11: ker_shfl_gpu_gauss<Real,11, 5><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 12: ker_shfl_gpu_gauss<Real,12, 4><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 13: ker_shfl_gpu_gauss<Real,13, 3><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 14: ker_shfl_gpu_gauss<Real,14, 2><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 15: ker_shfl_gpu_gauss<Real,15, 1><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 16: ker_shfl_gpu_gauss<Real,16,16><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 17: ker_shfl_gpu_gauss<Real,17,15><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 18: ker_shfl_gpu_gauss<Real,18,14><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 19: ker_shfl_gpu_gauss<Real,19,13><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 20: ker_shfl_gpu_gauss<Real,20,12><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 21: ker_shfl_gpu_gauss<Real,21,11><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 22: ker_shfl_gpu_gauss<Real,22,10><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 23: ker_shfl_gpu_gauss<Real,23, 9><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 24: ker_shfl_gpu_gauss<Real,24, 8><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 25: ker_shfl_gpu_gauss<Real,25, 7><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 26: ker_shfl_gpu_gauss<Real,26, 6><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 27: ker_shfl_gpu_gauss<Real,27, 5><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 28: ker_shfl_gpu_gauss<Real,28, 4><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 29: ker_shfl_gpu_gauss<Real,29, 3><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 30: ker_shfl_gpu_gauss<Real,30, 2><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        case 31: ker_shfl_gpu_gauss<Real,31, 1><<<M*batch_sz/256,256>>>(batch_sz, m_in, m_out); break;
        default: throw std::runtime_error("shfl_gpu_gauss: N>31 case is not supported yet"); 
    }
}

#endif
