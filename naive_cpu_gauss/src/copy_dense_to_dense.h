#ifndef __COPY_DENSE_TO_DENSE_H__
#define __COPY_DENSE_TO_DENSE_H__

#include "index_matrix.h"
#include "matrix_utils.h"

template<class Real>
void copy_dense_to_dense(int batch_sz, int N, int M, const Real *matrices_src, Real *matrices_dst)
{
    for (int batch = 0;batch < batch_sz;batch++) {
        for (int row = 0;row < N;row++) {
            for (int col = 0;col < M;col++) {
                matrices_dst[IM(batch,row,col)] = matrices_src[IM(batch,row,col)];
            }
        }
    }
}

#endif
