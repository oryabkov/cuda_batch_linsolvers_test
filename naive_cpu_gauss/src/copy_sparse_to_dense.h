#ifndef __COPY_SPARSE_TO_DENSE_H__
#define __COPY_SPARSE_TO_DENSE_H__

#include "index_matrix.h"
#include "matrix_utils.h"

template<class Real>
int copy_sparse_to_dense(batch_systems_data<Real> batch_systems, 
                         int batch_sz, int N, int M, Real *matrices)
{
    for (int batch = 0;batch < batch_sz;batch++) {
        for (int row = 0;row < N;row++) {
            for (int col = 0;col < M;col++) {
                matrices[IM(batch,row,col)] = Real(0.f);
            }
        }

        for (int c_m = 0;c_m < batch_systems.nz_num;c_m++) {
            matrices[IM(batch, batch_systems.I[batch][c_m],
                        batch_systems.J[batch][c_m])] = (Real)(batch_systems.A_vals[batch][c_m]);
        }
    }

    for(int batch = 0;batch < batch_sz;batch++) {
        for (int row = 0; row < N;row++) {
            matrices[IM(batch, row, N)] = (Real)(batch_systems.b_vals[batch][row]);
        }
    }
}

#endif
