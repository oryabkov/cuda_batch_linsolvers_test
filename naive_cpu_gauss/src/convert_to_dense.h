#ifndef __CONVERT_TO_DENSE_H__
#define __CONVERT_TO_DENSE_H__

#include "index_matrix.h"
#include "matrix_utils.h"

template<class Real>
int convert_to_dense(batch_systems_data<Real> batch_systems, 
                     int &batch_sz, int &N, int &M, Real *&matrices)
{
    int number_of_RHSes = 1;                            // number of RHSes
    batch_sz=batch_systems.matrices_num;
    N = batch_systems.matrices_shape;                   // number of rows
    M = batch_systems.matrices_shape + number_of_RHSes; // number of cols

    matrices = new Real[batch_sz*N*M];

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
