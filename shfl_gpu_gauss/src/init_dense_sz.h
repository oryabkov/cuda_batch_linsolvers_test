#ifndef __INIT_DENSE_SZ_H__
#define __INIT_DENSE_SZ_H__

#include "matrix_utils.h"

template<class Real>
void init_dense_sz(batch_systems_data<Real> batch_systems, int &batch_sz, int &N, int &M)
{
    int number_of_RHSes = 1;                            // number of RHSes
    batch_sz=batch_systems.matrices_num;
    N = batch_systems.matrices_shape;                   // number of rows
    M = batch_systems.matrices_shape + number_of_RHSes; // number of cols
}

#endif
