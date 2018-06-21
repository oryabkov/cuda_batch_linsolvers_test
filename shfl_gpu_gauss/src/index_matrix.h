#ifndef __INDEX_MATRIX_H__
#define __INDEX_MATRIX_H__

//m is batch index (matrix index); j is row index; k is column index
#define IM(m,j,k)       ((j)*(batch_size)*(M)+(k)+(m)*(M))

#endif
