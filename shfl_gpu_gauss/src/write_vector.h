#ifndef __WRITE_VECTOR_H__
#define __WRITE_VECTOR_H__

#include <string>
#include <stdexcept>

#include "index_matrix.h"

template<class Real>
void write_vector(int out_sz, int batch_sz, int N, int M, Real *matrices, const std::string &output_fn)
{
    std::cout << "Writing output solution file: " << output_fn << std::endl;

    FILE *f = fopen(output_fn.c_str(), "w");
    if (f == NULL)
        throw std::runtime_error("Error opening output file " + output_fn);

    for (int batch = 0;batch < out_sz;batch++)
    for (int row = 0;row < N;row++)
        fprintf(f,"%le\n", (double)matrices[IM(batch,row,N)]);

    fclose(f);
}

#endif
