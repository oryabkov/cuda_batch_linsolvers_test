#ifndef __MATRIX_UTILS_H__
#define __MATRIX_UTILS_H__

//TODO make functions to be methods
//TODO why do we need these arrays of arrays?
//TODO make it all more correct (RAII, malloc result check, ect)

#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
extern "C" {
#include "mmio.h"
}

template<class Real>
struct batch_systems_data 
{
    // TODO rename all this (fex, why matrices_num and not batch_sz, matrices_shape and not N)
    int matrices_num;
    int nz_num;
    int matrices_shape;

    int **I, **J;
    Real **A_vals, **b_vals;

    batch_systems_data() : matrices_num(0), nz_num(0), matrices_shape(0)
    {
    }
};

template<class Real>
bool read_matrix_vals_line(FILE *f_A, int &I, int &J, Real &val)
{
}

template<>
bool read_matrix_vals_line<float>(FILE *f_A, int &I, int &J, float &val)
{
    return (fscanf(f_A, "%d %d %f\n", &I, &J, &val) == 3);
}

template<>
bool read_matrix_vals_line<double>(FILE *f_A, int &I, int &J, double &val)
{
    return (fscanf(f_A, "%d %d %lg\n", &I, &J, &val) == 3);    
}

template<class Real>
bool read_vector_vals_line(FILE *f_b, Real &val)
{
}

template<>
bool read_vector_vals_line<float>(FILE *f_b, float &val)
{
    return (fscanf(f_b, "%f\n", &val) == 1);
}

template<>
bool read_vector_vals_line<double>(FILE *f_b, double &val)
{
    return (fscanf(f_b, "%lg\n", &val) == 1);
}

template<class Real>
void read_matrices(const std::string &input_path_A, const std::string &input_path_b,
                   batch_systems_data<Real> &batch_systems, int &matrices_num_orig, int matrices_num_div = 1)
{
    FILE *f_A,*f_b;
    MM_typecode matcode;
    //int ret_code;
    int M, N, nz;
    int i, *I, *J,ii,iii;
    Real *val,*b_vals;

    std::cout << "Reading input files:\n   matrix A: " << input_path_A << std::endl <<
                                       "   vector b: " << input_path_b << std::endl;

    if ((f_A = fopen(input_path_A.c_str(), "r")) == NULL)
        throw std::runtime_error("Can't open input matrix A file " + input_path_A);
    if ((f_b = fopen(input_path_b.c_str(), "r")) == NULL)
        throw std::runtime_error("Can't open input vector b file " + input_path_b);
    if (fscanf(f_A, "%d\n", &batch_systems.matrices_num) != 1) 
        throw std::runtime_error("Error while reading matrices number from file " + input_path_A);

    std::cout << "Matrices num:" << batch_systems.matrices_num << std::endl;
    matrices_num_orig = batch_systems.matrices_num;
    if (batch_systems.matrices_num%matrices_num_div != 0) {
        batch_systems.matrices_num = (batch_systems.matrices_num/matrices_num_div + 1) * matrices_num_div;
        std::cout << "Rounded up matrices num: " << batch_systems.matrices_num << std::endl;
    }

    // Allocating memory for matrices data structure 
    batch_systems.I = (int **) malloc(batch_systems.matrices_num * sizeof(int **));
    batch_systems.J = (int **) malloc(batch_systems.matrices_num * sizeof(int **));
    batch_systems.A_vals = (Real **) malloc(batch_systems.matrices_num * sizeof(Real **));
    /* set nz num in matrices structure only once */

    // Allocating memory for rhs data structure 
    batch_systems.b_vals = (Real **) malloc(batch_systems.matrices_num * sizeof(Real **));

    for (ii = 0;ii < matrices_num_orig;++ii) {

        if (mm_read_banner(f_A, &matcode) != 0)
            throw std::runtime_error("Could not process Matrix Market banner");

        /*  This is how one can screen matrix types if their application */
        /*  only supports a subset of the Matrix Market data types.      */
        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) )
            throw std::runtime_error("No support for Market Market type: [" + std::string(mm_typecode_to_str(matcode)) + "]");

        // read the size of sparse matrix .... 
        if ((mm_read_mtx_crd_size(f_A, &M, &N, &nz)) !=0)
            throw std::runtime_error("Error while reading matrix sizes");

        batch_systems.nz_num = nz;
        batch_systems.matrices_shape = std::max(M,N);

        // reserve memory for cuurent matrix 
        I = (int *) malloc(nz * sizeof(int));
        J = (int *) malloc(nz * sizeof(int));
        val = (Real *) malloc(nz * sizeof(Real));

        // reserve memory for current b values
        b_vals = (Real *) malloc(std::max(M,N) * sizeof(Real));


        /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
        /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
        /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

        for (i=0; i<nz; i++) {
            if (!read_matrix_vals_line<Real>(f_A, I[i], J[i], val[i]))
                throw std::runtime_error("Error while reading values from matrix A file " + input_path_A);
            // adjust from 1-based to 0-based 
            I[i]--;  
            J[i]--;
        }

        for (i=0; i<std::max(M,N); i++) {
            if (!read_vector_vals_line(f_b, b_vals[i]))
                throw std::runtime_error("Error while reading values from vector b file " + input_path_b);
        }

        // TODO remove many I,J matrices, leave only one
        batch_systems.I[ii] = (int*) malloc(sizeof(int*)*batch_systems.nz_num);
        batch_systems.J[ii] = (int*) malloc(sizeof(int*)*batch_systems.nz_num);
        batch_systems.A_vals[ii] = (Real *) malloc(sizeof(Real *)*batch_systems.nz_num);

        for(iii=0;iii<batch_systems.nz_num;iii++) {
            batch_systems.I[ii][iii] = I[iii];
            batch_systems.J[ii][iii] = J[iii];
            batch_systems.A_vals[ii][iii] = val[iii];
        }

        batch_systems.b_vals[ii] = (Real *) malloc(sizeof(Real *)*batch_systems.matrices_shape);

        for(iii=0;iii<batch_systems.matrices_shape;iii++) {
            batch_systems.b_vals[ii][iii] = b_vals[iii];
        }

        free(I); free(J); free(val);
        free(b_vals);
    }
    if (f_A !=stdin) fclose(f_A);
    if (f_b !=stdin) fclose(f_b);

    std::cout << "Matrix read done" << std::endl;

    if (matrices_num_orig != batch_systems.matrices_num) {
        assert(matrices_num_orig > 0);
        std::cout << "Copy last matrix to get rounded up matrices num" << std::endl;
    }
    for (ii = matrices_num_orig;ii < batch_systems.matrices_num;++ii) {        
        // TODO remove many I,J matrices, leave only one
        batch_systems.I[ii] = (int*) malloc(sizeof(int*)*batch_systems.nz_num);
        batch_systems.J[ii] = (int*) malloc(sizeof(int*)*batch_systems.nz_num);
        batch_systems.A_vals[ii] = (Real *) malloc(sizeof(Real *)*batch_systems.nz_num);

        for(iii=0;iii<batch_systems.nz_num;iii++) {
            batch_systems.I[ii][iii] = batch_systems.I[matrices_num_orig-1][iii];
            batch_systems.J[ii][iii] = batch_systems.J[matrices_num_orig-1][iii];
            batch_systems.A_vals[ii][iii] = batch_systems.A_vals[matrices_num_orig-1][iii];
        }

        batch_systems.b_vals[ii] = (Real *) malloc(sizeof(Real *)*batch_systems.matrices_shape);

        for(iii=0;iii<batch_systems.matrices_shape;iii++) {
            batch_systems.b_vals[ii][iii] = batch_systems.b_vals[matrices_num_orig-1][iii];
        }
    }
}

template<class T>
void free_array(int matrices_num, T **arr)
{
    for (int i = 0;i < matrices_num;++i)
        free(arr[i]);
    free(arr);
}

template<class Real>
void free_matrices(batch_systems_data<Real> &batch_systems)
{
    free_array(batch_systems.matrices_num, batch_systems.I);
    free_array(batch_systems.matrices_num, batch_systems.J);
    free_array(batch_systems.matrices_num, batch_systems.A_vals);
    free_array(batch_systems.matrices_num, batch_systems.b_vals);
}

template<class Real>
void print_matrices_stats(const batch_systems_data<Real> &batch_systems)
{
    Real max_b_elem, min_b_elem;
    int i,j;

    std::cout<<"Total done reading:\n   A matrices:"<<batch_systems.matrices_num<<std::endl<<
             "   Each has nnz elements:"<<batch_systems.nz_num<<std::endl;

    std::cout<<"Matrices count\tMatrix size\tNNZ\n";
    std::cout<<"M1:"<<batch_systems.matrices_num<<"\t"<<batch_systems.matrices_shape<<"\t"<<batch_systems.nz_num<<std::endl;

    for(i=0;i<batch_systems.matrices_num;i++){
        for(j=0;j<batch_systems.matrices_shape;j++){
            if (i==0 and j==0)
                max_b_elem = batch_systems.b_vals[0][0],min_b_elem = batch_systems.b_vals[0][0];
            if (batch_systems.b_vals[i][j]>max_b_elem)
                max_b_elem = batch_systems.b_vals[i][j];
            else if (batch_systems.b_vals[i][j]<min_b_elem)
                min_b_elem = batch_systems.b_vals[i][j];
        }
    }

    std::cout<<"   Maximum b elem is:"<<max_b_elem<<std::endl;
    std::cout<<"   Minimum b elem is:"<<min_b_elem<<std::endl;
}

#endif
