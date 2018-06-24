
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_timer_event.h"
#include "cuda_safe_call.h"
#include "index_matrix.h"
#include "matrix_utils.h"
#include "init_dense_sz.h"
#include "copy_dense_to_dense.h"
#include "copy_sparse_to_dense.h"
#include "write_vector.h"
#include "shfl_gpu_gauss.cuh"

namespace po = boost::program_options;

typedef SCALAR_TYPE     real;

int main(int argc, char **args)
{
    std::string         input_path_A, input_path_b, output_path_x;
    int                 repeat_times;
    int                 device_number;

    try {
        std::cout << "Use -h option for help" << std::endl;

        po::options_description desc("Tester options");
        desc.add_options()
            ("help,h", "Show help")
            ("INPUT_A,a", po::value<std::string>()->default_value("big_A.mm"), "Input .mm file with A matrices")
            ("INPUT_B,b", po::value<std::string>()->default_value("b_vector.csv"), "Input .csv file with b vector")
            ("output,o", po::value<std::string>()->default_value("x_vector.csv"), "Output solution file")
            ("device_number,d", po::value<int>()->default_value(0), "Device number")
            ("repeat_times,r", po::value<int>()->default_value(10), "Number ot test repeats");

        po::positional_options_description desc_pos;

        po::variables_map   vm;
        po::store(po::command_line_parser(argc,args).options(desc).positional(desc_pos).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        input_path_A = vm["INPUT_A"].as<std::string>();
        input_path_b = vm["INPUT_B"].as<std::string>();
        output_path_x = vm["output"].as<std::string>();

        device_number = vm["device_number"].as<int>();
        repeat_times = vm["repeat_times"].as<int>();
    } catch(std::exception& ex) {
        std::cerr << ex.what() << std::endl;
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

    // struct for matrices
    batch_systems_data<real> batch_systems;
    int                      matrices_num_orig;

    try {
        // round up matrices num to block size
        read_matrices(input_path_A, input_path_b, batch_systems, matrices_num_orig, 256);
        std::cout << "Using rounded batch_size: " << batch_systems.matrices_num << std::endl;
        std::cout << "done" << std::endl;
    } catch(std::exception& ex) {
        std::cerr << "Error while reading matrices and rhs: " << ex.what() << std::endl;
        return 1;
    }

    /* if all ok - we have our matrices in structure batch_systems
     * let's print number of matrices, number of nnz elems and stats*/
    print_matrices_stats(batch_systems);

    std::cout << "Converting matrices to dense format..." << std::endl;
    // note that extended matrix is used so rhs is appended
    int     batch_sz, N, M;
    real    *matrices, *matrices_0;
    init_dense_sz(batch_systems, batch_sz, N, M);
    if (M > 32) {
        std::cout << "Number of columns (M) more than 32 is not supported by this solver" << std::endl;
        return 1;
    }
    CUDA_SAFE_CALL( cudaMallocHost((void**)&matrices, sizeof(real)*batch_sz*N*M) );
    CUDA_SAFE_CALL( cudaMallocHost((void**)&matrices_0, sizeof(real)*batch_sz*N*M) );
    copy_sparse_to_dense(batch_systems, batch_sz, N, M, matrices);
    copy_dense_to_dense(batch_sz, N, M, matrices, matrices_0);
    std::cout << "done" << std::endl;

    std::cout << "Initializating device number " << device_number << std::endl;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    std::cout << "Device compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    if (deviceProp.major*100 + deviceProp.minor < 305) {
        std::cout << "CC less then 3.5 is not supported by this solver" << std::endl;
        return 1;
    }
    cudaSetDevice(device_number);
    std::cout << "done" << std::endl;

    real    *matrices_dev, *matrices_dev_0;

    std::cout << "Allocating device memory..." << std::endl;
    CUDA_SAFE_CALL( cudaMalloc((void**)&matrices_dev, sizeof(real)*batch_sz*N*M) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&matrices_dev_0, sizeof(real)*batch_sz*N*M) );
    std::cout << "done" << std::endl;

    std::cout << "Copying data to gpu..." << std::endl;
    CUDA_SAFE_CALL( cudaMemcpy(matrices_dev_0, matrices, sizeof(real)*batch_sz*N*M, cudaMemcpyHostToDevice) );
    std::cout << "done" << std::endl;

    cuda_timer_event    start, end;
    start.init(); end.init();

    std::cout << "Calculation..." << std::endl;
    start.record();

    for (int iter = 0;iter < repeat_times;++iter) {
        shfl_gpu_gauss(batch_sz, N, M, matrices_dev_0, matrices_dev);
    }

    end.record();
    std::cout << "done" << std::endl;

    std::cout << "Elapsed time:           " << end.elapsed_time(start)/1000. << " s" << std::endl;
    std::cout << "Repeat times:           " << repeat_times << std::endl;
    std::cout << "Time per iteration:     " << end.elapsed_time(start)/1000./repeat_times << " s" << std::endl;

    std::cout << std::endl << "T1:" << repeat_times << "\t" << batch_sz << "\t" 
              << end.elapsed_time(start)/1000. << std::endl << std::endl;

    std::cout << "Copying results back to host..." << std::endl;

    CUDA_SAFE_CALL( cudaMemcpy(matrices, matrices_dev, sizeof(real)*batch_sz*N*M, cudaMemcpyDeviceToHost) );

    // we explititly use here that number of rhs's is 1
    std::cout << "Calculating residual on cpu..." << std::endl;
    real    norm_C = 0.f;
    for (int s = 0;s < batch_sz;++s) {
        for (int ii1 = 0;ii1 < N;++ii1) {
            real    res = 0.f;
            for (int ii2 = 0;ii2 < N;++ii2) {
                res += matrices_0[IM(s,ii1,ii2)]*matrices[IM(s,ii2,N)];
            }
            norm_C = fmax(fabs(res - matrices_0[IM(s,ii1,N)]), norm_C);
        }
    }
    std::cout << "done" << std::endl;
    std::cout << "Residual norm_C:        " << norm_C << std::endl;

    try {
        //write_vector(batch_sz, N, M, matrices, output_path_x);
        // NOTE use matrices_num_orig here instead of batch_sz to match input files shape
        write_vector(matrices_num_orig, N, M, matrices, output_path_x);
    } catch(std::exception& ex) {
        std::cerr << "Error while writing result: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "Free memory..." << std::endl;
    free_matrices(batch_systems);
    CUDA_SAFE_CALL( cudaFreeHost(matrices) );
    CUDA_SAFE_CALL( cudaFreeHost(matrices_0) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev) );
    CUDA_SAFE_CALL( cudaFree(matrices_dev_0) );
    std::cout << "done" << std::endl;

    return 0;
}
