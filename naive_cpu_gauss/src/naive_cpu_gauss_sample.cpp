
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

#include "system_timer_event.h"
#include "index_matrix.h"

typedef SCALAR_TYPE     real;

int main(int argc, char **args)
{
    if (argc < 5) {
        std::cout << "USAGE: " << std::string(args[0]) << " batch_size N M iters_num" << std::endl;
        return 0;
    }

    int batch_sz = atoi(args[1]),
        N = atoi(args[2]),
        M = atoi(args[3]),
        iters_num = atoi(args[4]);

    if (M < N) {
        std::cout << "Number of columns (M) must be equal or greater than number of rows (N)" << std::endl
                  << "Note that we apply GJ to extened matrix, so M-N is actually is a number if RHSs" << std::endl;
        return 1;
    }

    real    *matrices = new real[batch_sz*N*M],
            *matrices_0 = new real[batch_sz*N*M];

    if (sizeof(real) == sizeof(float))
        std::cout << "Float variant is tested" << std::endl;
    else if (sizeof(real) == sizeof(double))
        std::cout << "Double variant is tested" << std::endl;
    else {
        std::cout << "Real is neither float nor double" << std::endl;
        return 1;
    }
    
    for (int s = 0;s < batch_sz;++s) {
        for (int ii1 = 0;ii1 < N;++ii1) 
        for (int ii2 = 0;ii2 < M;++ii2) {
            matrices[IM(s,ii1,ii2)] = (ii1 == ii2?2.f:0.f) + real(rand()%100000)/real(100000.f);
            matrices_0[IM(s,ii1,ii2)] = matrices[IM(s,ii1,ii2)];
        }
    }

    system_timer_event    start, end;
    start.init(); end.init();

    start.record();

    for (int iter = 0;iter < iters_num;++iter)
    for (int s = 0;s < batch_sz;++s) {
        //forward step
        for (int ii1 = 0;ii1 < N;++ii1) {
            real diag = matrices[IM(s,ii1,ii1)];
            for (int ii3 = 0;ii3 < M;++ii3) {
                matrices[IM(s,ii1,ii3)] /= diag;
            }

            for (int ii2 = 0;ii2 < N;++ii2) {
                if (ii2 <= ii1) continue;
                real mul = matrices[IM(s,ii2,ii1)];
                for (int ii3 = 0;ii3 < M;++ii3) {
                    matrices[IM(s,ii2,ii3)] -= mul*matrices[IM(s,ii1,ii3)];
                }
            }
        }

        //backward step
        for (int ii1 = N-1;ii1 >= 0;--ii1) {
            for (int ii2 = N-1;ii2 >= 0;--ii2) {
                if (ii2 >= ii1) continue;
                real mul = matrices[IM(s,ii2,ii1)];
                for (int ii3 = 0;ii3 < M;++ii3) {
                    matrices[IM(s,ii2,ii3)] -= mul*matrices[IM(s,ii1,ii3)];
                }
            }
        }
    }

    end.record();

    std::cout << "Elapsed time = " << end.elapsed_time(start)/1000. << " s\n";
    std::cout << "Iters_num = " << iters_num << std::endl;
    std::cout << "Time per iteration = " << end.elapsed_time(start)/1000./iters_num << " s" << std::endl;

    if (M != N) {
        real    norm_C = 0.f;
        for (int s = 0;s < batch_sz;++s) {
            for (int rhs_i = N;rhs_i < M;++rhs_i)
            for (int ii1 = 0;ii1 < N;++ii1) {
                real    res = 0.f;
                for (int ii2 = 0;ii2 < N;++ii2) {
                    res += matrices_0[IM(s,ii1,ii2)]*matrices[IM(s,ii2,rhs_i)];
                }
                norm_C = fmax(fabs(res - matrices_0[IM(s,ii1,rhs_i)]), norm_C);
            }
        }
        std::cout << "Norm_C = " << norm_C << std::endl;
    } else {
        std::cout << "Note, that you are solving systems without RHSs (N==M)" << std::endl;
    }

    delete []matrices;
    delete []matrices_0;

    return 0;
}