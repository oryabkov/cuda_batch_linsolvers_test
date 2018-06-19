
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "system_timer_event.h"

typedef SCALAR_TYPE     real;

#define IM(m,j,k)       ((m)*(N)*(M)+(j)+(k)*(N))
//#define IV(m,j)         ((m)*(N)+(j))

int main(int argc, char **args)
{
    if (argc < 5) {
        printf("USAGE: %s batch_size N M iters_num\n", args[0]);
        return 0;
    }

    int BATCH_SZ = atoi(args[1]),
        N = atoi(args[2]),
        M = atoi(args[3]),
        iters_num = atoi(args[4]);

    real    *matrices = new real[BATCH_SZ*N*M],
            *matrices_0 = new real[BATCH_SZ*N*M];

    if (sizeof(real) == sizeof(float))
        printf("float variant is tested\n");
    else if (sizeof(real) == sizeof(double))
        printf("double variant is tested\n");
    else {
        printf("real is neither float nor double\n");
        return 1;
    }
    
    for (int s = 0;s < BATCH_SZ;++s) {
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
    for (int s = 0;s < BATCH_SZ;++s) {
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

    printf("elapsed time = %f s\n", end.elapsed_time(start)/1000.);
    printf("iters_num = %d\n", iters_num);
    printf("time per iteration = %f s\n", end.elapsed_time(start)/1000./iters_num);

    real    norm_C = 0.f;
    for (int s = 0;s < BATCH_SZ;++s) {
        for (int ii1 = 0;ii1 < N;++ii1) {
            real    res = 0.f;
            for (int ii2 = 0;ii2 < N;++ii2) {
                res += matrices_0[IM(s,ii1,ii2)]*matrices[IM(s,ii2,N)];
                //printf("%f ",matrices[IM(s,ii1,ii2)]);
            }
            //printf("%f %f\n", res, matrices_0[IM(s,ii1,N)]);
            norm_C = fmax(fabs(res - matrices_0[IM(s,ii1,N)]), norm_C);
        }
    }
    printf("norm_C = %e \n", norm_C);

    delete []matrices;
    delete []matrices_0;

    return 0;
}