#ifndef __NAIVE_CPU_GAUSS_H__
#define __NAIVE_CPU_GAUSS_H__

#include "index_matrix.h"

template<class Real>
void naive_cpu_gauss(int batch_sz, int N, int M, Real *matrices)
{
    for (int s = 0;s < batch_sz;++s) {
        //forward step
        for (int ii1 = 0;ii1 < N;++ii1) {
            Real diag = matrices[IM(s,ii1,ii1)];
            for (int ii3 = 0;ii3 < M;++ii3) {
                matrices[IM(s,ii1,ii3)] /= diag;
            }

            for (int ii2 = 0;ii2 < N;++ii2) {
                if (ii2 <= ii1) continue;
                Real mul = matrices[IM(s,ii2,ii1)];
                for (int ii3 = 0;ii3 < M;++ii3) {
                    matrices[IM(s,ii2,ii3)] -= mul*matrices[IM(s,ii1,ii3)];
                }
            }
        }

        //backward step
        for (int ii1 = N-1;ii1 >= 0;--ii1) {
            for (int ii2 = N-1;ii2 >= 0;--ii2) {
                if (ii2 >= ii1) continue;
                Real mul = matrices[IM(s,ii2,ii1)];
                for (int ii3 = 0;ii3 < M;++ii3) {
                    matrices[IM(s,ii2,ii3)] -= mul*matrices[IM(s,ii1,ii3)];
                }
            }
        }
    }
}

#endif
