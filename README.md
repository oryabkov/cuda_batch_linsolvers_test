## Description
This repository contains small set of tests aimed to compare
different approaches for solution of large number of small linear
systems on GPU (so called batch mode). Only CUDA realizations and
libraries are considered (no OpenCL versions). The basic programming
language used in tests is С/C++. There are also some support scripts
written in Python 3.

Each directory contains separate solution approach. Each directory
generally contains two programs. First one is called \*_sample which is
sample that generates random linear systems given sizes of problem
(generally they are matrix size, number of right hand sides and batch
size, which is the number of systems to solve). Second one is called
*_tester and performs solution of systems from given files. In some
cases first variant is less versatile (for example, in handwritten GPU
solvers system size is fixed at compile time). But at the same time they
are more convenient and readable to get idea of the approach. Also there
are different targets for float and double variants. Each directory 
has its own Makefile and could be built separately. Dependencies (if
any) specified directly through make variables. Also see README.md in
directories to know more about presented approaches and how to build
samples.

Early versions of this code were used to obtain benchmark results
presented in conference paper [1]. 

## Current solutions

* naive_cpu_gauss - handwritten Gauss-Jordan elimination without any 
complex optimizations (sort of baseline)
* shfl_gpu_gauss - handwritten Gauss-Jordan elimination for CUDA that
makes use of CUDA register shuffle technique

## Requirements

* Tests were build and tested under GNU Linux environment. So you'll 
  need at least ``make`` utility, ``gcc`` (we tested with version 4.8.5 
  and later, but you also have to consider CUDA toolkit/gcc compatibility
  issues). We hope that tests will work on any POSIX system but we
  never really tested them. Windows OS is out of our scope, so you'll 
  have to build them manually under this OS, if you want.
* All \*_tester targets require Boost library version at least 1.64.
  (However, suppose older versions also will do.)
* For GPU solvers CUDA toolkit version at least 6.5 is required.
* Particular solutions may have their own requirements, see README.md
  in subdirectories.

## Notes

* Our original work was motivated by the problem of chemical kinetics
  implicit CUDA solver for gas dynamics. In this problem one usually
  have to solve many similarly structured small or medium size linear
  systems. For more information, see [1].
* There are situations where you need, for example, to solve system 
  with multiply different right hand sides (RHS). Moreover these RHSs
  may not be known in advance and may emerge one by one. In other
  situations one may need to explicitly invert matrix. This creates 
  many different use cases for batch solver, which were not our primary
  goal because in chem kinetics (out motivation problem) there is often
  no need to solve systems with the same matrix for different right hand
  sides (RHS) many times. That is why our tests are mainly aimed for the
  following situation: you solve one set (batch) of linear systems with
  one set of RHSs, one RHS per one matrix. 
* Because of the previous note we don't restrict methods under
  consideration just to LU decomposition with following solution 
  of triangular systems. Although, this method seems to be preferable
  from the algorithm complexity point of view (especially for multiply 
  sequentially appearing RHSs situation), in CUDA there exist different
  realization aspects which sometimes make more complex algorithm
  work faster (because of lower number of memory accesses, for example).
  That's why we also consider solutions based on Gauss elimination of
  extended matrix and even QR decomposition methods.
* It is also worth noting that solution of linear systems is not the
  same problem as LU decomposition only. As anonymous reviewer noted
  CUBLAS library [2] contains very efficient LU batch decomposition
  routine for CUDA. Even more efficient realization included in MAGMA
  library [3] (especially after 2.3 version). However CUBLAS does not
  contain routines to solve triangular systems obtained after
  decomposition. There are such routines in MAGMA but they seem 
  to be not as efficient as LU decomposition itself. So we consider
  here whole problem of linear systems solution, not just its parts.
* In chem kinetics linear systems with sparse matrices may emerge and
  because some libraries provide special treatment for sparse case,
  we also investigated dependency on matrix sparsity (i.e. fraction
  of nonzero elements). Of course, sparsity does not effect dense 
  solvers.
* Separate issue is pivoting. Look at REAMDE.md file in solution's
  subdirectory to know whether this solution have support for 
  pivoting.

## Contacts
Sorry, we are currently preparing other benchmark utilities for public
placement. Also pivoting support is not ready for handwritten solutions
yet. If you are interested in benchmark results from paper [1] 
or have some problems building tests or have better solution for the
problem, you may contact me at oleg.contacts@yandex.ru

## Связаться с нами
Извините, мы сейчас подготавливаем остальные тесты для заливки на
github. Также для небиблиотечных вариантов еще не готовы версии с 
выбором ведущего элемента. Если Вам интересны результаты тестов из
статьи или у Вас возникли проблемы со сборкой или Вы знаете лучшее
решение задачи, Вы можете связаться со мной по адресу
oleg.contacts@yandex.ru

## References
[1] Nikolay M. Evstigneev, Oleg I. Ryabkov, Eugene A. Tsatsorin 
On the Inversion of Multiple Matrices on GPU in Batched Mode //
Parallel computational technologies (PCT’2018) agora.guru.ru/pavt
<br/>
[2] https://docs.nvidia.com/cuda/cublas/index.html <br/>
[3] http://icl.cs.utk.edu/magma/ <br/>