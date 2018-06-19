#Description
This repository contains small set of tests aimed to compare
different approaches for solution of large number of small linear
systems on GPU (so called batch mode). Each directory contains
separate solution approach. Each directory generally contains two
programs. First one is called \*_sample which is sample that generates
random linear systems given sizes of problem (generally they are
matrix size, number of right hand sides and batch size, which is the
number of systems to solve). Second one is called *_test and performes
solution of systems from given files. In some cases first variant is
less versatile (for example, in handwritten GPU solvers system size 
is fixed at compile time). But at the same time they are more 
convinient and readable to get idea of the approach. Each directory 
has its own Makefile and could be built separatly. Dependencies (if
any) specified directly through make variables. Also see README.md in
directories to know more about presented approaches and how to build
samples.

#Current solutions

* naive_cpu_gauss - handwritten gauss-jordan elimination without any 
complex optimizations (sort of baseline)
* naive_gpu_gauss - handwritten gauss-jordan elimination for CUDA
* shfl_gpu_gauss - handwritten gauss-jordan elimination for CUDA that
makes use of CUDA register shuffle technique

#Requirements

* All \*_test targets require boost library.

#Notes
Sorry, we are currently preparing other benchmark utils for public
placement. If you are interested in benchmark results from paper, you
may contact me at oleg.contacts@yandex.ru

Извините, мы сейчас подготавливаем остальные тесты для заливки на
github. Если Вам интересны результаты тестов из статьи, Вы можете
связаться со мной по адресу oleg.contacts@yandex.ru