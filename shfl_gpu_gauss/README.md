## Description
Handwritten Gauss-Jordan elimination for CUDA that expoits 
registers shuffle. The basic idea is to use one CUDA thread 
per one matrix column so all operations could be done locally 
and at the same time register pressure is significantly reduced
(at least comparing with naive realization, where one thread
processes one system). To get leading element value register 
shuffle is used. Extended matrix size (matrix size + RHSs number)
can not exceed warp size in this case (i.e. 32) which leads to
the limitation on the maximum system size. Also note that extended
matrix columns number is rounded up to the closes degree of 2,
so unnecessary FLOPs could performed which is sort of drawback
of the solution.

There are basically two test programs: sample and tester (although
there are many different targets). Sample generates its own random
systems of equations, solves them and checks residual. Tester is 
used to solve preliminarily generated systems and output results 
that can be later checked against known solution. In sample matrix
size is known at compile time, that's why there are many sample
targets one per one matrix size. In tester special wraper function 
is used to create runtime interface for different matrix sizes.

## Requirements

* Tests were build and tested under GNU Linux environment. So you'll 
  need at least ``make`` utility, ``gcc`` (we tested with version
  4.8.5 and later, but you also have to consider CUDA toolkit/gcc
  compatibility issues). We hope that tests will work on any POSIX
  system but we never really tested them. Windows OS is out of our
  scope, so you'll have to build them manually under this OS, if you
  want.
* All \*_tester targets require Boost library version at least 1.64.
  (However, suppose older versions also will do.)
* CUDA toolkit version at least 6.5 is required. CUDA device with
  Compute Capability at least 3.5 is requered (minimum CC that 
  has register shuffle).

## Quick start

* Change your directory to ```shfl_gpu_gauss```
* To build all samples type: ```make samples```<br/>
  To build all testers type: ```make testers```<br/>
  To build both type: ```make all```
* By default binaries are built in ```build``` subdirectory. You 
  can change this using make variable (for example, by typing
  ```make BUILD_DIR = build_test all```).
* For building tester BOOST library is needed. If it's not in
  standart search paths you need to add them as make variables
  (for example, ```make BOOST_INC_DIR = /path/to/boost/headers
  BOOST_LIB_DIR = /path/to/boost/libs testers```)
* Change your current directory to build directory to run tests
  by typing: ```cd build``` (or however you called your build
  directory).
* To run sample type: 
  ```./shfl_gpu_gauss_float_sample_11 0 1000000 5```<br/>
  where 0 is the CUDA device number,<br/>
  1000000 is the number of systems in batch,<br/>
  _11 suffix in sample name means the number of variables in the
  system,<br/>
  5 is the number of repetitions.<br/>
  Note that the number of RHSs is at least 1, however because
  of the realization real number of RHSs is bigger. For 11
  variables it is 16-11=5. For 16 variables it whould be 32-16=16,
  which is equivalent of matrix inversion.
* Interesting output are fields "Time per iteration" and 
  "Residual norm_C". Note that residual norm should be at least
  less than 1.
* Use ```shfl_gpu_gauss_float_sample_XX``` for single precision
  and ```shfl_gpu_gauss_double_sample_XX``` for double precision.
* Type ```./shfl_gpu_gauss_float_tester -h``` to know how to 
  call the tester manually. For running tester you'll need 
  pregenerated matrix and rhs files in matrix market format.


## Notes

* No pivoting variant realization is yet added.
