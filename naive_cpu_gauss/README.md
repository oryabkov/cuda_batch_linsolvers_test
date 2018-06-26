## Description
Handwritten Gauss-Jordan elimination without any complex
optimizations. Used as simple baseline. There are basically
two test programs: sample and tester (although there are 
many different targets). Sample generates its own random
systems of equations, solves them and checks residual. 
Tester is used to solve preliminarily generated systems and
output results that can be later checked against known 
solution.

## Requirements

* Tests were build and tested under GNU Linux environment. So you'll 
  need at least ``make`` utility, ``gcc`` (we tested with version
  4.8.5 and later). We hope that tests will work on any POSIX system
  but we never really tested them. Windows OS is out of our scope, so
  you'll have to build them manually under this OS, if you want.
* All \*_tester targets require Boost library version at least 1.64.
  (However, suppose older versions also will do.)

## Quick start

* Change your directory to ```naive_cpu_gauss```
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
  ```./naive_cpu_gauss_float_sample 1000000 10 11 5```<br/>
  where 1000000 is the number of systems in batch,<br/>
  10 is the number of variables in the system,<br/>
  11 is sum of variables number and right hand sides number,<br/>
  5 is the number of repetitions.
* Interesting output are fields "Time per iteration" and 
  "Residual norm_C". Note that residual norm should be at least
  less than 1.
* Use ```naive_cpu_gauss_float_sample``` for single precision
  and ```naive_cpu_gauss_double_sample``` for double precision.
* Type ```./naive_cpu_gauss_float_tester -h``` to know how to 
  call the tester manually. For running tester you'll need 
  pregenerated matrix and rhs files in matrix market format.


## Notes

* No pivoting variant realization is yet added.
