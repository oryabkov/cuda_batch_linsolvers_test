#ifndef __CUDA_SAFE_CALL_H__
#define __CUDA_SAFE_CALL_H__

#include <stdexcept>
#include <string>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define CUDA_SAFE_CALL(X)																						\
	do {                        																					\
		cudaError_t cuda_res = (X);                                                                                            									\
		if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUDA_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") + std::string(cudaGetErrorString(cuda_res)));	\
	} while (0)

#endif
