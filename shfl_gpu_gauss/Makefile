NVCC                 = nvcc
CC                   = cc
CUDA_ARCH            = sm_35
NVCCFLAGS            = -O3 -DNDEBUG -arch=$(CUDA_ARCH)
CFLAGS               = -O3 -DNDEBUG
MKDIR_P              = mkdir -p
BUILD_DIR            = build
HEADER_FILES         = src/cuda_timer_event.h src/timer_event.h src/index_matrix.h src/cuda_safe_call.h src/ker_shfl_gpu_gauss.cuh
TESTERS_HEADER_FILES = src/copy_dense_to_dense.h src/copy_sparse_to_dense.h src/init_dense_sz.h src/write_vector.h src/matrix_utils.h src/shfl_gpu_gauss.cuh
#for default case when boost in standart path we set this to point to nomatterwhat dir
BOOST_INC_DIR        = ./src
BOOST_LIB_DIR        = ./src

INCLUDE_DIRS_FLAGS   = -I./src -I$(BOOST_INC_DIR)
LIBS_DIRS_FLAGS      = -L$(BOOST_LIB_DIR)

.PHONY: all samples testers clean

all: samples testers
samples: ${BUILD_DIR}/shfl_gpu_gauss_float_sample_4 ${BUILD_DIR}/shfl_gpu_gauss_double_sample_4 ${BUILD_DIR}/shfl_gpu_gauss_float_sample_11 ${BUILD_DIR}/shfl_gpu_gauss_double_sample_11 ${BUILD_DIR}/shfl_gpu_gauss_float_sample_16 ${BUILD_DIR}/shfl_gpu_gauss_double_sample_16 ${BUILD_DIR}/shfl_gpu_gauss_float_sample_31 ${BUILD_DIR}/shfl_gpu_gauss_double_sample_31
testers: ${BUILD_DIR}/shfl_gpu_gauss_float_tester ${BUILD_DIR}/shfl_gpu_gauss_double_tester

#unfortunately we have to create all targets for sample manually 
${BUILD_DIR}/shfl_gpu_gauss_float_sample_4: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=4 -DMATRIX_SZ_EXT=8 -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_double_sample_4: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=4 -DMATRIX_SZ_EXT=8 -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_float_sample_11: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=11 -DMATRIX_SZ_EXT=16 -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_double_sample_11: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=11 -DMATRIX_SZ_EXT=16 -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_float_sample_16: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=16 -DMATRIX_SZ_EXT=32 -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_double_sample_16: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=16 -DMATRIX_SZ_EXT=32 -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_float_sample_31: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=31 -DMATRIX_SZ_EXT=32 -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/shfl_gpu_gauss_double_sample_31: src/shfl_gpu_gauss_sample.cu $(HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DMATRIX_SZ=31 -DMATRIX_SZ_EXT=32 -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@


${BUILD_DIR}/mmio.o: src/mmio.c src/mmio.h
	$(CC) $(CFLAGS) -c src/mmio.c -o ${BUILD_DIR}/mmio.o
${BUILD_DIR}/shfl_gpu_gauss_float_tester: src/shfl_gpu_gauss_tester.cu $(HEADER_FILES) ${BUILD_DIR}/mmio.o src/mmio.h $(TESTERS_HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) src/shfl_gpu_gauss_tester.cu ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options
${BUILD_DIR}/shfl_gpu_gauss_double_tester: src/shfl_gpu_gauss_tester.cu $(HEADER_FILES) ${BUILD_DIR}/mmio.o src/mmio.h $(TESTERS_HEADER_FILES)
	$(NVCC) $(NVCCFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) src/shfl_gpu_gauss_tester.cu ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options

clean:
	rm -rf $(BUILD_DIR)

$(shell   $(MKDIR_P) $(BUILD_DIR))