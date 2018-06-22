CXX           = g++
CC            = cc
CXXFLAGS      = -O3 -DNDEBUG
CFLAGS        = -O3 -DNDEBUG
MKDIR_P       = mkdir -p
BUILD_DIR     = build
HEADER_FILES  = src/system_timer_event.h src/timer_event.h src/index_matrix.h src/naive_cpu_gauss.h
#for default case when boost in standart path we set this to point to nomatterwhat dir
BOOST_INC_DIR = ./src
BOOST_LIB_DIR = ./src

INCLUDE_DIRS_FLAGS = -I./src -I$(BOOST_INC_DIR)
LIBS_DIRS_FLAGS    = -L$(BOOST_LIB_DIR)

.PHONY: all samples testers clean

all: samples testers
samples: ${BUILD_DIR}/naive_cpu_gauss_float_sample ${BUILD_DIR}/naive_cpu_gauss_double_sample
testers: ${BUILD_DIR}/naive_cpu_gauss_float_tester ${BUILD_DIR}/naive_cpu_gauss_double_tester

${BUILD_DIR}/naive_cpu_gauss_float_sample: src/naive_cpu_gauss_sample.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@
${BUILD_DIR}/naive_cpu_gauss_double_sample: src/naive_cpu_gauss_sample.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) $< -o $@


${BUILD_DIR}/mmio.o: src/mmio.c src/mmio.h
	$(CC) $(CFLAGS) -c src/mmio.c -o ${BUILD_DIR}/mmio.o
${BUILD_DIR}/naive_cpu_gauss_float_tester: src/naive_cpu_gauss_tester.cpp $(HEADER_FILES) ${BUILD_DIR}/mmio.o src/mmio.h src/matrix_utils.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) src/naive_cpu_gauss_tester.cpp ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options
${BUILD_DIR}/naive_cpu_gauss_double_tester: src/naive_cpu_gauss_tester.cpp $(HEADER_FILES) ${BUILD_DIR}/mmio.o src/mmio.h src/matrix_utils.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) $(INCLUDE_DIRS_FLAGS) $(LIBS_DIRS_FLAGS) src/naive_cpu_gauss_tester.cpp ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options

clean:
	rm -rf $(BUILD_DIR)

$(shell   $(MKDIR_P) $(BUILD_DIR))