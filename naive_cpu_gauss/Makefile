CXX          = g++
CC           = cc
CXXFLAGS     = -O3 -DNDEBUG
CFLAGS       = -O3 -DNDEBUG
MKDIR_P      = mkdir -p
BUILD_DIR    = build
HEADER_FILES = src/system_timer_event.h src/timer_event.h src/index_matrix.h src/naive_cpu_gauss.h

.PHONY: all samples testers clean

all: samples testers
samples: ${BUILD_DIR}/naive_cpu_gauss_float_sample ${BUILD_DIR}/naive_cpu_gauss_double_sample
testers: ${BUILD_DIR}/naive_cpu_gauss_float_tester ${BUILD_DIR}/naive_cpu_gauss_double_tester

${BUILD_DIR}/naive_cpu_gauss_float_sample: src/naive_cpu_gauss_sample.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) -I./src $< -o $@
${BUILD_DIR}/naive_cpu_gauss_double_sample: src/naive_cpu_gauss_sample.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) -I./src $< -o $@


${BUILD_DIR}/mmio.o: src/mmio.c src/mmio.h
	$(CC) $(CFLAGS) -c src/mmio.c -o ${BUILD_DIR}/mmio.o
${BUILD_DIR}/naive_cpu_gauss_float_tester: src/naive_cpu_gauss_tester.cpp ${BUILD_DIR}/mmio.o $(HEADER_FILES) src/mmio.h src/matrix_utils.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) -I./src src/naive_cpu_gauss_tester.cpp ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options
${BUILD_DIR}/naive_cpu_gauss_double_tester: src/naive_cpu_gauss_tester.cpp $(HEADER_FILES) ${BUILD_DIR}/mmio.o src/matrix_utils.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) -I./src src/naive_cpu_gauss_tester.cpp ${BUILD_DIR}/mmio.o -o $@ -lboost_program_options

clean:
	rm -rf $(BUILD_DIR)

$(shell   $(MKDIR_P) $(BUILD_DIR))