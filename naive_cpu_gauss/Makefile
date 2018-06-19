CXX       = g++
CXXFLAGS  = -O3 -DNDEBUG
MKDIR_P   = mkdir -p
BUILD_DIR = build

.PHONY: all clean

all: ${BUILD_DIR}/naive_cpu_gauss_float_sample ${BUILD_DIR}/naive_cpu_gauss_double_sample

${BUILD_DIR}/naive_cpu_gauss_float_sample: src/naive_cpu_gauss_sample.cpp src/system_timer_event.h src/timer_event.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=float $(LDFLAGS) -I./src $< -o $@
${BUILD_DIR}/naive_cpu_gauss_double_sample: src/naive_cpu_gauss_sample.cpp src/system_timer_event.h src/timer_event.h
	$(CXX) $(CXXFLAGS) -DSCALAR_TYPE=double $(LDFLAGS) -I./src $< -o $@

clean:
	rm -rf $(BUILD_DIR)

$(shell   $(MKDIR_P) $(BUILD_DIR))