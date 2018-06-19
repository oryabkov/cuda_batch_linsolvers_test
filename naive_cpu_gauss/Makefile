CXX       = g++
CXXFLAGS  = -O3 -DNDEBUG
MKDIR_P   = mkdir -p
BUILD_DIR = build

.PHONY: all clean

all: ${BUILD_DIR}/naive_cpu_gauss_sample

${BUILD_DIR}/naive_cpu_gauss_sample: src/naive_cpu_gauss_sample.cpp src/system_timer_event.h src/timer_event.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I./src $< -o $@

clean:
	rm -rf $(BUILD_DIR)

$(shell   $(MKDIR_P) $(BUILD_DIR))