STARPU_VERSION=1.4

CC = gcc
NVCC = nvcc

CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))

CFLAGS += -O3 -Wall -Wextra
NVCCFLAGS = $(shell pkg-config --cflags starpu-$(STARPU_VERSION)) -std=c++11

# to avoid having to use LD_LIBRARY_PATH
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

# Automatically enable CUDA / OpenCL
STARPU_CONFIG=$(shell pkg-config --variable=includedir starpu-$(STARPU_VERSION))/starpu/$(STARPU_VERSION)/starpu_config.h
ifneq ($(shell grep "STARPU_USE_CUDA 1" $(STARPU_CONFIG)),)
USE_CUDA=1
endif
# ifneq ($(shell grep "STARPU_USE_OPENCL 1" $(STARPU_CONFIG)),)
# USE_OPENCL=1
# endif
# ifneq ($(shell grep "STARPU_SIMGRID 1" $(STARPU_CONFIG)),)
# USE_SIMGRID=1
# endif

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data