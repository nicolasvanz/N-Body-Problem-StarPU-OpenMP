STARPU_VERSION ?= 1.4
USE_MPI ?= 1
DEBUG ?= 0

NVCC = nvcc

ifeq ($(USE_MPI),1)
  CC = mpicc
  CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-$(STARPU_VERSION))
  LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION) --libs starpumpi-$(STARPU_VERSION))
  NVCCFLAGS = $(shell pkg-config --cflags starpu-$(STARPU_VERSION) --cflags starpumpi-$(STARPU_VERSION)) -std=c++11
  CFLAGS += -DUSE_MPI=1
else
  CC ?= cc
  CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
  LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
  NVCCFLAGS = $(shell pkg-config --cflags starpu-$(STARPU_VERSION)) -std=c++11
  CFLAGS += -DUSE_MPI=0
endif

CFLAGS += -O3 -Wall -Wextra -lm -fopenmp

ifeq ($(DEBUG),1)
  CFLAGS += -DDEBUG
  NVCCFLAGS += -DDEBUG
endif

ifeq ($(USE_CUDA),1)
  CFLAGS += -DOPTIONS_DEFAULT_MODE=MODE_GPU
else
  CFLAGS += -DOPTIONS_DEFAULT_MODE=MODE_CPU
endif

# to avoid having to use LD_LIBRARY_PATH
LDLIBS += -fopenmp -lm -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

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
	$(NVCC) -ccbin $(CC) $(NVCCFLAGS) $< -c -o $@

all: $(PROGS)

clean:
	rm -f $(PROGS) *.o */*.o */*/*.o
	rm -f paje.trace dag.dot *.rec trace.html starpu.log
	rm -f *.gp *.eps *.data
	rm -f bin2txt
