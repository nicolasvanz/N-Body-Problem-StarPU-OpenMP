# Compiler and flags
USE_MPI ?= 1
MPI_CC ?= clang
MPICC ?= OMPI_CC=$(MPI_CC) mpicc
CC ?= clang
OFFLOAD ?= 1
OMP_TARGET ?= nvptx64
GPU_ARCH ?= sm_80
DEBUG ?= 0

BASE_CFLAGS = -fopenmp -O3 -Wall -std=c99
OFFLOAD_CFLAGS_CLANG = -fopenmp-targets=$(OMP_TARGET) \
	-Xopenmp-target=$(OMP_TARGET) -march=$(GPU_ARCH)
LDFLAGS = -lm
ifeq ($(USE_MPI),1)
  CC := $(MPICC)
  CFLAGS_MPI := -DUSE_MPI=1
else
  CFLAGS_MPI := -DUSE_MPI=0
endif

CFLAGS = $(BASE_CFLAGS) $(CFLAGS_MPI)
COMPILER_ID := $(shell $(CC) --version 2>/dev/null | head -1)

ifeq ($(OFFLOAD),1)
  CFLAGS += $(OFFLOAD_CFLAGS_CLANG) -DOPENMP_OFFLOAD=1
  ifeq (,$(findstring clang,$(COMPILER_ID)))
    $(error OpenMP offload requires a clang-based compiler. Try: USE_MPI=1 OMPI_CC=clang make -C src/openmp)
  endif
  CFLAGS += -DOPTIONS_DEFAULT_MODE=MODE_GPU
else
  CFLAGS += -DOPENMP_OFFLOAD=0
  CFLAGS += -DOPTIONS_DEFAULT_MODE=MODE_CPU
endif

ifeq ($(DEBUG),1)
  CFLAGS += -DDEBUG
endif

# Source files and target
OBJS = nbody.o nbody_cpu.o nbody_gpu.o options.o debug_paths.o
TARGET = nbody
EXTRA = bin2txt

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each .c file into .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

options.o: ../common/options.c ../include/options.h
	$(CC) $(CFLAGS) -c $< -o $@

debug_paths.o: ../common/debug_paths.c ../include/debug_paths.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(EXTRA)

diff-txt:
	@gcc -O2 -o bin2txt ../bin2txt.c
	@./bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
