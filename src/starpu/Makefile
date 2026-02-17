# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

PROGS=nbody

include ./starpu.mk

NBODY_OBJS := nbody.o nbody_single.o nbody_tiled.o nbody_cpu.o options.o debug_paths.o
ifeq ($(USE_MPI),1)
NBODY_OBJS += nbody_mpi.o nbody_tiled_mpi.o
else
NBODY_OBJS += nbody_mpi_stub.o
endif
NBODY_LINK_OBJS := $(NBODY_OBJS)

.PHONY: FORCE
FORCE:

ifeq ($(USE_CUDA),1)
NBODY_LINK_OBJS += nbody_cuda.o
nbody: LDLIBS+=-L$(CUDA_LIBDIR) -Wl,-rpath -Wl,$(CUDA_LIBDIR) -lcudart -lstdc++
endif

nbody: $(NBODY_LINK_OBJS) FORCE
	$(CC) $(NBODY_LINK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

options.o: ../common/options.c ../include/options.h
	$(CC) $(CFLAGS) -c $< -o $@

debug_paths.o: ../common/debug_paths.c ../include/debug_paths.h
	$(CC) $(CFLAGS) -c $< -o $@

diff-txt:
	@gcc -O2 -o bin2txt ../bin2txt.c
	@./bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
