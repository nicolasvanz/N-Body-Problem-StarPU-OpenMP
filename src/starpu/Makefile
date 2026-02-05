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

nbody: nbody.o nbody_cpu.o options.o debug_paths.o

ifeq ($(USE_CUDA),1)
nbody: nbody_cuda.o
nbody: LDLIBS+=-L$(CUDA_PATH)/lib64 -lcudart -lstdc++
endif

options.o: ../common/options.c ../include/options.h
	$(CC) $(CFLAGS) -c $< -o $@

debug_paths.o: ../common/debug_paths.c ../include/debug_paths.h
	$(CC) $(CFLAGS) -c $< -o $@

PYTHON ?= python3
RTOL ?= 1e-3
ATOL ?= 1e-5
MAX_REPORT ?= 10

compare:
	@$(PYTHON) ../compare.py \
		--computed ../debug/computed_pos_12 \
		--solution ../debug/solution_pos_12 \
		--kind pos \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)
	@$(PYTHON) ../compare.py \
		--computed ../debug/computed_vel_12 \
		--solution ../debug/solution_vel_12 \
		--kind vel \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)

compile:
	gcc -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	@$(MAKE) compare

diff-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
