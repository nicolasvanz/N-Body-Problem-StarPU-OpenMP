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

PROGS=vector_scal_task_insert 

include ./starpu.mk

vector_scal_task_insert: vector_scal_task_insert.o vector_scal_cpu.o

ifeq ($(USE_CUDA),1)
vector_scal_task_insert: vector_scal_cuda.o
vector_scal_task_insert: LDLIBS+=-L$(CUDA_PATH)/lib64 -lcudart -lstdc++
endif
