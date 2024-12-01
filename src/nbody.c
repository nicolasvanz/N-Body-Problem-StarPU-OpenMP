/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to submit a task to StarPU
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */
#include <starpu.h>

#define    NX    1<<30

extern void vector_scal_cpu(void *buffers[], void *_args);
extern void vector_scal_cuda(void *buffers[], void *_args);

static struct starpu_perfmodel perfmodel = {
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "vector_scal"
};

static struct starpu_codelet cl = {
	/* CPU implementation of the codelet */
	.cpu_funcs = {vector_scal_cpu},

#ifdef STARPU_USE_CUDA
	/* CUDA implementation of the codelet */
	.cuda_funcs = {vector_scal_cuda},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},

	.model = &perfmodel,
};

int main(void)
{
	/* We consider a vector of float that is initialized just as any of C
	 * data */
	float *vector;
	double start_time;
	unsigned i;

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	vector = malloc(sizeof(vector[0]) * NX);
	for (i = 0; i < NX; i++)
		vector[i] = 1.0f;

	fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);

	/* Tell StaPU to associate the "vector" vector with the "vector_handle"
	 * identifier. When a task needs to access a piece of data, it should
	 * refer to the handle that is associated to it.
	 * In the case of the "vector" data interface:
	 *  - the first argument of the registration method is a pointer to the
	 *    handle that should describe the data
	 *  - the second argument is the memory node where the data (ie. "vector")
	 *    resides initially: 0 stands for an address in main memory, as
	 *    opposed to an adress on a GPU for instance.
	 *  - the third argument is the adress of the vector in RAM
	 *  - the fourth argument is the number of elements in the vector
	 *  - the fifth argument is the size of each element.
	 */
	starpu_data_handle_t vector_handle;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector,
				    NX, sizeof(vector[0]));

	float factor = 3.14;

	start_time = starpu_timing_now();
	ret = starpu_task_insert(&cl,
				 /* an argument is passed to the codelet, beware that this is a
				  * READ-ONLY buffer and that the codelet may be given a pointer to a
				  * COPY of the argument */
				 STARPU_VALUE, &factor, sizeof(factor),
				 /* the codelet manipulates one buffer in RW mode */
				 STARPU_RW, vector_handle,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Wait for tasks completion */
	starpu_task_wait_for_all();
	fprintf(stderr, "computation took %fµs\n", starpu_timing_now() - start_time);

	/* StarPU does not need to manipulate the array anymore so we can stop
	 * monitoring it */
	starpu_data_unregister(vector_handle);

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);
	free(vector);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	return 0;
}
