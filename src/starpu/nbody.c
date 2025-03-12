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
#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG
#define PARTS 4

extern void bodyForce_cpu(void *buffers[], void *_args);
extern void bodyForce_cuda(void *buffers[], void *_args);
extern void integratePositions_cpu(void *buffers[], void *_args);
extern void integratePositions_cuda(void *buffers[], void *_args);

static struct starpu_perfmodel bodyforce_perfmodel = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "bodyforce"};

static struct starpu_perfmodel integratepositions_perfmodel = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "integratepositions"};

static struct starpu_codelet bodyForce_cl = {
	.cpu_funcs = {bodyForce_cpu},

#ifdef STARPU_USE_CUDA
	.cuda_funcs = {bodyForce_cuda},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &bodyforce_perfmodel,
};

static struct starpu_codelet integratePositions_cl = {
	.cpu_funcs = {integratePositions_cpu},

#ifdef STARPU_USE_CUDA
	.cuda_funcs = {integratePositions_cuda},
#endif
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = &integratepositions_perfmodel,
};

int main(const int argc, const char **argv)
{
	int nBodies = 2 << 12;

#ifndef DEBUG
	if (argc > 1)
		nBodies = 2 << atoi(argv[1]);
#else
	(void)argc;
	(void)argv;
#endif

#ifdef DEBUG
	const char *initialized_pos = "../debug/initialized_pos_12";
	const char *initialized_vel = "../debug/initialized_vel_12";
	const char *computed_pos = "../debug/computed_pos_12";
	const char *computed_vel = "../debug/computed_vel_12";
#endif

	/* starpu configs */
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.sched_policy_name = "dmda";

	int ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	Pos *pos;
	Vel *vel;
	starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
	starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);

#ifdef DEBUG
	read_values_from_file(initialized_pos, pos, sizeof(Pos), nBodies);
	read_values_from_file(initialized_vel, vel, sizeof(Vel), nBodies);
#else
	for (int i = 0; i < nBodies; i++)
	{
		pos[i].x = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
		pos[i].y = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
		pos[i].z = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
	}
	for (int i = 0; i < nBodies; i++)
	{
		vel[i].vx = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
		vel[i].vy = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
		vel[i].vz = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
	}
#endif

	/* starpu data handles */
	starpu_data_handle_t pos_handle;
	starpu_vector_data_register(
		&pos_handle,
		STARPU_MAIN_RAM,
		(uintptr_t)pos,
		nBodies,
		sizeof(Pos));

	starpu_data_handle_t vel_handle;
	starpu_vector_data_register(
		&vel_handle,
		STARPU_MAIN_RAM,
		(uintptr_t)vel,
		nBodies,
		sizeof(Vel));

	struct starpu_data_filter filter = {
		.filter_func = starpu_vector_filter_block,
		.nchildren = PARTS};
	int *offset = (int *)malloc(sizeof(int) * PARTS);
	offset[0] = 0;
	for (int i = 1; i < PARTS; i++)
	{
		offset[i] = offset[i - 1] + nBodies / PARTS;
	}

	const int nIters = 10;
	double start = starpu_timing_now();

	starpu_data_partition(vel_handle, &filter);
	for (int i = 0; i < nIters; i++)
	{
		for (int j = 0; j < starpu_data_get_nb_children(vel_handle); j++)
		{
			ret = starpu_task_insert(
				&bodyForce_cl,
				STARPU_VALUE, &offset[j], sizeof(offset[j]),
				STARPU_R, pos_handle,
				STARPU_RW, starpu_data_get_sub_data(vel_handle, 1, j),
				0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}

		for (int j = 0; j < starpu_data_get_nb_children(vel_handle); j++)
		{
			ret = starpu_task_insert(
				&integratePositions_cl,
				STARPU_VALUE, &offset[j], sizeof(offset[j]),
				STARPU_RW, pos_handle,
				STARPU_R, starpu_data_get_sub_data(vel_handle, 1, j),
				0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
	}

	starpu_data_unpartition(vel_handle, STARPU_MAIN_RAM);
	starpu_data_unregister(pos_handle);
	starpu_data_unregister(vel_handle);

	double timing = starpu_timing_now() - start; // in microsseconds
	printf("%lf\n", timing);

#ifdef DEBUG
	write_values_to_file(computed_pos, pos, sizeof(Pos), nBodies);
	write_values_to_file(computed_vel, vel, sizeof(Vel), nBodies);
#endif

	starpu_free_noflag(pos, sizeof(Pos) * nBodies);
	starpu_free_noflag(vel, sizeof(Vel) * nBodies);

	starpu_shutdown();
}
