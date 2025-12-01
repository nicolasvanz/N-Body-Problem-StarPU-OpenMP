/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800),
 * Inria
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
#include <starpu_mpi.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG

extern void bodyForce_tile_cpu(void *buffers[], void *_args);
// extern void bodyForce_cuda(void *buffers[], void *_args);
extern void integratePositions_cpu(void *buffers[], void *_args);
// extern void integratePositions_cuda(void *buffers[], void *_args);

static void acc_init_cpu(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned int n = STARPU_VECTOR_GET_NX(buffers[0]);

    for (unsigned i = 0; i < n; i++) {
        a[i].vx = 0.0f;
        a[i].vy = 0.0f;
        a[i].vz = 0.0f;
    }
}

static void acc_redux_cpu(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    Vel *dst = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]); // STARPU_RW
    Vel *src = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]); // STARPU_R
    unsigned int n = STARPU_VECTOR_GET_NX(buffers[0]);

    for (unsigned i = 0; i < n; i++) {
        dst[i].vx += src[i].vx;
        dst[i].vy += src[i].vy;
        dst[i].vz += src[i].vz;
    }
}

static struct starpu_codelet acc_init_cl = {
    .cpu_funcs = { acc_init_cpu },
    .nbuffers  = 1,
    .modes     = { STARPU_W },
};

static struct starpu_codelet acc_redux_cl = {
    .cpu_funcs = { acc_redux_cpu },
    .nbuffers  = 2,
    .modes     = { STARPU_RW, STARPU_R },
};


/* body force */
static struct starpu_perfmodel bodyforce_tile_perfmodel = {
    .type   = STARPU_HISTORY_BASED,
    .symbol = "bodyforce_tile",
};
static struct starpu_codelet bodyForce_tile_cl = {
    .cpu_funcs = { bodyForce_tile_cpu },
    .nbuffers  = 3,
    .modes     = { STARPU_R, STARPU_R, STARPU_MPI_REDUX }, // pos_I, pos_J, acc_I
    .model     = &bodyforce_tile_perfmodel,
};

/* integrate positions */
static struct starpu_perfmodel integratepositions_perfmodel = {
    .type   = STARPU_HISTORY_BASED,
    .symbol = "integratepositions",
};

static struct starpu_codelet integratePositions_cl = {
    .cpu_funcs = { integratePositions_cpu },
    .nbuffers  = 3,
    .modes     = { STARPU_RW, STARPU_RW, STARPU_R }, // pos_I, vel_I, acc_I
    .model     = &integratepositions_perfmodel,
};

int main(int argc, char **argv) {
    int rank, size, ret;
    int nBodies = 2 << 12;

    if (argc > 1)
        nBodies = 2 << (atoi(argv[1]) - 1);

    setbuf(stdout, NULL);
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, &conf);
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

    int nPartitions = size * starpu_worker_get_count();

    if (nBodies % nPartitions != 0 && rank == 0) {
        fprintf(stderr,
            "WARNING: nBodies (%d) not divisible by nPartitions (%d). "
            "This code assumes equal-sized partitions.\n",
            nBodies, nPartitions);
    }

    Pos *pos;
    Vel *vel;
    Vel *acc;

    if (rank == 0) {
        starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
        starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);
        starpu_malloc((void **)&acc, sizeof(Vel) * nBodies);

        for (int i = 0; i < nBodies; i++) {
            pos[i].x = ((float)rand() / (float)(RAND_MAX)) * 1000000.0f;
            pos[i].y = ((float)rand() / (float)(RAND_MAX)) * 1000000.0f;
            pos[i].z = ((float)rand() / (float)(RAND_MAX)) * 1000000.0f;
            vel[i].vx = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
            vel[i].vy = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
            vel[i].vz = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
            acc[i].vx = 0.0f;
            acc[i].vy = 0.0f;
            acc[i].vz = 0.0f;
        }
    }

    // vectors are allocated in rank 0
    int memory_region = (rank == 0) ? STARPU_MAIN_RAM : -1;
    uintptr_t pos_ptr = (rank == 0) ? (uintptr_t)pos : 0;
    uintptr_t vel_ptr = (rank == 0) ? (uintptr_t)vel : 0;
    uintptr_t acc_ptr = (rank == 0) ? (uintptr_t)acc : 0;

    starpu_data_handle_t pos_handle, vel_handle, acc_handle;

    starpu_vector_data_register(
        &pos_handle, memory_region, pos_ptr, nBodies, sizeof(Pos)
    );
    starpu_vector_data_register(
        &vel_handle, memory_region, vel_ptr, nBodies, sizeof(Vel)
    );
    starpu_vector_data_register(
        &acc_handle, memory_region, acc_ptr, nBodies, sizeof(Vel)
    );

    starpu_data_set_reduction_methods(acc_handle, &acc_redux_cl, &acc_init_cl);

    starpu_data_handle_t *pos_handles = (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) * nPartitions);
    starpu_data_handle_t *vel_handles = (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) * nPartitions);
    starpu_data_handle_t *acc_handles = (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) * nPartitions);

    struct starpu_data_filter filter = {
        .filter_func = starpu_vector_filter_block,
        .nchildren = nPartitions
    };

    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    starpu_data_partition_plan(acc_handle, &filter, acc_handles);

    starpu_mpi_tag_t tag = 0;

    starpu_mpi_data_register(pos_handle, tag++, 0);
    starpu_mpi_data_register(vel_handle, tag++, 0);
    starpu_mpi_data_register(acc_handle, tag++, 0);

    for (int i = 0; i < nPartitions; i++) {
        starpu_mpi_data_register(pos_handles[i], tag++, 0);
        starpu_mpi_data_register(vel_handles[i], tag++, 0);
        starpu_mpi_data_register(acc_handles[i], tag++, 0);
    }

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int iter = 0; iter < nIters; iter++) {
        for (int i = 0; i < nPartitions; i++) {
            for (int j = 0; j < nPartitions; j++) {
                starpu_mpi_task_insert(
                    MPI_COMM_WORLD,
                    &bodyForce_tile_cl,
                    STARPU_R, pos_handles[i],
                    STARPU_R, pos_handles[j],
                    STARPU_MPI_REDUX, acc_handles[i],
                    STARPU_EXECUTE_ON_NODE, i % size
                );
            }
        }

        for (int i = 0; i < nPartitions; i++) {
            starpu_mpi_task_insert(
                MPI_COMM_WORLD, 
                &integratePositions_cl,
                STARPU_RW, pos_handles[i],
                STARPU_RW, vel_handles[i],
                STARPU_R, acc_handles[i],
                STARPU_EXECUTE_ON_NODE, i % size
            );
        }
    }
    starpu_task_wait_for_all();
    
    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
    starpu_data_unpartition_submit(acc_handle, nPartitions, acc_handles, -1);
    
    starpu_task_wait_for_all();
    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);
    starpu_data_partition_clean(acc_handle, nPartitions, acc_handles);

    if (rank == 0) {
        starpu_data_acquire(pos_handle, STARPU_R);
        starpu_data_acquire(vel_handle, STARPU_R);
        pos = starpu_data_get_local_ptr(pos_handle);
        vel = starpu_data_get_local_ptr(vel_handle);
        double timing = starpu_timing_now() - start; // in microsseconds
        printf("%lf\n", timing);
        starpu_data_release(pos_handle);
        starpu_data_release(vel_handle);
    }

    starpu_data_unregister(pos_handle);
    starpu_data_unregister(vel_handle);
    starpu_data_unregister(acc_handle);

    if (rank == 0) {
        starpu_free_noflag(pos, sizeof(Pos) * nBodies);
        starpu_free_noflag(vel, sizeof(Vel) * nBodies);
        starpu_free_noflag(acc, sizeof(Vel) * nBodies);
    }
    free(pos_handles);
    free(vel_handles);
    free(acc_handles);

    starpu_mpi_shutdown();
}
