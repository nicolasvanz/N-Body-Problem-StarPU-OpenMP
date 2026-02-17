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
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <starpu.h>

#ifndef USE_MPI
#define USE_MPI 0
#endif

#ifndef NBODY_USE_CUDA
#ifdef STARPU_USE_CUDA
#define NBODY_USE_CUDA 1
#else
#define NBODY_USE_CUDA 0
#endif
#endif

#if NBODY_USE_CUDA && !defined(STARPU_USE_CUDA)
#error "NBODY_USE_CUDA=1 requires a StarPU build with CUDA support."
#endif

#if USE_MPI
#include <starpu_mpi.h>
#endif

#include "../include/body.h"
#include "../include/debug_paths.h"
#include "../include/files.h"
#include "../include/options.h"

// #define DEBUG

extern void bodyForce_cpu(void *buffers[], void *_args);
extern void integratePositions_cpu(void *buffers[], void *_args);
extern void clearAcceleration_cpu(void *buffers[], void *_args);
extern void reduceAcceleration_cpu(void *buffers[], void *_args);
extern void bodyForce_tile_cpu(void *buffers[], void *_args);
extern void integratePositions_tiled_cpu(void *buffers[], void *_args);
#if NBODY_USE_CUDA
extern void bodyForce_cuda(void *buffers[], void *_args);
extern void integratePositions_cuda(void *buffers[], void *_args);
#endif

static struct starpu_perfmodel bodyforce_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "bodyforce"};

static struct starpu_perfmodel integratepositions_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "integratepositions"};

static struct starpu_perfmodel bodyforce_tile_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "bodyforce_tile"};

static struct starpu_perfmodel integratepositions_tiled_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "integratepositions_tiled"};

static void nbody_vector_filter_block(void *parent_interface,
                                      void *child_interface,
                                      struct starpu_data_filter *f,
                                      unsigned id,
                                      unsigned nchunks) {
    (void)f;

    struct starpu_vector_interface *vector_parent =
        (struct starpu_vector_interface *)parent_interface;
    struct starpu_vector_interface *vector_child =
        (struct starpu_vector_interface *)child_interface;

    unsigned child_nx = 0;
    size_t child_offset = 0;
    starpu_filter_nparts_compute_chunk_size_and_offset(
        (unsigned)vector_parent->nx,
        nchunks,
        vector_parent->elemsize,
        id,
        1,
        &child_nx,
        &child_offset);

    vector_child->id = vector_parent->id;
    vector_child->nx = child_nx;
    vector_child->elemsize = vector_parent->elemsize;
    vector_child->allocsize = child_nx * vector_parent->elemsize;
    /*
        currently starpu uses slice_base for openmp handling. as we don't use
        starpu + openmp, we use slice_base as a offset variable that is 
        reliable for mpi environments. STARPU_VECTOR_GET_OFFSET seems to be
        unstable at mpi environments: apparently we can lose offset information
        when distributing or gathering subvectors to/from other nodes
    */
    vector_child->slice_base =
        vector_parent->slice_base + child_offset / vector_parent->elemsize;

    if (vector_parent->dev_handle) {
        if (vector_parent->ptr) {
            vector_child->ptr = vector_parent->ptr + child_offset;
        }
        vector_child->dev_handle = vector_parent->dev_handle;
        vector_child->offset = vector_parent->offset + child_offset;
    }
}


static int configure_codelets(compute_mode_t mode,
                              struct starpu_codelet *bodyforce_cl,
                              struct starpu_codelet *integrate_cl) {
    memset(bodyforce_cl, 0, sizeof(*bodyforce_cl));
    memset(integrate_cl, 0, sizeof(*integrate_cl));

    bodyforce_cl->nbuffers = 2;
    bodyforce_cl->modes[0] = STARPU_R;
    bodyforce_cl->modes[1] = STARPU_RW;
    bodyforce_cl->model = &bodyforce_perfmodel;

    integrate_cl->nbuffers = 2;
    integrate_cl->modes[0] = STARPU_RW;
    integrate_cl->modes[1] = STARPU_R;
    integrate_cl->model = &integratepositions_perfmodel;

    if (mode == MODE_CPU) {
        bodyforce_cl->cpu_funcs[0] = bodyForce_cpu;
        integrate_cl->cpu_funcs[0] = integratePositions_cpu;
        bodyforce_cl->where = STARPU_CPU;
        integrate_cl->where = STARPU_CPU;
        return 0;
    }

#if NBODY_USE_CUDA
    if (mode == MODE_GPU) {
        bodyforce_cl->cuda_funcs[0] = bodyForce_cuda;
        integrate_cl->cuda_funcs[0] = integratePositions_cuda;
        bodyforce_cl->where = STARPU_CUDA;
        integrate_cl->where = STARPU_CUDA;
        return 0;
    }
    if (mode == MODE_HYBRID) {
        bodyforce_cl->cpu_funcs[0] = bodyForce_cpu;
        integrate_cl->cpu_funcs[0] = integratePositions_cpu;
        bodyforce_cl->cuda_funcs[0] = bodyForce_cuda;
        integrate_cl->cuda_funcs[0] = integratePositions_cuda;
        bodyforce_cl->where = 0;
        integrate_cl->where = 0;
        return 0;
    }
#else
    (void)mode;
#endif

    return -1;
}

static int configure_tiled_codelets(backend_t backend,
                                    compute_mode_t mode,
                                    struct starpu_codelet *acc_init_cl,
                                    struct starpu_codelet *acc_redux_cl,
                                    struct starpu_codelet *bodyforce_tile_cl,
                                    struct starpu_codelet *integrate_tiled_cl) {
    (void)backend;
    if (mode != MODE_CPU) {
        return -1;
    }

    memset(acc_init_cl, 0, sizeof(*acc_init_cl));
    memset(acc_redux_cl, 0, sizeof(*acc_redux_cl));
    memset(bodyforce_tile_cl, 0, sizeof(*bodyforce_tile_cl));
    memset(integrate_tiled_cl, 0, sizeof(*integrate_tiled_cl));

    acc_init_cl->cpu_funcs[0] = clearAcceleration_cpu;
    acc_init_cl->nbuffers = 1;
    acc_init_cl->modes[0] = STARPU_W;
    acc_init_cl->where = STARPU_CPU;

    acc_redux_cl->cpu_funcs[0] = reduceAcceleration_cpu;
    acc_redux_cl->nbuffers = 2;
    acc_redux_cl->modes[0] = STARPU_RW | STARPU_COMMUTE;
    acc_redux_cl->modes[1] = STARPU_R;
    acc_redux_cl->where = STARPU_CPU;

    bodyforce_tile_cl->cpu_funcs[0] = bodyForce_tile_cpu;
    bodyforce_tile_cl->nbuffers = 3;
    bodyforce_tile_cl->modes[0] = STARPU_R;
    bodyforce_tile_cl->modes[1] = STARPU_R;
    bodyforce_tile_cl->modes[2] = STARPU_RW | STARPU_COMMUTE;
    bodyforce_tile_cl->where = STARPU_CPU;
    bodyforce_tile_cl->model = &bodyforce_tile_perfmodel;

    integrate_tiled_cl->cpu_funcs[0] = integratePositions_tiled_cpu;
    integrate_tiled_cl->nbuffers = 3;
    integrate_tiled_cl->modes[0] = STARPU_RW;
    integrate_tiled_cl->modes[1] = STARPU_RW;
    integrate_tiled_cl->modes[2] = STARPU_R;
    integrate_tiled_cl->where = STARPU_CPU;
    integrate_tiled_cl->model = &integratepositions_tiled_perfmodel;

    return 0;
}

static void init_bodies(Pos *pos, Vel *vel, int nBodies) {
#ifdef DEBUG
    char initialized_pos[PATH_MAX];
    char initialized_vel[PATH_MAX];
    make_debug_path(initialized_pos, sizeof(initialized_pos), "initialized_pos_12");
    make_debug_path(initialized_vel, sizeof(initialized_vel), "initialized_vel_12");
    read_values_from_file(initialized_pos, pos, sizeof(Pos), nBodies);
    read_values_from_file(initialized_vel, vel, sizeof(Vel), nBodies);
#else
    for (int i = 0; i < nBodies; i++) {
        pos[i].x = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
        pos[i].y = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
        pos[i].z = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
    }
    for (int i = 0; i < nBodies; i++) {
        vel[i].vx = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
        vel[i].vy = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
        vel[i].vz = ((float)rand() / (float)(RAND_MAX)) * 10.0f;
    }
#endif
}

static void write_debug_outputs(Pos *pos, Vel *vel, int nBodies) {
#ifdef DEBUG
    char computed_pos[PATH_MAX];
    char computed_vel[PATH_MAX];
    make_debug_path(computed_pos, sizeof(computed_pos), "computed_pos_12");
    make_debug_path(computed_vel, sizeof(computed_vel), "computed_vel_12");
    write_values_to_file(computed_pos, pos, sizeof(Pos), nBodies);
    write_values_to_file(computed_vel, vel, sizeof(Vel), nBodies);
#else
    (void)pos;
    (void)vel;
    (void)nBodies;
#endif
}

static int run_single(const options_t *opts,
                      const struct starpu_codelet *bodyforce_cl,
                      const struct starpu_codelet *integrate_cl) {
    int ret = 0;
    int nBodies = opts->nBodies;
    Pos *pos;
    Vel *vel;

#ifdef DEBUG
    nBodies = 2 << 12;
    printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
#endif

    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    starpu_init(&conf);

    int nPartitions = opts->partitions_set ? opts->nPartitions : starpu_worker_get_count();
    if (nPartitions <= 0 || nPartitions > nBodies) {
        fprintf(stderr,
                "ERROR: invalid partition count %d (valid range: 1..%d)\n",
                nPartitions,
                nBodies);
        starpu_shutdown();
        return 1;
    }

    starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
    starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);

    init_bodies(pos, vel, nBodies);

    int memory_region = STARPU_MAIN_RAM;
    uintptr_t pos_ptr = (uintptr_t)pos;
    uintptr_t vel_ptr = (uintptr_t)vel;

    starpu_data_handle_t pos_handle;
    starpu_vector_data_register(
        &pos_handle, memory_region, pos_ptr, nBodies, sizeof(Pos));
    starpu_data_handle_t *pos_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    starpu_data_handle_t vel_handle;
    starpu_vector_data_register(
        &vel_handle, memory_region, vel_ptr, nBodies, sizeof(Vel));
    starpu_data_handle_t *vel_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    struct starpu_data_filter filter = {
        .filter_func = nbody_vector_filter_block, .nchildren = nPartitions};
    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    starpu_data_partition_submit(vel_handle, nPartitions, vel_handles);

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int i = 0; i < nIters; i++) {
        starpu_data_partition_submit(pos_handle, nPartitions, pos_handles);

        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_task_insert(bodyforce_cl,
                                     STARPU_R,
                                     pos_handle,
                                     STARPU_RW,
                                     vel_handles[j],
                                     0);
        }
        starpu_task_wait_for_all();

        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_task_insert(integrate_cl,
                                     STARPU_RW,
                                     pos_handles[j],
                                     STARPU_R,
                                     vel_handles[j],
                                     0);
        }
        starpu_task_wait_for_all();

        starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
        starpu_task_wait_for_all();
    }

    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_task_wait_for_all();
    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);

    starpu_data_acquire(pos_handle, STARPU_R);
    starpu_data_acquire(vel_handle, STARPU_R);
    pos = starpu_data_get_local_ptr(pos_handle);
    vel = starpu_data_get_local_ptr(vel_handle);
    double timing = starpu_timing_now() - start; // in microseconds
    printf("%lf\n", timing);

    write_debug_outputs(pos, vel, nBodies);

    starpu_data_release(pos_handle);
    starpu_data_release(vel_handle);

    starpu_data_unregister(pos_handle);
    starpu_data_unregister(vel_handle);

    starpu_free_noflag(pos, sizeof(Pos) * nBodies);
    starpu_free_noflag(vel, sizeof(Vel) * nBodies);
    free(pos_handles);
    free(vel_handles);

    starpu_shutdown();
    return ret;
}

static int run_single_tiled(const options_t *opts,
                            const struct starpu_codelet *acc_init_cl,
                            const struct starpu_codelet *acc_redux_cl,
                            const struct starpu_codelet *bodyforce_tile_cl,
                            const struct starpu_codelet *integrate_tiled_cl) {
    int ret = 0;
    int nBodies = opts->nBodies;
    Pos *pos;
    Vel *vel;
    Vel *acc;

#ifdef DEBUG
    nBodies = 2 << 12;
    printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
#endif

    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    starpu_init(&conf);

    int nPartitions =
        opts->partitions_set ? opts->nPartitions : starpu_worker_get_count();
    if (nPartitions <= 0 || nPartitions > nBodies) {
        fprintf(stderr,
                "ERROR: invalid partition count %d (valid range: 1..%d)\n",
                nPartitions,
                nBodies);
        starpu_shutdown();
        return 1;
    }

    starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
    starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);
    starpu_malloc((void **)&acc, sizeof(Vel) * nBodies);

    init_bodies(pos, vel, nBodies);
    memset(acc, 0, sizeof(Vel) * nBodies);

    starpu_data_handle_t pos_handle;
    starpu_data_handle_t vel_handle;
    starpu_data_handle_t acc_handle;
    starpu_vector_data_register(
        &pos_handle, STARPU_MAIN_RAM, (uintptr_t)pos, nBodies, sizeof(Pos));
    starpu_vector_data_register(
        &vel_handle, STARPU_MAIN_RAM, (uintptr_t)vel, nBodies, sizeof(Vel));
    starpu_vector_data_register(
        &acc_handle, STARPU_MAIN_RAM, (uintptr_t)acc, nBodies, sizeof(Vel));

    starpu_data_handle_t *pos_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);
    starpu_data_handle_t *vel_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);
    starpu_data_handle_t *acc_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);

    struct starpu_data_filter filter = {
        .filter_func = nbody_vector_filter_block, .nchildren = nPartitions};
    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    starpu_data_partition_plan(acc_handle, &filter, acc_handles);

    for (int i = 0; i < nPartitions; i++) {
        starpu_data_set_reduction_methods(
            acc_handles[i], acc_redux_cl, acc_init_cl);
    }

    starpu_data_partition_submit(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_submit(vel_handle, nPartitions, vel_handles);
    starpu_data_partition_submit(acc_handle, nPartitions, acc_handles);

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int iter = 0; iter < nIters; iter++) {
        for (int i = 0; i < nPartitions; i++) {
            ret = starpu_task_insert(
                acc_init_cl, STARPU_W, acc_handles[i], 0);
        }

        for (int i = 0; i < nPartitions; i++) {
            for (int j = 0; j < nPartitions; j++) {
                ret = starpu_task_insert(bodyforce_tile_cl,
                                         STARPU_R,
                                         pos_handles[i],
                                         STARPU_R,
                                         pos_handles[j],
                                         STARPU_REDUX,
                                         acc_handles[i],
                                         0);
            }
        }

        for (int i = 0; i < nPartitions; i++) {
            ret = starpu_task_insert(integrate_tiled_cl,
                                     STARPU_RW,
                                     pos_handles[i],
                                     STARPU_RW,
                                     vel_handles[i],
                                     STARPU_R,
                                     acc_handles[i],
                                     0);
        }
    }

    starpu_task_wait_for_all();
    starpu_data_unpartition_submit(acc_handle, nPartitions, acc_handles, -1);
    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
    starpu_task_wait_for_all();

    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);
    starpu_data_partition_clean(acc_handle, nPartitions, acc_handles);

    starpu_data_acquire(pos_handle, STARPU_R);
    starpu_data_acquire(vel_handle, STARPU_R);
    pos = starpu_data_get_local_ptr(pos_handle);
    vel = starpu_data_get_local_ptr(vel_handle);
    double timing = starpu_timing_now() - start; // in microseconds
    printf("%lf\n", timing);
    write_debug_outputs(pos, vel, nBodies);
    starpu_data_release(pos_handle);
    starpu_data_release(vel_handle);

    starpu_data_unregister(pos_handle);
    starpu_data_unregister(vel_handle);
    starpu_data_unregister(acc_handle);

    starpu_free_noflag(pos, sizeof(Pos) * nBodies);
    starpu_free_noflag(vel, sizeof(Vel) * nBodies);
    starpu_free_noflag(acc, sizeof(Vel) * nBodies);
    free(pos_handles);
    free(vel_handles);
    free(acc_handles);

    starpu_shutdown();
    return ret;
}

#if USE_MPI
static int run_mpi(const options_t *opts,
                   const struct starpu_codelet *bodyforce_cl,
                   const struct starpu_codelet *integrate_cl,
                   int argc,
                   char **argv) {
    int rank, size, ret = 0, nPartitions;
    int nBodies = opts->nBodies;
    Pos *pos = NULL;
    Vel *vel = NULL;
    starpu_mpi_tag_t tag = 0;

    setbuf(stdout, NULL);
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, &conf);
    starpu_mpi_cache_set(0);
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    nBodies = 2 << 12;
    if (rank == 0) {
        printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
    }
#endif

    nPartitions = opts->partitions_set ? opts->nPartitions : size * starpu_worker_get_count();
    if (nPartitions <= 0 || nPartitions > nBodies) {
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: invalid partition count %d (valid range: 1..%d)\n",
                    nPartitions,
                    nBodies);
        }
        starpu_mpi_shutdown();
        return 1;
    }

    if (rank == 0) {
        starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
        starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);
        init_bodies(pos, vel, nBodies);
    }

    int memory_region = (rank == 0) ? STARPU_MAIN_RAM : -1;
    uintptr_t pos_ptr = (rank == 0) ? (uintptr_t)pos : 0;
    uintptr_t vel_ptr = (rank == 0) ? (uintptr_t)vel : 0;

    starpu_data_handle_t pos_handle;
    starpu_vector_data_register(
        &pos_handle, memory_region, pos_ptr, nBodies, sizeof(Pos));
    starpu_data_handle_t *pos_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    starpu_data_handle_t vel_handle;
    starpu_vector_data_register(
        &vel_handle, memory_region, vel_ptr, nBodies, sizeof(Vel));
    starpu_data_handle_t *vel_handles = (starpu_data_handle_t *)malloc(
        sizeof(starpu_data_handle_t) * nPartitions);

    starpu_mpi_data_register(pos_handle, tag++, 0);
    starpu_mpi_data_register(vel_handle, tag++, 0);

    struct starpu_data_filter filter = {
        .filter_func = nbody_vector_filter_block, .nchildren = nPartitions};
    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    for (int i = 0; i < nPartitions; i++) {
        starpu_mpi_data_register(vel_handles[i], tag++, 0);
        starpu_mpi_data_register(pos_handles[i], tag++, 0);
    }

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int i = 0; i < nIters; i++) {
        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         bodyforce_cl,
                                         STARPU_R,
                                         pos_handle,
                                         STARPU_RW,
                                         vel_handles[j],
                                         STARPU_EXECUTE_ON_NODE,
                                         j % size,
                                         0);
        }

        for (int j = 0; j < nPartitions; j++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         integrate_cl,
                                         STARPU_RW,
                                         pos_handles[j],
                                         STARPU_R,
                                         vel_handles[j],
                                         STARPU_EXECUTE_ON_NODE,
                                         j % size,
                                         0);
        }
    }

    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);

    starpu_mpi_get_data_on_node(MPI_COMM_WORLD, pos_handle, 0);
    starpu_mpi_get_data_on_node(MPI_COMM_WORLD, vel_handle, 0);

    if (rank == 0) {
        starpu_data_acquire(pos_handle, STARPU_R);
        starpu_data_acquire(vel_handle, STARPU_R);
        pos = starpu_data_get_local_ptr(pos_handle);
        vel = starpu_data_get_local_ptr(vel_handle);
        double timing = starpu_timing_now() - start; // in microseconds
        printf("%lf\n", timing);
        write_debug_outputs(pos, vel, nBodies);
        starpu_data_release(pos_handle);
        starpu_data_release(vel_handle);
    }

    starpu_data_unregister(pos_handle);
    starpu_data_unregister(vel_handle);

    if (rank == 0) {
        starpu_free_noflag(pos, sizeof(Pos) * nBodies);
        starpu_free_noflag(vel, sizeof(Vel) * nBodies);
    }
    free(pos_handles);
    free(vel_handles);

    starpu_mpi_shutdown();
    return ret;
}

static int run_mpi_tiled(const options_t *opts,
                         const struct starpu_codelet *acc_init_cl,
                         const struct starpu_codelet *acc_redux_cl,
                         const struct starpu_codelet *bodyforce_tile_cl,
                         const struct starpu_codelet *integrate_tiled_cl,
                         int argc,
                         char **argv) {
    int rank, size, ret = 0, nPartitions;
    int nBodies = opts->nBodies;
    Pos *pos = NULL;
    Vel *vel = NULL;
    Vel *acc = NULL;
    starpu_mpi_tag_t tag = 0;

    setbuf(stdout, NULL);
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, &conf);
    starpu_mpi_cache_set(0);
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    nBodies = 2 << 12;
    if (rank == 0) {
        printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
    }
#endif

    nPartitions =
        opts->partitions_set ? opts->nPartitions : size * starpu_worker_get_count();
    if (nPartitions <= 0 || nPartitions > nBodies) {
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: invalid partition count %d (valid range: 1..%d)\n",
                    nPartitions,
                    nBodies);
        }
        starpu_mpi_shutdown();
        return 1;
    }

    if (rank == 0) {
        starpu_malloc((void **)&pos, sizeof(Pos) * nBodies);
        starpu_malloc((void **)&vel, sizeof(Vel) * nBodies);
        starpu_malloc((void **)&acc, sizeof(Vel) * nBodies);
        init_bodies(pos, vel, nBodies);
        memset(acc, 0, sizeof(Vel) * nBodies);
    }

    int memory_region = (rank == 0) ? STARPU_MAIN_RAM : -1;
    uintptr_t pos_ptr = (rank == 0) ? (uintptr_t)pos : 0;
    uintptr_t vel_ptr = (rank == 0) ? (uintptr_t)vel : 0;
    uintptr_t acc_ptr = (rank == 0) ? (uintptr_t)acc : 0;

    starpu_data_handle_t pos_handle;
    starpu_data_handle_t vel_handle;
    starpu_data_handle_t acc_handle;
    starpu_vector_data_register(
        &pos_handle, memory_region, pos_ptr, nBodies, sizeof(Pos));
    starpu_vector_data_register(
        &vel_handle, memory_region, vel_ptr, nBodies, sizeof(Vel));
    starpu_vector_data_register(
        &acc_handle, memory_region, acc_ptr, nBodies, sizeof(Vel));

    starpu_data_handle_t *pos_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);
    starpu_data_handle_t *vel_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);
    starpu_data_handle_t *acc_handles =
        (starpu_data_handle_t *)malloc(sizeof(starpu_data_handle_t) *
                                       nPartitions);

    starpu_mpi_data_register(pos_handle, tag++, 0);
    starpu_mpi_data_register(vel_handle, tag++, 0);
    starpu_mpi_data_register(acc_handle, tag++, 0);

    struct starpu_data_filter filter = {
        .filter_func = nbody_vector_filter_block, .nchildren = nPartitions};
    starpu_data_partition_plan(pos_handle, &filter, pos_handles);
    starpu_data_partition_plan(vel_handle, &filter, vel_handles);
    starpu_data_partition_plan(acc_handle, &filter, acc_handles);

    for (int i = 0; i < nPartitions; i++) {
        starpu_mpi_data_register(pos_handles[i], tag++, 0);
        starpu_mpi_data_register(vel_handles[i], tag++, 0);
        starpu_mpi_data_register(acc_handles[i], tag++, 0);
        starpu_data_set_reduction_methods(
            acc_handles[i], acc_redux_cl, acc_init_cl);
    }

    const int nIters = 10;
    double start = starpu_timing_now();

    for (int iter = 0; iter < nIters; iter++) {
        for (int i = 0; i < nPartitions; i++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         acc_init_cl,
                                         STARPU_W,
                                         acc_handles[i],
                                         STARPU_EXECUTE_ON_NODE,
                                         i % size,
                                         0);
        }

        for (int i = 0; i < nPartitions; i++) {
            for (int j = 0; j < nPartitions; j++) {
                ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                             bodyforce_tile_cl,
                                             STARPU_R,
                                             pos_handles[i],
                                             STARPU_R,
                                             pos_handles[j],
                                             STARPU_MPI_REDUX,
                                             acc_handles[i],
                                             STARPU_EXECUTE_ON_NODE,
                                             i % size,
                                             0);
            }
        }

        for (int i = 0; i < nPartitions; i++) {
            ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
                                         integrate_tiled_cl,
                                         STARPU_RW,
                                         pos_handles[i],
                                         STARPU_RW,
                                         vel_handles[i],
                                         STARPU_R,
                                         acc_handles[i],
                                         STARPU_EXECUTE_ON_NODE,
                                         i % size,
                                         0);
        }
    }

    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_data_unpartition_submit(acc_handle, nPartitions, acc_handles, -1);
    starpu_data_unpartition_submit(vel_handle, nPartitions, vel_handles, -1);
    starpu_data_unpartition_submit(pos_handle, nPartitions, pos_handles, -1);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    starpu_data_partition_clean(pos_handle, nPartitions, pos_handles);
    starpu_data_partition_clean(vel_handle, nPartitions, vel_handles);
    starpu_data_partition_clean(acc_handle, nPartitions, acc_handles);

    starpu_mpi_get_data_on_node(MPI_COMM_WORLD, pos_handle, 0);
    starpu_mpi_get_data_on_node(MPI_COMM_WORLD, vel_handle, 0);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    if (rank == 0) {
        starpu_data_acquire(pos_handle, STARPU_R);
        starpu_data_acquire(vel_handle, STARPU_R);
        pos = starpu_data_get_local_ptr(pos_handle);
        vel = starpu_data_get_local_ptr(vel_handle);
        double timing = starpu_timing_now() - start; // in microseconds
        printf("%lf\n", timing);
        write_debug_outputs(pos, vel, nBodies);
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
    return ret;
}
#endif

int main(int argc, char **argv) {
    init_debug_paths(argv[0]);

    options_t opts = {
        .nBodies = 2 << 12,
        .nPartitions = 0,
        .mode = MODE_CPU,
        .backend = BACKEND_SINGLE,
        .algorithm = ALGO_CLASSIC,
        .show_help = 0,
        .backend_set = 0,
        .mode_set = 0,
        .partitions_set = 0,
        .algorithm_set = 0,
    };

    if (parse_options(argc, argv, &opts) != 0 || opts.show_help) {
        print_usage(argv[0]);
        return opts.show_help ? 0 : 1;
    }

    if (mode_uses_gpu(opts.mode)) {
#if !NBODY_USE_CUDA
        fprintf(stderr, "ERROR: GPU/Hybrid mode requires StarPU with CUDA.\n");
        return 1;
#endif
    }

    if (opts.algorithm == ALGO_TILED && opts.mode != MODE_CPU) {
        fprintf(stderr, "ERROR: tiled algorithm currently supports only CPU mode.\n");
        return 1;
    }

    if (opts.algorithm == ALGO_TILED) {
        struct starpu_codelet acc_init_cl;
        struct starpu_codelet acc_redux_cl;
        struct starpu_codelet bodyforce_tile_cl;
        struct starpu_codelet integrate_tiled_cl;

        if (configure_tiled_codelets(opts.backend,
                                     opts.mode,
                                     &acc_init_cl,
                                     &acc_redux_cl,
                                     &bodyforce_tile_cl,
                                     &integrate_tiled_cl) != 0) {
            fprintf(stderr, "ERROR: failed to configure tiled StarPU codelets.\n");
            return 1;
        }

        if (opts.backend == BACKEND_MPI) {
#if USE_MPI
            return run_mpi_tiled(&opts,
                                 &acc_init_cl,
                                 &acc_redux_cl,
                                 &bodyforce_tile_cl,
                                 &integrate_tiled_cl,
                                 argc,
                                 argv);
#else
            fprintf(stderr,
                    "ERROR: MPI backend requested but binary built without MPI.\n");
            return 1;
#endif
        }

        return run_single_tiled(
            &opts, &acc_init_cl, &acc_redux_cl, &bodyforce_tile_cl, &integrate_tiled_cl);
    }

    struct starpu_codelet bodyforce_cl;
    struct starpu_codelet integrate_cl;
    if (configure_codelets(opts.mode, &bodyforce_cl, &integrate_cl) != 0) {
        fprintf(stderr, "ERROR: failed to configure StarPU codelets.\n");
        return 1;
    }

    if (opts.backend == BACKEND_MPI) {
#if USE_MPI
        return run_mpi(&opts, &bodyforce_cl, &integrate_cl, argc, argv);
#else
        fprintf(stderr, "ERROR: MPI backend requested but binary built without MPI.\n");
        return 1;
#endif
    }

    return run_single(&opts, &bodyforce_cl, &integrate_cl);
}
