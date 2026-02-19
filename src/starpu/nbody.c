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

#include "nbody_runtime.h"

#include "../include/debug_paths.h"
#include "../include/files.h"

extern void bodyForce_cpu(void *buffers[], void *_args);
extern void integratePositions_cpu(void *buffers[], void *_args);
extern void clearAcceleration_cpu(void *buffers[], void *_args);
extern void reduceAcceleration_cpu(void *buffers[], void *_args);
extern void bodyForce_tile_cpu(void *buffers[], void *_args);
extern void integratePositions_tiled_cpu(void *buffers[], void *_args);
#if NBODY_USE_CUDA
extern void clearAcceleration_cuda(void *buffers[], void *_args);
extern void reduceAcceleration_cuda(void *buffers[], void *_args);
extern void bodyForce_cuda(void *buffers[], void *_args);
extern void integratePositions_cuda(void *buffers[], void *_args);
extern void bodyForce_tile_cuda(void *buffers[], void *_args);
extern void integratePositions_tiled_cuda(void *buffers[], void *_args);
#endif

static struct starpu_perfmodel bodyforce_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "bodyforce"};

static struct starpu_perfmodel integratepositions_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "integratepositions"};

static struct starpu_perfmodel bodyforce_tile_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "bodyforce_tile"};

static struct starpu_perfmodel integratepositions_tiled_perfmodel = {
    .type = STARPU_HISTORY_BASED, .symbol = "integratepositions_tiled"};

void nbody_vector_filter_block(void *parent_interface,
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

void nbody_init_bodies(Pos *pos, Vel *vel, int nBodies) {
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

void nbody_write_debug_outputs(Pos *pos, Vel *vel, int nBodies) {
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
    memset(acc_init_cl, 0, sizeof(*acc_init_cl));
    memset(acc_redux_cl, 0, sizeof(*acc_redux_cl));
    memset(bodyforce_tile_cl, 0, sizeof(*bodyforce_tile_cl));
    memset(integrate_tiled_cl, 0, sizeof(*integrate_tiled_cl));

    acc_init_cl->nbuffers = 1;
    acc_init_cl->modes[0] = STARPU_W;

    acc_redux_cl->nbuffers = 2;
    acc_redux_cl->modes[0] = STARPU_RW | STARPU_COMMUTE;
    acc_redux_cl->modes[1] = STARPU_R;

    bodyforce_tile_cl->nbuffers = 3;
    bodyforce_tile_cl->modes[0] = STARPU_R;
    bodyforce_tile_cl->modes[1] = STARPU_R;
    bodyforce_tile_cl->modes[2] = (backend == BACKEND_MPI)
                                      ? (STARPU_RW | STARPU_COMMUTE)
                                      : STARPU_REDUX;
    bodyforce_tile_cl->model = &bodyforce_tile_perfmodel;

    integrate_tiled_cl->nbuffers = 3;
    integrate_tiled_cl->modes[0] = STARPU_RW;
    integrate_tiled_cl->modes[1] = STARPU_RW;
    integrate_tiled_cl->modes[2] = STARPU_R;
    integrate_tiled_cl->model = &integratepositions_tiled_perfmodel;

    if (mode == MODE_CPU) {
        acc_init_cl->cpu_funcs[0] = clearAcceleration_cpu;
        acc_redux_cl->cpu_funcs[0] = reduceAcceleration_cpu;
        bodyforce_tile_cl->cpu_funcs[0] = bodyForce_tile_cpu;
        integrate_tiled_cl->cpu_funcs[0] = integratePositions_tiled_cpu;
        acc_init_cl->where = STARPU_CPU;
        acc_redux_cl->where = STARPU_CPU;
        bodyforce_tile_cl->where = STARPU_CPU;
        integrate_tiled_cl->where = STARPU_CPU;
        return 0;
    }

#if NBODY_USE_CUDA
    if (mode == MODE_GPU) {
        acc_init_cl->cuda_funcs[0] = clearAcceleration_cuda;
        acc_redux_cl->cuda_funcs[0] = reduceAcceleration_cuda;
        bodyforce_tile_cl->cuda_funcs[0] = bodyForce_tile_cuda;
        integrate_tiled_cl->cuda_funcs[0] = integratePositions_tiled_cuda;
        acc_init_cl->where = STARPU_CUDA;
        acc_redux_cl->where = STARPU_CUDA;
        bodyforce_tile_cl->where = STARPU_CUDA;
        integrate_tiled_cl->where = STARPU_CUDA;
        return 0;
    }

    if (mode == MODE_HYBRID) {
        acc_init_cl->cpu_funcs[0] = clearAcceleration_cpu;
        acc_redux_cl->cpu_funcs[0] = reduceAcceleration_cpu;
        bodyforce_tile_cl->cpu_funcs[0] = bodyForce_tile_cpu;
        integrate_tiled_cl->cpu_funcs[0] = integratePositions_tiled_cpu;

        acc_init_cl->cuda_funcs[0] = clearAcceleration_cuda;
        acc_redux_cl->cuda_funcs[0] = reduceAcceleration_cuda;
        bodyforce_tile_cl->cuda_funcs[0] = bodyForce_tile_cuda;
        integrate_tiled_cl->cuda_funcs[0] = integratePositions_tiled_cuda;

        acc_init_cl->where = 0;
        acc_redux_cl->where = 0;
        bodyforce_tile_cl->where = 0;
        integrate_tiled_cl->where = 0;
        return 0;
    }
#else
    (void)mode;
#endif

    return -1;
}

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
            return nbody_run_mpi_tiled(&opts,
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

        return nbody_run_single_tiled(
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
        return nbody_run_mpi(&opts, &bodyforce_cl, &integrate_cl, argc, argv);
#else
        fprintf(stderr, "ERROR: MPI backend requested but binary built without MPI.\n");
        return 1;
#endif
    }

    return nbody_run_single(&opts, &bodyforce_cl, &integrate_cl);
}
