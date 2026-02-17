#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <starpu.h>

#ifndef USE_MPI
#define USE_MPI 0
#endif

#if USE_MPI
#include <starpu_mpi.h>

#include "nbody_runtime.h"

int nbody_run_mpi_tiled(const options_t *opts,
                        struct starpu_codelet *acc_init_cl,
                        struct starpu_codelet *acc_redux_cl,
                        struct starpu_codelet *bodyforce_tile_cl,
                        struct starpu_codelet *integrate_tiled_cl,
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
        nbody_init_bodies(pos, vel, nBodies);
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
        nbody_write_debug_outputs(pos, vel, nBodies);
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
