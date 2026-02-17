#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <starpu.h>

#include "nbody_runtime.h"

int nbody_run_single_tiled(const options_t *opts,
                           struct starpu_codelet *acc_init_cl,
                           struct starpu_codelet *acc_redux_cl,
                           struct starpu_codelet *bodyforce_tile_cl,
                           struct starpu_codelet *integrate_tiled_cl) {
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

    nbody_init_bodies(pos, vel, nBodies);
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
    nbody_write_debug_outputs(pos, vel, nBodies);
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
