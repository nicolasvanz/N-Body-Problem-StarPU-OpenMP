#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <starpu.h>

#include "nbody_runtime.h"

typedef struct {
    int nBodies;
    int nPartitions;
    int starpu_initialized;
    int partitions_planned;
    int partitions_cleaned;
    Pos *pos;
    Vel *vel;
    starpu_data_handle_t pos_handle;
    starpu_data_handle_t vel_handle;
    starpu_data_handle_t *pos_handles;
    starpu_data_handle_t *vel_handles;
    int *remote_worker_ids;
    int nRemoteWorkers;
} ms_context_t;

static int parse_int_env(const char *name, int default_value) {
    const char *value = getenv(name);
    if (value == NULL || *value == '\0') {
        return default_value;
    }

    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }
    return (int)parsed;
}

static int current_mpi_rank(void) {
    const char *rank_vars[] = {
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
        "MPI_RANKID",
        NULL,
    };

    for (int i = 0; rank_vars[i] != NULL; i++) {
        const char *value = getenv(rank_vars[i]);
        if (value == NULL || *value == '\0') {
            continue;
        }
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        if (end != value && *end == '\0') {
            return (int)parsed;
        }
    }

    return -1;
}

static int worker_matches_mode(int workerid, compute_mode_t mode) {
#ifdef STARPU_MPI_SC
    int is_cuda_lane = starpu_mpi_sc_worker_is_cuda((unsigned)workerid);
    if (mode == MODE_CPU) {
        return !is_cuda_lane;
    }
    if (mode == MODE_GPU) {
        return is_cuda_lane;
    }
    return 1;
#else
    (void)workerid;
    return mode != MODE_GPU;
#endif
}

static void ms_context_cleanup(ms_context_t *ctx) {
    if (ctx->starpu_initialized) {
        if (ctx->partitions_planned && !ctx->partitions_cleaned &&
            ctx->pos_handles != NULL && ctx->vel_handles != NULL &&
            ctx->pos_handle != 0 && ctx->vel_handle != 0 && ctx->nPartitions > 0) {
            starpu_task_wait_for_all();
            starpu_data_partition_clean(
                ctx->pos_handle, ctx->nPartitions, ctx->pos_handles);
            starpu_data_partition_clean(
                ctx->vel_handle, ctx->nPartitions, ctx->vel_handles);
        }
    }

    if (ctx->pos_handle != 0) {
        starpu_data_unregister(ctx->pos_handle);
    }
    if (ctx->vel_handle != 0) {
        starpu_data_unregister(ctx->vel_handle);
    }
    if (ctx->pos != NULL) {
        starpu_free_noflag(ctx->pos, sizeof(Pos) * ctx->nBodies);
    }
    if (ctx->vel != NULL) {
        starpu_free_noflag(ctx->vel, sizeof(Vel) * ctx->nBodies);
    }

    free(ctx->pos_handles);
    free(ctx->vel_handles);
    free(ctx->remote_worker_ids);

    if (ctx->starpu_initialized) {
        starpu_shutdown();
    }
}

static int ms_collect_workers(compute_mode_t mode,
                              int **remote_worker_ids,
                              int *nremote_workers) {
    unsigned n_workers =
        starpu_worker_get_count_by_type(NBODY_MPI_MS_WORKER_KIND);
    if (n_workers == 0) {
        *remote_worker_ids = NULL;
        *nremote_workers = 0;
        return 0;
    }

    int *all_worker_ids = (int *)malloc(sizeof(int) * n_workers);
    if (all_worker_ids == NULL) {
        return -1;
    }

    unsigned fetched = starpu_worker_get_ids_by_type(
        NBODY_MPI_MS_WORKER_KIND, all_worker_ids, n_workers);
    int *filtered = (int *)malloc(sizeof(int) * fetched);
    if (filtered == NULL) {
        free(all_worker_ids);
        return -1;
    }

    int count = 0;
    for (unsigned i = 0; i < fetched; i++) {
        int wid = all_worker_ids[i];
        if (worker_matches_mode(wid, mode)) {
            filtered[count++] = wid;
        }
    }

    free(all_worker_ids);
    *remote_worker_ids = filtered;
    *nremote_workers = count;
    return 0;
}

/* Local (source-side) workers able to run this mode. These are ordinary
 * STARPU_CPU/CUDA workers on the source, distinct from the MS sink lanes
 * counted by ms_collect_workers. Under dmda the scheduler can place tasks on
 * them (the codelet .where already lists STARPU_CPU/STARPU_CUDA), so a source
 * with local workers can compute even with zero sinks. */
static int ms_count_local_workers(compute_mode_t mode) {
    int n = 0;
    if (mode == MODE_CPU || mode == MODE_HYBRID) {
        n += (int)starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    }
    if (mode == MODE_GPU || mode == MODE_HYBRID) {
        n += (int)starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    }
    return n;
}

int nbody_run_master_slave_classic(const options_t *opts,
                                   struct starpu_codelet *bodyforce_cl,
                                   struct starpu_codelet *integrate_cl) {
    int ret = 0;
    int rank = current_mpi_rank();
    int server_rank = parse_int_env("STARPU_MPI_SERVER_NODE", 0);

    ms_context_t ctx = {
        .nBodies = opts->nBodies,
        .nPartitions = 0,
        .starpu_initialized = 0,
        .partitions_planned = 0,
        .partitions_cleaned = 0,
        .pos = NULL,
        .vel = NULL,
        .pos_handle = 0,
        .vel_handle = 0,
        .pos_handles = NULL,
        .vel_handles = NULL,
        .remote_worker_ids = NULL,
        .nRemoteWorkers = 0,
    };

#ifdef DEBUG
    ctx.nBodies = 2 << 12;
    if (rank == server_rank || rank < 0) {
        printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
    }
#endif

    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.sched_policy_name = "dmda";

    do {
        ret = starpu_init(&conf);
        if (ret == -ENODEV) {
            ret = 77;
            break;
        }
        if (ret != 0) {
            fprintf(stderr, "ERROR: starpu_init failed: %d\n", ret);
            ret = 1;
            break;
        }
        ctx.starpu_initialized = 1;

        if (rank >= 0 && rank != server_rank) {
            ret = 0;
            break;
        }

        int use_dmda = parse_int_env("NBODY_SC_SCHED_DMDA", 0);

        ret = ms_collect_workers(
            opts->mode, &ctx.remote_worker_ids, &ctx.nRemoteWorkers);
        if (ret != 0) {
            fprintf(stderr, "ERROR: allocation failed while collecting workers\n");
            ret = 1;
            break;
        }

        int nLocalCapable = ms_count_local_workers(opts->mode);

        if (ctx.nRemoteWorkers == 0) {
            /* No sinks. The STARPU_EXECUTE_ON_WORKER path pins each task to a
             * remote lane (and would divide by zero here), so source-only
             * execution is valid only under dmda. */
            if (!use_dmda) {
                fprintf(stderr,
                        "ERROR: no remote MPI_SC lanes; source-only execution "
                        "requires dmda (set NBODY_SC_SCHED_DMDA=1).\n");
                ret = 1;
                break;
            }
            /* Truly invalid: no sinks AND no local workers for this mode (e.g.
             * a 2-core source that reserves both cores for coordination and
             * keeps zero CPU workers). Nothing can run. */
            if (nLocalCapable == 0) {
                fprintf(stderr,
                        "ERROR: no MPI_SC lanes and no local workers for mode %d; "
                        "source cannot compute (check STARPU_MPI_SC_N* / source cores).\n",
                        (int)opts->mode);
                ret = 1;
                break;
            }
            /* else: source-only baseline -- run all partitions on local
             * workers via dmda (the c=1 source-on point). */
        }

        int default_parts =
            ctx.nRemoteWorkers > 0 ? ctx.nRemoteWorkers : nLocalCapable;
        ctx.nPartitions = opts->partitions_set ? opts->nPartitions : default_parts;
        if (ctx.nPartitions <= 0 || ctx.nPartitions > ctx.nBodies) {
            fprintf(stderr,
                    "ERROR: invalid partition count %d (valid range: 1..%d)\n",
                    ctx.nPartitions,
                    ctx.nBodies);
            ret = 1;
            break;
        }

        starpu_malloc((void **)&ctx.pos, sizeof(Pos) * ctx.nBodies);
        starpu_malloc((void **)&ctx.vel, sizeof(Vel) * ctx.nBodies);
        nbody_init_bodies(ctx.pos, ctx.vel, ctx.nBodies);

        starpu_vector_data_register(&ctx.pos_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)ctx.pos,
                                    ctx.nBodies,
                                    sizeof(Pos));
        starpu_vector_data_register(&ctx.vel_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)ctx.vel,
                                    ctx.nBodies,
                                    sizeof(Vel));

        ctx.pos_handles = (starpu_data_handle_t *)malloc(
            sizeof(starpu_data_handle_t) * ctx.nPartitions);
        ctx.vel_handles = (starpu_data_handle_t *)malloc(
            sizeof(starpu_data_handle_t) * ctx.nPartitions);
        if (ctx.pos_handles == NULL || ctx.vel_handles == NULL) {
            fprintf(stderr, "ERROR: allocation failed for partition handles\n");
            ret = 1;
            break;
        }

        struct starpu_data_filter filter = {
            .filter_func = nbody_vector_filter_block, .nchildren = ctx.nPartitions};
        starpu_data_partition_plan(ctx.pos_handle, &filter, ctx.pos_handles);
        starpu_data_partition_plan(ctx.vel_handle, &filter, ctx.vel_handles);
        ctx.partitions_planned = 1;

        const int nIters = 10;
        double start = starpu_timing_now();

        for (int iter = 0; iter < nIters && ret == 0; iter++) {
            for (int j = 0; j < ctx.nPartitions; j++) {
                int ins;
                if (use_dmda)
                    ins = starpu_task_insert(bodyforce_cl,
                        STARPU_R, ctx.pos_handle,
                        STARPU_RW, ctx.vel_handles[j],
                        0);
                else
                    ins = starpu_task_insert(bodyforce_cl,
                        STARPU_EXECUTE_ON_WORKER,
                        ctx.remote_worker_ids[j % ctx.nRemoteWorkers],
                        STARPU_R, ctx.pos_handle,
                        STARPU_RW, ctx.vel_handles[j],
                        0);
                ret = ins;
                if (ret != 0) {
                    fprintf(stderr,
                            "ERROR: bodyforce task submission failed (%d)\n",
                            ret);
                    break;
                }
            }

            for (int j = 0; j < ctx.nPartitions && ret == 0; j++) {
                int ins;
                if (use_dmda)
                    ins = starpu_task_insert(integrate_cl,
                        STARPU_RW, ctx.pos_handles[j],
                        STARPU_R, ctx.vel_handles[j],
                        0);
                else
                    ins = starpu_task_insert(integrate_cl,
                        STARPU_EXECUTE_ON_WORKER,
                        ctx.remote_worker_ids[j % ctx.nRemoteWorkers],
                        STARPU_RW, ctx.pos_handles[j],
                        STARPU_R, ctx.vel_handles[j],
                        0);
                ret = ins;
                if (ret != 0) {
                    fprintf(stderr,
                            "ERROR: integrate task submission failed (%d)\n",
                            ret);
                    break;
                }
            }
        }
        if (ret != 0) {
            break;
        }

        ret = starpu_task_wait_for_all();
        if (ret != 0) {
            fprintf(stderr, "ERROR: starpu_task_wait_for_all failed (%d)\n", ret);
            break;
        }

        starpu_data_unpartition_submit(
            ctx.vel_handle, ctx.nPartitions, ctx.vel_handles, -1);
        starpu_data_unpartition_submit(
            ctx.pos_handle, ctx.nPartitions, ctx.pos_handles, -1);
        ret = starpu_task_wait_for_all();
        if (ret != 0) {
            fprintf(stderr,
                    "ERROR: starpu_task_wait_for_all after unpartition failed (%d)\n",
                    ret);
            break;
        }

        starpu_data_partition_clean(
            ctx.pos_handle, ctx.nPartitions, ctx.pos_handles);
        starpu_data_partition_clean(
            ctx.vel_handle, ctx.nPartitions, ctx.vel_handles);
        ctx.partitions_cleaned = 1;

        starpu_data_acquire(ctx.pos_handle, STARPU_R);
        starpu_data_acquire(ctx.vel_handle, STARPU_R);
        ctx.pos = starpu_data_get_local_ptr(ctx.pos_handle);
        ctx.vel = starpu_data_get_local_ptr(ctx.vel_handle);
        printf("%lf\n", starpu_timing_now() - start);
        nbody_write_debug_outputs(ctx.pos, ctx.vel, ctx.nBodies);
        starpu_data_release(ctx.pos_handle);
        starpu_data_release(ctx.vel_handle);
    } while (0);

    ms_context_cleanup(&ctx);
    return ret;
}
