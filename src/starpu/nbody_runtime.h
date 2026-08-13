#ifndef NBODY_STARPU_RUNTIME_H
#define NBODY_STARPU_RUNTIME_H

#include <starpu.h>

#ifndef USE_MPI
#define USE_MPI 0
#endif

#include "../../include/body.h"
#include "../../include/options.h"

#ifdef STARPU_MPI_SC
#define NBODY_MPI_MS_WORKER_KIND STARPU_MPI_SC_WORKER
#define NBODY_MPI_MS_MASK STARPU_MPI_SC
#else
#define NBODY_MPI_MS_WORKER_KIND STARPU_MPI_MS_WORKER
#define NBODY_MPI_MS_MASK STARPU_MPI_MS
#endif

void nbody_vector_filter_block(void *parent_interface,
                               void *child_interface,
                               struct starpu_data_filter *f,
                               unsigned id,
                               unsigned nchunks);

void nbody_init_bodies(Pos *pos, Vel *vel, int nBodies);
void nbody_write_debug_outputs(Pos *pos, Vel *vel, int nBodies);

int nbody_finalize_partitioned_bodyforce(struct starpu_codelet *cl, int nPartitions);
void nbody_release_partitioned_bodyforce(struct starpu_codelet *cl);

int nbody_run_single(const options_t *opts,
                     struct starpu_codelet *bodyforce_cl,
                     struct starpu_codelet *integrate_cl);

int nbody_run_master_slave_classic(const options_t *opts,
                                   struct starpu_codelet *bodyforce_cl,
                                   struct starpu_codelet *integrate_cl);

int nbody_run_single_tiled(const options_t *opts,
                           struct starpu_codelet *acc_init_cl,
                           struct starpu_codelet *acc_redux_cl,
                           struct starpu_codelet *bodyforce_tile_cl,
                           struct starpu_codelet *integrate_tiled_cl);

#if USE_MPI
int nbody_run_mpi(const options_t *opts,
                  struct starpu_codelet *bodyforce_cl,
                  struct starpu_codelet *integrate_cl,
                  int argc,
                  char **argv);

int nbody_run_mpi_tiled(const options_t *opts,
                        struct starpu_codelet *acc_init_cl,
                        struct starpu_codelet *acc_redux_cl,
                        struct starpu_codelet *bodyforce_tile_cl,
                        struct starpu_codelet *integrate_tiled_cl,
                        int argc,
                        char **argv);
#endif

#endif
