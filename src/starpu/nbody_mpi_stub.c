#include <stdio.h>

#include "nbody_runtime.h"

int nbody_run_mpi(const options_t *opts,
                  struct starpu_codelet *bodyforce_cl,
                  struct starpu_codelet *integrate_cl,
                  int argc,
                  char **argv) {
    (void)opts;
    (void)bodyforce_cl;
    (void)integrate_cl;
    (void)argc;
    (void)argv;
    fprintf(stderr, "ERROR: MPI backend requested but binary built without MPI.\n");
    return 1;
}

int nbody_run_mpi_tiled(const options_t *opts,
                        struct starpu_codelet *acc_init_cl,
                        struct starpu_codelet *acc_redux_cl,
                        struct starpu_codelet *bodyforce_tile_cl,
                        struct starpu_codelet *integrate_tiled_cl,
                        int argc,
                        char **argv) {
    (void)opts;
    (void)acc_init_cl;
    (void)acc_redux_cl;
    (void)bodyforce_tile_cl;
    (void)integrate_tiled_cl;
    (void)argc;
    (void)argv;
    fprintf(stderr, "ERROR: MPI backend requested but binary built without MPI.\n");
    return 1;
}
