#ifndef NBODY_OPTIONS_H
#define NBODY_OPTIONS_H

typedef enum {
    MODE_CPU = 0,
    MODE_GPU = 1,
    MODE_HYBRID = 2,
} compute_mode_t;

typedef enum {
    BACKEND_SINGLE = 0,
    BACKEND_MPI = 1,
} backend_t;

typedef struct {
    int nBodies;
    compute_mode_t mode;
    backend_t backend;
    int show_help;
    int backend_set;
    int mode_set;
} options_t;

void print_usage(const char *prog);
int parse_options(int argc, char **argv, options_t *opts);
int mode_uses_gpu(compute_mode_t mode);

#endif
