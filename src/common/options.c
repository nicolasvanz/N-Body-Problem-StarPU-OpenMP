#include "../include/options.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef USE_MPI
#define USE_MPI 0
#endif

#ifndef OPTIONS_DEFAULT_MODE
#define OPTIONS_DEFAULT_MODE MODE_CPU
#endif

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0 || v > INT_MAX) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_mode(const char *s, compute_mode_t *mode) {
    if (strcmp(s, "cpu") == 0) {
        *mode = MODE_CPU;
        return 1;
    }
    if (strcmp(s, "gpu") == 0) {
        *mode = MODE_GPU;
        return 1;
    }
    if (strcmp(s, "hybrid") == 0) {
        *mode = MODE_HYBRID;
        return 1;
    }
    return 0;
}

static int parse_backend(const char *s, backend_t *backend) {
    if (strcmp(s, "single") == 0) {
        *backend = BACKEND_SINGLE;
        return 1;
    }
    if (strcmp(s, "mpi") == 0) {
        *backend = BACKEND_MPI;
        return 1;
    }
    return 0;
}

static int env_indicates_mpi(void) {
    const char *vars[] = {
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMI_RANK",
        "MV2_COMM_WORLD_SIZE",
        "MPI_LOCALRANKID",
        NULL,
    };
    for (int i = 0; vars[i] != NULL; i++) {
        if (getenv(vars[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

static compute_mode_t default_mode(void) {
    return OPTIONS_DEFAULT_MODE;
}

static backend_t default_backend(void) {
#if USE_MPI
    if (env_indicates_mpi()) {
        return BACKEND_MPI;
    }
#endif
    return BACKEND_SINGLE;
}

void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options] [exp]\n"
            "\n"
            "Options:\n"
            "  -n, --n <count>         Number of bodies (absolute value)\n"
            "  --exp <e>               Legacy exponent (nBodies = 2 << e)\n"
            "  -b, --backend <type>    Backend: mpi or single\n"
            "  --mpi                   Shorthand for --backend mpi\n"
            "  --single                Shorthand for --backend single\n"
            "  -m, --mode <mode>       Mode: cpu, gpu, or hybrid\n"
            "  -h, --help              Show this help\n"
            "\n"
            "Legacy:\n"
            "  If a positional numeric argument is provided, it is treated as\n"
            "  an exponent (nBodies = 2 << exp) for backward compatibility.\n",
            prog);
}

int parse_options(int argc, char **argv, options_t *opts) {
    int n_direct = -1;
    int exp = -1;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            opts->show_help = 1;
            return 0;
        }
        if (strcmp(arg, "-n") == 0 || strcmp(arg, "--n") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &n_direct)) {
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "--exp") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &exp)) {
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "-b") == 0 || strcmp(arg, "--backend") == 0) {
            if (i + 1 >= argc || !parse_backend(argv[++i], &opts->backend)) {
                return -1;
            }
            opts->backend_set = 1;
            continue;
        }
        if (strcmp(arg, "--mpi") == 0) {
            opts->backend = BACKEND_MPI;
            opts->backend_set = 1;
            continue;
        }
        if (strcmp(arg, "--single") == 0) {
            opts->backend = BACKEND_SINGLE;
            opts->backend_set = 1;
            continue;
        }
        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--mode") == 0) {
            if (i + 1 >= argc || !parse_mode(argv[++i], &opts->mode)) {
                return -1;
            }
            opts->mode_set = 1;
            continue;
        }
        if (arg[0] == '-') {
            return -1;
        }
        if (!parse_int(arg, &exp)) {
            return -1;
        }
    }

    if (n_direct > 0) {
        opts->nBodies = n_direct;
    } else if (exp > 0) {
        if (exp >= (int)(sizeof(int) * 8 - 2)) {
            return -1;
        }
        opts->nBodies = 1 << (exp + 1);
    }

    if (!opts->backend_set) {
        opts->backend = default_backend();
    }
    if (!opts->mode_set) {
        opts->mode = default_mode();
    }

    return 0;
}

int mode_uses_gpu(compute_mode_t mode) {
    return mode == MODE_GPU || mode == MODE_HYBRID;
}
