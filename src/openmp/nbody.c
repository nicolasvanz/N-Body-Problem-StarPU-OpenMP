#include <omp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/body.h"
#include "../include/debug_paths.h"
#include "../include/files.h"
#include "../include/options.h"

#ifndef USE_MPI
#define USE_MPI 0
#endif

#ifndef OPENMP_OFFLOAD
#define OPENMP_OFFLOAD 0
#endif

#if USE_MPI
#include <mpi.h>
#endif

extern void bodyForce_cpu(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n);
extern void bodyForce_gpu(
    Pos *global_pos, Vel *local_vel, int local_start, int local_n, int n);
extern void integratePositions_cpu(Pos *local_pos, Vel *local_vel, int local_n);
extern void integratePositions_gpu(Pos *local_pos, Vel *local_vel, int local_n);

typedef void (*bodyforce_fn)(Pos *, Vel *, int, int, int);
typedef void (*integrate_fn)(Pos *, Vel *, int);

#if OPENMP_OFFLOAD
static void require_offload(int use_mpi, int rank) {
    int offload_ok = 0;
#pragma omp target map(tofrom : offload_ok)
    { offload_ok = !omp_is_initial_device(); }
    if (!offload_ok) {
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: OpenMP target offload not active. "
                    "Ensure libomptarget CUDA plugin is available and rebuild "
                    "with the correct GPU_ARCH (e.g. make GPU_ARCH=sm_75).\n");
        }
#if USE_MPI
        if (use_mpi) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#endif
        exit(1);
    }
}
#endif

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
        pos[i].x = ((float)rand() / (float)RAND_MAX) * 100.0f;
        pos[i].y = ((float)rand() / (float)RAND_MAX) * 100.0f;
        pos[i].z = ((float)rand() / (float)RAND_MAX) * 100.0f;
        vel[i].vx = ((float)rand() / (float)RAND_MAX) * 10.0f;
        vel[i].vy = ((float)rand() / (float)RAND_MAX) * 10.0f;
        vel[i].vz = ((float)rand() / (float)RAND_MAX) * 10.0f;
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

static int run_single(const options_t *opts, bodyforce_fn bodyforce,
                      integrate_fn integrate) {
    int nBodies = opts->nBodies;

#ifdef DEBUG
    nBodies = 2 << 12;
    printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
#endif

    Pos *pos = (Pos *)(malloc(sizeof(Pos) * nBodies));
    Vel *vel = (Vel *)(malloc(sizeof(Vel) * nBodies));
    if (!pos || !vel) {
        fprintf(stderr, "ERROR: failed to allocate bodies\n");
        free(pos);
        free(vel);
        return 1;
    }

    init_bodies(pos, vel, nBodies);

    const int nIters = 10;
    double start = omp_get_wtime();

#if OPENMP_OFFLOAD
    if (mode_uses_gpu(opts->mode)) {
        require_offload(0, 0);
    }
#endif

    for (int iter = 0; iter < nIters; iter++) {
#pragma omp task
        bodyforce(pos, vel, 0, nBodies, nBodies);
#pragma omp task
        integrate(pos, vel, nBodies);
    }
    printf("%lf\n", omp_get_wtime() - start);

    write_debug_outputs(pos, vel, nBodies);

    free(pos);
    free(vel);
    return 0;
}

#if USE_MPI
static int run_mpi(const options_t *opts, bodyforce_fn bodyforce,
                   integrate_fn integrate, int argc, char **argv) {
    int nBodies = opts->nBodies;
    int rank, size;
    double start = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    nBodies = 2 << 12;
    if (rank == 0) {
        printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
    }
#endif

#if OPENMP_OFFLOAD
    if (mode_uses_gpu(opts->mode)) {
        require_offload(1, rank);
    }
#endif

    int base = nBodies / size;
    int rem = nBodies % size;
    int *sendCounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int count = (i < rem) ? base + 1 : base;
        sendCounts[i] = count;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    int local_n = sendCounts[rank];
    int local_start = displs[rank];

    Pos *global_pos = NULL;
    Vel *global_vel = NULL;

    if (rank == 0) {
        global_pos = (Pos *)(malloc(sizeof(Pos) * nBodies));
        global_vel = (Vel *)(malloc(sizeof(Vel) * nBodies));
#ifdef DEBUG
        char initialized_pos[PATH_MAX];
        char initialized_vel[PATH_MAX];
        make_debug_path(initialized_pos, sizeof(initialized_pos), "initialized_pos_12");
        make_debug_path(initialized_vel, sizeof(initialized_vel), "initialized_vel_12");
        read_values_from_file(initialized_pos, global_pos, sizeof(Pos), nBodies);
        read_values_from_file(initialized_vel, global_vel, sizeof(Vel), nBodies);
#else
        for (int i = 0; i < nBodies; i++) {
            global_pos[i].x = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_pos[i].y = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_pos[i].z = ((float)rand() / (float)RAND_MAX) * 100.0f;
            global_vel[i].vx = ((float)rand() / (float)RAND_MAX) * 10.0f;
            global_vel[i].vy = ((float)rand() / (float)RAND_MAX) * 10.0f;
            global_vel[i].vz = ((float)rand() / (float)RAND_MAX) * 10.0f;
        }
#endif
    }

    Pos *local_pos = malloc(sizeof(Pos) * local_n);
    Vel *local_vel = malloc(sizeof(Vel) * local_n);
    if (!local_pos || !local_vel) {
        fprintf(stderr, "ERROR: failed to allocate local buffers\n");
        free(local_pos);
        free(local_vel);
        free(global_pos);
        if (rank == 0) {
            free(global_vel);
        }
        free(sendCounts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Datatype MPI_Pos, MPI_Vel;
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Pos); // Assuming Pos is 3 floats
    MPI_Type_commit(&MPI_Pos);

    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_Vel); // Assuming Vel is 3 floats
    MPI_Type_commit(&MPI_Vel);

    if (rank != 0)
        global_pos = malloc(sizeof(Pos) * nBodies);

    if (rank == 0) {
        start = omp_get_wtime();
    }
    
    MPI_Scatterv(global_vel,
                 sendCounts,
                 displs,
                 MPI_Vel,
                 local_vel,
                 local_n,
                 MPI_Vel,
                 0,
                 MPI_COMM_WORLD);
    MPI_Bcast(global_pos, nBodies, MPI_Pos, 0, MPI_COMM_WORLD);
    memcpy(local_pos, global_pos + local_start, sizeof(Pos) * local_n);

    const int nIters = 10;
    for (int iter = 0; iter < nIters; iter++) {
        #pragma omp task
        bodyforce(global_pos, local_vel, local_start, local_n, nBodies);
        #pragma omp task
        integrate(local_pos, local_vel, local_n);
        MPI_Allgatherv(local_pos,
                       local_n,
                       MPI_Pos,
                       global_pos,
                       sendCounts,
                       displs,
                       MPI_Pos,
                       MPI_COMM_WORLD);
    }
    MPI_Gatherv(local_vel,
                local_n,
                MPI_Vel,
                (rank == 0) ? global_vel : NULL,
                sendCounts,
                displs,
                MPI_Vel,
                0,
                MPI_COMM_WORLD);

    if (rank == 0)
        printf("%lf\n", omp_get_wtime() - start); // seconds

    if (rank == 0) {
        write_debug_outputs(global_pos, global_vel, nBodies);
    }

    free(local_pos);
    free(local_vel);
    free(global_pos);
    if (rank == 0)
        free(global_vel);
    free(sendCounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
#endif

int main(int argc, char **argv) {
    init_debug_paths(argv[0]);

    options_t opts = {
        .nBodies = 2 << 12,
        .nPartitions = 0,
        .mode = MODE_CPU,
        .backend = BACKEND_SINGLE,
        .show_help = 0,
        .backend_set = 0,
        .mode_set = 0,
        .partitions_set = 0,
    };

    if (parse_options(argc, argv, &opts) != 0 || opts.show_help) {
        print_usage(argv[0]);
        return opts.show_help ? 0 : 1;
    }

    if (mode_uses_gpu(opts.mode) && !OPENMP_OFFLOAD) {
        fprintf(stderr, "ERROR: GPU/Hybrid mode requires OPENMP offload build.\n");
        return 1;
    }

    bodyforce_fn bodyforce = bodyForce_cpu;
    integrate_fn integrate = integratePositions_cpu;
    if (opts.mode == MODE_GPU) {
        bodyforce = bodyForce_gpu;
        integrate = integratePositions_gpu;
    } else if (opts.mode == MODE_HYBRID) {
        bodyforce = bodyForce_gpu;
        integrate = integratePositions_cpu;
    }

    if (opts.backend == BACKEND_MPI) {
#if USE_MPI
        return run_mpi(&opts, bodyforce, integrate, argc, argv);
#else
        fprintf(stderr, "ERROR: MPI backend requested but binary built without MPI.\n");
        return 1;
#endif
    }

    return run_single(&opts, bodyforce, integrate);
}
