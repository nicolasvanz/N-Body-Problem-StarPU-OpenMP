#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/body.h"
#include "../include/files.h"

// #define DEBUG
#define BODYFORCE_USE_CPU 0
#define INTEGRATEPOSITIONS_USE_CPU 1

extern void bodyForce_cpu(Pos *p, Vel *v, int n);
extern void bodyForce_gpu(Pos *p, Vel *v, int n);
extern void integratePositions_cpu(Pos *p, Vel *v, int n);
extern void integratePositions_gpu(Pos *p, Vel *v, int n);

int main(const int argc, const char **argv) {
  int nBodies = 2 << 12;

#ifndef DEBUG
  if (argc > 1) nBodies = 2 << (atoi(argv[1]) - 1);
#else
  (void)argc;
  (void)argv;
  printf("WARNING: Running on debug mode. Fixing nbodies to 2 << 12\n");
#endif

  Pos *pos = (Pos *)(malloc(sizeof(Pos) * nBodies));
  Vel *vel = (Vel *)(malloc(sizeof(Vel) * nBodies));

#ifdef DEBUG
  const char *initialized_pos = "../debug/initialized_pos_12";
  const char *initialized_vel = "../debug/initialized_vel_12";
  const char *computed_pos = "../debug/computed_pos_12";
  const char *computed_vel = "../debug/computed_vel_12";
#endif

#ifdef DEBUG
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

  const int nIters = 10;

  double start = omp_get_wtime();
  for (int iter = 0; iter < nIters; iter++) {
#if (BODYFORCE_USE_CPU == 1)
#pragma omp task
    bodyForce_cpu(pos, vel, nBodies);
#else
#pragma omp task
    bodyForce_gpu(pos, vel, nBodies);
#endif

#if (INTEGRATEPOSITIONS_USE_CPU == 1)
#pragma omp task
    integratePositions_cpu(pos, vel, nBodies);
#else
#pragma omp task
    integratePositions_gpu(pos, vel, nBodies);
#endif
  }
  printf("%lf\n", omp_get_wtime() - start);  // seconds

#ifdef DEBUG
  write_values_to_file(computed_pos, pos, sizeof(Pos), nBodies);
  write_values_to_file(computed_vel, vel, sizeof(Vel), nBodies);
#endif

  free(vel);
  free(pos);
}