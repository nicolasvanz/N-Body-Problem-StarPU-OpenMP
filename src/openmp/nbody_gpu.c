#include <omp.h>
#include <math.h>
#include "../include/body.h"

void bodyForce_gpu(Pos *p, Vel *v, int n)
{
#pragma omp target teams distribute parallel for map(to : p[0 : n]) map(tofrom : v[0 : n]) thread_limit(64)
  for (int i = 0; i < n; i++)
  {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (unsigned j = 0; j < n; j++)
    {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = my_rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    v[i].vx += dt * Fx;
    v[i].vy += dt * Fy;
    v[i].vz += dt * Fz;
  }
}

void integratePositions_gpu(Pos *p, Vel *v, int n)
{
#pragma omp target teams distribute parallel for map(tofrom : p[0 : n]) map(to : v[0 : n]) thread_limit(64)
  for (int i = 0; i < n; i++)
  {
    p[i].x += v[i].vx * dt;
    p[i].y += v[i].vy * dt;
    p[i].z += v[i].vz * dt;
  }
}