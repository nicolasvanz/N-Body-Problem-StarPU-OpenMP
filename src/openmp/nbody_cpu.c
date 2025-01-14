#include <omp.h>
#include <math.h>
#include <stdio.h>
#include "../include/body.h"

void bodyForce_cpu(Pos *p, Vel *v, int n)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		float Fx = 0.0f;
		float Fy = 0.0f;
		float Fz = 0.0f;

		for (int j = 0; j < n; j++)
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

void integratePositions_cpu(Pos *p, Vel *v, int n)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		p[i].x += v[i].vx * dt;
		p[i].y += v[i].vy * dt;
		p[i].z += v[i].vz * dt;
	}
}