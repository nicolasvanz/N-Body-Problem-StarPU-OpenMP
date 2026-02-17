/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <stdio.h>

#include "../include/body.h"

void integratePositions_cpu(void *buffers[], void *_args) {
    (void)_args;

    /* length of the vector */
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    /* local copy of the vector pointer */
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    for (unsigned i = 0; i < nVel; i++) {
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

void clearAcceleration_cpu(void *buffers[], void *_args) {
    (void)_args;

    unsigned int nAcc = STARPU_VECTOR_GET_NX(buffers[0]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]);

    for (unsigned i = 0; i < nAcc; i++) {
        a[i].vx = 0.0f;
        a[i].vy = 0.0f;
        a[i].vz = 0.0f;
    }
}

void reduceAcceleration_cpu(void *buffers[], void *_args) {
    (void)_args;

    unsigned int nAcc = STARPU_VECTOR_GET_NX(buffers[0]);
    Vel *dst = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *src = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    for (unsigned i = 0; i < nAcc; i++) {
        dst[i].vx += src[i].vx;
        dst[i].vy += src[i].vy;
        dst[i].vz += src[i].vz;
    }
}

void bodyForce_tile_cpu(void *buffers[], void *_args) {
    (void)_args;

    Pos *pI = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned int nI = STARPU_VECTOR_GET_NX(buffers[0]);
    Pos *pJ = (Pos *)STARPU_VECTOR_GET_PTR(buffers[1]);
    unsigned int nJ = STARPU_VECTOR_GET_NX(buffers[1]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[2]);

    for (unsigned i = 0; i < nI; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (unsigned j = 0; j < nJ; j++) {
            float dx = pJ[j].x - pI[i].x;
            float dy = pJ[j].y - pI[i].y;
            float dz = pJ[j].z - pI[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = my_rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        a[i].vx += Fx;
        a[i].vy += Fy;
        a[i].vz += Fz;
    }
}

void integratePositions_tiled_cpu(void *buffers[], void *_args) {
    (void)_args;

    unsigned int n = STARPU_VECTOR_GET_NX(buffers[0]);
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[2]);

    for (unsigned i = 0; i < n; i++) {
        v[i].vx += dt * a[i].vx;
        v[i].vy += dt * a[i].vy;
        v[i].vz += dt * a[i].vz;

        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

void bodyForce_cpu(void *buffers[], void *_args) {
    (void)_args;

    /* length of the vector */
    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    /* local copy of the vector pointer */
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    size_t offset = STARPU_VECTOR_GET_SLICE_BASE(buffers[1]);

    for (unsigned i = 0; i < nVel; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (unsigned j = 0; j < nPos; j++) {
            float dx = p[j].x - p[i + offset].x;
            float dy = p[j].y - p[i + offset].y;
            float dz = p[j].z - p[i + offset].z;
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
