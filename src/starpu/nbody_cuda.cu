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

#include "../include/body.h"

static __global__ void
bodyForce(const Pos *global_pos, const Pos *local_pos, Vel *v, int nPos, int nVel) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < nVel; i += stride) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < nPos; j++) {
            float dx = global_pos[j].x - local_pos[i].x;
            float dy = global_pos[j].y - local_pos[i].y;
            float dz = global_pos[j].z - local_pos[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
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

static __global__ void integratePositions(Pos *p, Vel *v, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) { // integrate position
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

extern "C" void bodyForce_cuda(void *buffers[], void *_args) {
    /* length of the vector */
    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[2]);

    /* local copy of the vector pointer */
    Pos *global_pos = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Pos *local_pos = (Pos *)STARPU_VECTOR_GET_PTR(buffers[1]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[2]);

    unsigned threads_per_block = 64;
    unsigned nblocks = 60;

    bodyForce<<<nblocks,
                threads_per_block,
                0,
                starpu_cuda_get_local_stream()>>>(global_pos, local_pos, v, nPos, nVel);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void integratePositions_cuda(void *buffers[], void *_args) {
    /* length of the vector */
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    /* local copy of the vector pointer */
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned threads_per_block = 64;
    unsigned nblocks = (nVel + threads_per_block - 1) / threads_per_block;

    integratePositions<<<nblocks,
                         threads_per_block,
                         0,
                         starpu_cuda_get_local_stream()>>>(p, v, nVel);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
