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

#include <atomic>

#include "../include/body.h"

static const unsigned int CUDA_FALLBACK_THREADS_PER_BLOCK = 128U;
static std::atomic<int> clear_acc_threads(0);
static std::atomic<int> reduce_acc_threads(0);
static std::atomic<int> bodyforce_threads(0);
static std::atomic<int> bodyforce_tile_threads(0);
static std::atomic<int> integrate_threads(0);
static std::atomic<int> integrate_tiled_threads(0);

template <typename KernelFunc>
static inline unsigned int cuda_threads_per_block(KernelFunc kernel,
                                                  std::atomic<int> *cached_threads) {
    int cached = cached_threads->load(std::memory_order_relaxed);
    if (cached > 0) {
        return (unsigned int)cached;
    }

    int min_grid_size = 0;
    int suggested_threads = 0;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &suggested_threads, kernel, 0, 0);
    (void)min_grid_size;
    if (err != cudaSuccess || suggested_threads <= 0) {
        suggested_threads = (int)CUDA_FALLBACK_THREADS_PER_BLOCK;
    }

    cached_threads->store(suggested_threads, std::memory_order_relaxed);
    return (unsigned int)suggested_threads;
}

static __global__ void clearAcceleration(Vel *a, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) {
        a[i].vx = 0.0f;
        a[i].vy = 0.0f;
        a[i].vz = 0.0f;
    }
}

static __global__ void reduceAcceleration(Vel *dst, const Vel *src, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) {
        dst[i].vx += src[i].vx;
        dst[i].vy += src[i].vy;
        dst[i].vz += src[i].vz;
    }
}

static __global__ void
bodyForce(Pos *p, Vel *v, int nPos, int nVel, int offset) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < nVel; i += stride) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < nPos; j++) {
            float dx = p[j].x - p[i + offset].x;
            float dy = p[j].y - p[i + offset].y;
            float dz = p[j].z - p[i + offset].z;
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

static __global__ void bodyForceTile(
    const Pos *pI, const Pos *pJ, Vel *a, int nI, int nJ) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < nI; i += stride) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < nJ; j++) {
            float dx = pJ[j].x - pI[i].x;
            float dy = pJ[j].y - pI[i].y;
            float dz = pJ[j].z - pI[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
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

static __global__ void integratePositions(Pos *p, Vel *v, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) { // integrate position
        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

static __global__ void
integratePositionsTiled(Pos *p, Vel *v, const Vel *a, int n) {
    int initialIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = initialIndex; i < n; i += stride) {
        v[i].vx += dt * a[i].vx;
        v[i].vy += dt * a[i].vy;
        v[i].vz += dt * a[i].vz;

        p[i].x += v[i].vx * dt;
        p[i].y += v[i].vy * dt;
        p[i].z += v[i].vz * dt;
    }
}

static inline unsigned int cuda_nblocks(unsigned int n, unsigned int threads_per_block) {
    unsigned int by_size = (n + threads_per_block - 1) / threads_per_block;
    return by_size > 0 ? by_size : 1;
}

extern "C" void clearAcceleration_cuda(void *buffers[], void *_args) {
    (void)_args;

    unsigned int nAcc = STARPU_VECTOR_GET_NX(buffers[0]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]);

    unsigned int threads_per_block =
        cuda_threads_per_block(clearAcceleration, &clear_acc_threads);
    unsigned int nblocks = cuda_nblocks(nAcc, threads_per_block);

    clearAcceleration<<<nblocks,
                        threads_per_block,
                        0,
                        starpu_cuda_get_local_stream()>>>(a, (int)nAcc);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void reduceAcceleration_cuda(void *buffers[], void *_args) {
    (void)_args;

    unsigned int nAcc = STARPU_VECTOR_GET_NX(buffers[0]);
    Vel *dst = (Vel *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *src = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned int threads_per_block =
        cuda_threads_per_block(reduceAcceleration, &reduce_acc_threads);
    unsigned int nblocks = cuda_nblocks(nAcc, threads_per_block);

    reduceAcceleration<<<nblocks,
                         threads_per_block,
                         0,
                         starpu_cuda_get_local_stream()>>>(dst, src, (int)nAcc);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void bodyForce_cuda(void *buffers[], void *_args) {
    (void)_args;

    /* length of the vector */
    unsigned int nPos = STARPU_VECTOR_GET_NX(buffers[0]);
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    /* local copy of the vector pointer */
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);
    unsigned int offset = (unsigned int)STARPU_VECTOR_GET_SLICE_BASE(buffers[1]);

    unsigned int threads_per_block =
        cuda_threads_per_block(bodyForce, &bodyforce_threads);
    unsigned int nblocks = cuda_nblocks(nVel, threads_per_block);

    bodyForce<<<nblocks,
                threads_per_block,
                0,
                starpu_cuda_get_local_stream()>>>(p, v, nPos, nVel, offset);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void bodyForce_tile_cuda(void *buffers[], void *_args) {
    (void)_args;

    Pos *pI = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned int nI = STARPU_VECTOR_GET_NX(buffers[0]);
    Pos *pJ = (Pos *)STARPU_VECTOR_GET_PTR(buffers[1]);
    unsigned int nJ = STARPU_VECTOR_GET_NX(buffers[1]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[2]);

    unsigned int threads_per_block =
        cuda_threads_per_block(bodyForceTile, &bodyforce_tile_threads);
    unsigned int nblocks = cuda_nblocks(nI, threads_per_block);

    bodyForceTile<<<nblocks,
                    threads_per_block,
                    0,
                    starpu_cuda_get_local_stream()>>>(pI, pJ, a, (int)nI, (int)nJ);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void integratePositions_cuda(void *buffers[], void *_args) {
    /* length of the vector */
    unsigned int nVel = STARPU_VECTOR_GET_NX(buffers[1]);

    /* local copy of the vector pointer */
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);

    unsigned int threads_per_block =
        cuda_threads_per_block(integratePositions, &integrate_threads);
    unsigned int nblocks = cuda_nblocks(nVel, threads_per_block);

    integratePositions<<<nblocks,
                         threads_per_block,
                         0,
                         starpu_cuda_get_local_stream()>>>(p, v, nVel);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void integratePositions_tiled_cuda(void *buffers[], void *_args) {
    (void)_args;

    unsigned int n = STARPU_VECTOR_GET_NX(buffers[0]);
    Pos *p = (Pos *)STARPU_VECTOR_GET_PTR(buffers[0]);
    Vel *v = (Vel *)STARPU_VECTOR_GET_PTR(buffers[1]);
    Vel *a = (Vel *)STARPU_VECTOR_GET_PTR(buffers[2]);

    unsigned int threads_per_block =
        cuda_threads_per_block(integratePositionsTiled, &integrate_tiled_threads);
    unsigned int nblocks = cuda_nblocks(n, threads_per_block);

    integratePositionsTiled<<<nblocks,
                              threads_per_block,
                              0,
                              starpu_cuda_get_local_stream()>>>(p, v, a, (int)n);

    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
