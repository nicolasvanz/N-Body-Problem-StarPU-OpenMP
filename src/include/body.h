#ifndef __BODY_H
#define __BODY_H

#include <math.h>

static inline float my_rsqrtf(float x) {
    return 1.0f / (float)sqrt((double)x);
}

typedef struct { float x, y, z; } Pos;
typedef struct { float vx, vy, vz; } Vel;

#define SOFTENING 1e-9f
#define dt 0.01f

#endif