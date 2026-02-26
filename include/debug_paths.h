#ifndef NBODY_DEBUG_PATHS_H
#define NBODY_DEBUG_PATHS_H

#include <limits.h>
#include <stddef.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

void init_debug_paths(const char *argv0);
void make_debug_path(char *out, size_t out_sz, const char *file);

#endif
