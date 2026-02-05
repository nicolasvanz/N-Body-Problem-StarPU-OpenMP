#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "../include/debug_paths.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static char debug_dir[PATH_MAX];

void init_debug_paths(const char *argv0) {
    if (debug_dir[0] != '\0') {
        return;
    }

    const char *env = getenv("NBODY_DEBUG_DIR");
    if (env != NULL && env[0] != '\0') {
        snprintf(debug_dir, sizeof(debug_dir), "%s", env);
        return;
    }

    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len >= 0) {
        exe_path[len] = '\0';
    } else {
        if (argv0 == NULL || argv0[0] == '\0') {
            snprintf(exe_path, sizeof(exe_path), ".");
        } else {
            snprintf(exe_path, sizeof(exe_path), "%s", argv0);
        }
    }

    char *slash = strrchr(exe_path, '/');
    if (slash != NULL) {
        *slash = '\0';
    } else {
        snprintf(exe_path, sizeof(exe_path), ".");
    }

    snprintf(debug_dir, sizeof(debug_dir), "%s/../debug", exe_path);
}

void make_debug_path(char *out, size_t out_sz, const char *file) {
    snprintf(out, out_sz, "%s/%s", debug_dir, file);
}
