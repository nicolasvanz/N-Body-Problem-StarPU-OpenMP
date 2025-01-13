#ifndef __FILES_H
#define __FILES_H

#include <stdio.h>

void write_values_to_file(const char *file, void *data, size_t size, int count)
{
    FILE *f = fopen(file, "wb");
    fwrite(data, size, count, f);
    fclose(f);
}

void read_values_from_file(const char *file, void *data, size_t size, int count)
{
    FILE *f = fopen(file, "rb");
    fread(data, size, count, f);
    fclose(f);
}

#endif