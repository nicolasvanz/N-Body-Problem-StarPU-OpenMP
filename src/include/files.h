#ifndef __FILES_H
#define __FILES_H

#include <stdio.h>

void write_values_to_file(const char *file, void *data, size_t size, int count)
{
	FILE *f = fopen(file, "wb");
	if (f == NULL)
	{
		printf("error oppenning %s\n", file);
		exit(1);
	}
	if (fwrite(data, size, count, f) != (size_t)count)
	{
		printf("error writting all elements to %s\n", file);
	};

	fclose(f);
}

void read_values_from_file(const char *file, void *data, size_t size, int count)
{
	FILE *f = fopen(file, "rb");
	if (f == NULL)
	{
		printf("error oppenning %s\n", file);
		exit(1);
	}
	if (fread(data, size, count, f) != (size_t)count)
	{
		printf("error reading all elements from file %s\n", file);
	}
	fclose(f);
}

#endif