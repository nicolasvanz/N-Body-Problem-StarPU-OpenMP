# Compiler and flags
CC = clang
# CFLAGS = -fopenmp -foffload=nvptx-none -O3 -Wall -std=c99 -lm
# CFLAGS = -fopenmp -fopenmp-targets=nvptx64 -O3 -Wall -std=c99
CFLAGS = -fopenmp -O3 -Wall -std=c99

# Source files and target
SRCS = nbody.c nbody_cpu.c nbody_gpu.c 
OBJS = $(SRCS:.c=.o)
TARGET = nbody

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile each .c file into .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)
