# Compiler and flags
CC = clang
# CFLAGS = -fopenmp -foffload=nvptx-none -O3 -Wall -std=c99 -lm
CFLAGS = -fopenmp -fopenmp-targets=nvptx64 -O3 -Wall -std=c99

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

compare:
	@DIFF_OUTPUT=$$( \
		diff ../debug/computed_pos_12 ../debug/solution_pos_12; \
		diff ../debug/computed_vel_12 ../debug/solution_vel_12); \
	if [ -z "$$DIFF_OUTPUT" ]; then \
		echo "All files match. No differences found."; \
	else \
		echo "The files differ. Note: cuda and cpu funcitons may show different floating precision"; \
		echo "$$DIFF_OUTPUT"; \
	fi

compile:
	gcc -O2 -o bin2txt ../bin2txt.c

run-bin2txt: compile
	./bin2txt

compare-txt: run-bin2txt
	sdiff ../debug/computed_pos_12.txt ../debug/solution_pos_12.txt || true; \
	sdiff ../debug/computed_vel_12.txt ../debug/solution_vel_12.txt || true;
