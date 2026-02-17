# N-Body Problem (OpenMP + StarPU)

This project provides two implementations of the same N-body simulation:

- `openmp`: OpenMP implementation with optional target offload (GPU).
- `starpu`: StarPU task-based implementation with optional CUDA support.

Both backends share the same runtime CLI options (`--n`, `--exp`, `--mode`, `--backend`, etc.).

## Repository Layout

- `Makefile`: top-level entrypoint for building/running either backend.
- `src/openmp`: OpenMP implementation and backend-specific Makefile.
- `src/starpu`: StarPU implementation and backend-specific Makefile.
- `src/common/options.c`: shared command-line parsing.
- `src/debug`: debug input/reference/output binary files.
- `src/compare.py`: tolerance-based comparison for debug outputs.

## Requirements

For OpenMP backend:

- C compiler with OpenMP support (`clang` recommended).
- For GPU offload (`OFFLOAD=1`): clang OpenMP offload toolchain and CUDA offload plugin.
- Optional MPI runtime/compiler (`mpicc`, `mpirun`) when `USE_MPI=1`.

For StarPU backend:

- StarPU development packages (`pkg-config` must find `starpu-<version>`).
- Optional StarPU MPI package when `USE_MPI=1` (`starpumpi-<version>`).
- Optional CUDA toolkit (`nvcc`) for GPU mode.
- Optional MPI runtime/compiler when `USE_MPI=1`.

## Top-Level Build and Run

Use the root `Makefile` for most workflows.

- Build selected backend:
```bash
make IMPL=openmp all
make IMPL=starpu all
```

- Run selected backend through the top-level wrapper:
```bash
make run IMPL=openmp ARGS="--single --cpu --n 131072"
make run IMPL=starpu ARGS="--single --gpu --exp 18"
```

- Other common targets:
```bash
make clean
make IMPL=openmp diff-txt
make IMPL=starpu diff-txt
make compare
```

`make run` executes `./nbody` directly. For MPI execution, run the binary with `mpirun`/`mpiexec` (examples below).

## Compilation Options

## Common (root `Makefile`)

- `IMPL=openmp|starpu` (default: `openmp`): select backend.
- `PYTHON=python3`: Python interpreter for `make compare`.
- `RTOL=1e-3`, `ATOL=1e-5`, `MAX_REPORT=10`: comparison tolerances/report size.

## OpenMP (`src/openmp/Makefile`)

- `USE_MPI=1` (default): compile MPI-enabled binary (`-DUSE_MPI=1`).
- `USE_MPI=0`: compile without MPI.
- `MPI_CC=clang` (default): compiler used by OpenMPI wrapper (`OMPI_CC=$(MPI_CC)`).
- `MPICC="OMPI_CC=$(MPI_CC) mpicc"`: MPI compile command used when `USE_MPI=1`.
- `CC=clang` (default): compiler used when `USE_MPI=0`.
- `OFFLOAD=1` (default): enable OpenMP GPU offload (`-DOPENMP_OFFLOAD=1`).
- `OFFLOAD=0`: CPU-only OpenMP build (`-DOPENMP_OFFLOAD=0`).
- `OMP_TARGET=nvptx64` (default): OpenMP offload target triple.
- `GPU_ARCH=sm_80` (default): GPU architecture passed to target flags.
- `DEBUG=1`: enable debug mode (`-DDEBUG`) and deterministic debug I/O.

Important:

- `OFFLOAD=1` requires a clang-based compiler.
- When `OFFLOAD=1`, default runtime mode becomes GPU.
- When `OFFLOAD=0`, default runtime mode becomes CPU.

## StarPU (`src/starpu/starpu.mk`)

- `STARPU_VERSION=1.4` (default): used in `pkg-config` names.
- `USE_MPI=1` (default): build with StarPU MPI support.
- `USE_MPI=0`: build without MPI.
- `USE_CUDA=1|0`: enable/disable CUDA codelet build.
  - If StarPU config reports CUDA enabled, `USE_CUDA` is auto-enabled unless overridden.
- `NVCC=nvcc` (default): CUDA compiler.
- `CUDA_PATH`, `CUDA_LIBDIR`: CUDA install/lib locations.
- `DEBUG=1`: enable debug mode (`-DDEBUG`).

Important:

- GPU/hybrid runtime modes require a build with CUDA support (`USE_CUDA=1` and StarPU with CUDA).
- If CUDA is unavailable, use CPU mode.

## Runtime Arguments (`./nbody`)

Shared by both backends:

- `-n, --n <count>`: number of bodies (positive integer).
- `--exp <e>`: legacy exponent, computes `nBodies = 1 << (e + 1)` (same as `2 << e`).
- `-b, --backend <single|mpi>`: execution backend.
- `--single`: shorthand for `--backend single`.
- `--mpi`: shorthand for `--backend mpi`.
- `-m, --mode <cpu|gpu|hybrid>`: compute mode.
- `--cpu`: shorthand for `--mode cpu`.
- `--gpu`: shorthand for `--mode gpu`.
- `--hybrid`: shorthand for `--mode hybrid`.
- `-a, --algo <classic|tiled>`: simulation algorithm (default: `classic`).
- `--classic`: shorthand for `--algo classic`.
- `--tiled`: shorthand for `--algo tiled`.
- `-h, --help`: print usage.
- Positional integer argument (legacy): treated as exponent (`--exp` behavior).

Default behavior:

- `nBodies` default is `2 << 12` (8192).
- `mode` default depends on build:
  - OpenMP: GPU if `OFFLOAD=1`, else CPU.
  - StarPU: GPU if built with CUDA, else CPU.
- `backend` defaults to MPI only when compiled with MPI support and MPI environment variables are detected; otherwise single-process.
- `algorithm` defaults to `classic`.

Algorithm notes:

- `classic` is the original two-phase loop (all velocity updates, then position updates).
- `tiled` is currently implemented only for the StarPU backend in CPU mode (`--cpu`) and works with both `--single` and `--mpi`.

## Compile Examples

## OpenMP CPU-only, single-process capable

```bash
make openmp-clean
make openmp USE_MPI=0 OFFLOAD=0
```

## OpenMP with MPI + GPU offload

```bash
make openmp-clean
make openmp USE_MPI=1 MPI_CC=clang OFFLOAD=1 OMP_TARGET=nvptx64 GPU_ARCH=sm_80
```

## StarPU CPU-only

```bash
make starpu-clean
make starpu USE_MPI=0 USE_CUDA=0
```

## StarPU with MPI + CUDA

```bash
make starpu-clean
make starpu USE_MPI=1 USE_CUDA=1
```

## Run Examples

## Single-process runs

```bash
make run-openmp ARGS="--single --cpu --n 131072"
./src/openmp/nbody --single --gpu --exp 18

make run-starpu ARGS="--single --cpu --n 131072"
./src/starpu/nbody --single --gpu --exp 18
```

## MPI runs

```bash
mpirun -np 2 ./src/openmp/nbody --mpi --cpu --exp 18
mpirun -np 2 ./src/openmp/nbody --mpi --gpu --n 262144

mpirun -np 2 ./src/starpu/nbody --mpi --cpu --exp 18
mpirun -np 2 ./src/starpu/nbody --mpi --gpu --n 262144
```

## Debug Output and Result Comparison

`DEBUG=1` changes behavior to deterministic debug data:

- Input loaded from:
  - `src/debug/initialized_pos_12`
  - `src/debug/initialized_vel_12`
- Number of bodies forced to `2 << 12` (8192), regardless of `--n`/`--exp`.
- Output written to:
  - `src/debug/computed_pos_12`
  - `src/debug/computed_vel_12`

Reference files:

- `src/debug/solution_pos_12`
- `src/debug/solution_vel_12`

## Workflow to compare debug results

1. Build one backend in debug mode.
```bash
make openmp-clean
make openmp DEBUG=1 USE_MPI=0 OFFLOAD=0
# or:
# make starpu DEBUG=1 USE_MPI=0 USE_CUDA=0
```

2. Run it.
```bash
./src/openmp/nbody --single --cpu
# or:
# ./src/starpu/nbody --single --cpu
```

3. Run tolerance-based comparison.
```bash
make compare
```

Optional: override tolerance/reporting.
```bash
make compare RTOL=1e-4 ATOL=1e-6 MAX_REPORT=20
```

4. For line-by-line text diff output:
```bash
make openmp diff-txt
# or
make starpu diff-txt
```

`make compare` returns non-zero if differences exceed tolerance, or if NaN/Inf mismatches are found.

## Notes

- OpenMP backend prints elapsed time in seconds.
- StarPU backend prints elapsed time in microseconds.
