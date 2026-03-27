# Cluster Config Workflow

This folder contains:
- AMI-baking scripts for CPU/CUDA/master-slave images.
- A multi-node helper to sync/build/run across EC2 nodes.

## Files

- `bake-ami.sh`: launches a temporary EC2 builder, runs setup, creates an AMI, and terminates the builder.
- `ami-bake.env.example`: template for bake parameters.
- `setup-cpu.sh`: provisions CPU image dependencies and builds StarPU/FxT.
- `setup-cuda.sh`: provisions CUDA image dependencies and builds StarPU/FxT with CUDA.
- `setup-ms.sh`: provisions CUDA image dependencies and builds StarPU from a git repo (or fallback local source) with FxT + MPI server-client.
- `cluster-mpi.sh`: parses cluster IP output, syncs repo to all nodes, builds on all nodes, and runs MPI from one node with generated hostfile.
- `cluster-mpi.env.example`: template for `cluster-mpi.sh` settings.
- `run-experiments.sh`: automates benchmark matrices for StarPU MPI vs master-slave and stores CSV summaries.
- `experiments.env.example`: template for `run-experiments.sh` settings.

## What The Scripts Install

`setup-cpu.sh`:
- StarPU 1.4.7 + FxT 0.3.15 (MPI enabled, CUDA disabled).
- OpenMPI toolchain for StarPU MPI builds.
- clang 18 + OpenMP runtime (`libomp18`) on Amazon Linux 2023.
- Verifies a minimal clang OpenMP compile probe.

`setup-cuda.sh`:
- StarPU 1.4.7 + FxT 0.3.15 (MPI + CUDA enabled).
- CUDA toolkit detection and environment wiring.
- OpenMPI toolchain for StarPU MPI builds.

`setup-ms.sh`:
- Clones and builds your custom StarPU checkout from:
  - `STARPU_REPO_URL` + `STARPU_REPO_REF` (default path), or
  - local source in `$STARPU_SRC_DIR` if `STARPU_REPO_URL` is empty.
- Enables `--enable-mpi-server-client --enable-cuda --enable-fxt`.
- Configures `--enable-maxmpidev` via `STARPU_MAXMPIDEV` (default `16`).
- Installs under `/usr/local` and enables FxT trace env exports.

## Prerequisites

- AWS CLI configured (`aws configure` or profile/env vars).
- Network path to builder instance for SSH.
- EC2 key pair + matching local private key file.

## Configure Bake Parameters

```bash
cp cluster-config/ami-bake.env.example cluster-config/ami-bake.env
```

Edit `cluster-config/ami-bake.env`:
- `REGION`
- `BASE_AMI_CPU`
- `BASE_AMI_CUDA`
- `BASE_AMI_MASTER_SLAVE` (optional; defaults to `BASE_AMI_CUDA`)
- `STARPU_REPO_URL` (for `--type master-slave`, defaults to `https://github.com/nicolasvanz/starpu-ms-gpu.git`)
- `STARPU_REPO_REF` (for `--type master-slave`, defaults to `enhancement/ms-cuda-support`)
- `STARPU_REPO_TOKEN` (required if repo is private and cloned over HTTPS)
- `STARPU_MAXMPIDEV` (for `--type master-slave`, defaults to `16`)
- `REMOTE_STARPU_SRC` (builder checkout path, defaults to `$HOME/code/starpu`)
- `SUBNET_ID`
- `SECURITY_GROUP_IDS`
- `KEY_NAME`
- `SSH_KEY`

## Bake Commands

CPU AMI:

```bash
./cluster-config/bake-ami.sh --type cpu
```

CUDA AMI:

```bash
./cluster-config/bake-ami.sh --type cuda
```

Custom master-slave image (clones your StarPU repo on builder):

```bash
./cluster-config/bake-ami.sh --type master-slave
```

Note:
- `--type master-slave` clones `STARPU_REPO_URL` at `STARPU_REPO_REF` on the builder by default.
- For private repos, set `STARPU_REPO_TOKEN` in `ami-bake.env`.
- If `STARPU_REPO_URL` is set to empty, the script falls back to uploading `LOCAL_STARPU_SRC`.

Keep builder instance after failure or success:

```bash
./cluster-config/bake-ami.sh --type cpu --keep-builder
```

## Common Overrides

```bash
./cluster-config/bake-ami.sh --type cpu --region us-east-1
./cluster-config/bake-ami.sh --type cpu --base-ami ami-xxxx --instance-type m5.2xlarge
./cluster-config/bake-ami.sh --type cuda --sg-ids sg-aaa,sg-bbb
./cluster-config/bake-ami.sh --type master-slave --base-ami ami-zzzz --instance-type g6.xlarge
```

## Multi-Node Test Workflow

1) Create config:

```bash
cp cluster-config/cluster-mpi.env.example cluster-config/cluster-mpi.env
# edit SSH_KEY and other values if needed
```

2) Save IPs from your infra tool output (raw format supported):

```bash
./cluster-config/cluster-mpi.sh set-ips "['123.123.123.123','111.111.111.111']"
./cluster-config/cluster-mpi.sh set-ips "123.123.123.123,111.111.111.111"
```

If you want MPI to use private IPs, pass them as second argument:

```bash
./cluster-config/cluster-mpi.sh set-ips \
  "123.123.123.123,111.111.111.111" \
  "['10.10.0.10','10.10.0.11']"
```

If you pass only public IPs, `cluster-mpi.sh` can auto-resolve private MPI IPs over SSH
(`AUTO_RESOLVE_MPI_PRIVATE_IPS=1`).
Use private MPI IPs whenever possible. Public IPs are fine for control/SSH, while MPI traffic should usually use private IPs.

3) Sync local repo to all nodes:

```bash
./cluster-config/cluster-mpi.sh sync
```

4) Build on all nodes:

```bash
./cluster-config/cluster-mpi.sh build starpu
# or:
# ./cluster-config/cluster-mpi.sh build openmp

# Override build args for this invocation:
./cluster-config/cluster-mpi.sh build starpu "USE_MPI=1 USE_CUDA=0 DEBUG=1"

# Recommended when switching flags (e.g. USE_CUDA 1 -> 0):
./cluster-config/cluster-mpi.sh build starpu --clean "USE_MPI=1 USE_CUDA=0 DEBUG=1"
```

`cluster-mpi.sh` also auto-cleans when build args change between invocations on a node.

5) Generate hostfile:

```bash
./cluster-config/cluster-mpi.sh hostfile
```

6) Run custom commands on cluster nodes:

```bash
# Run on launch host only:
./cluster-config/cluster-mpi.sh exec-launch "make compare"

# Run on all nodes:
./cluster-config/cluster-mpi.sh exec-all "rm -rf N-Body-Problem-StarPU-OpenMP/build-*"
```

7) Run MPI from one launch node:

```bash
./cluster-config/cluster-mpi.sh run starpu

# Override runtime args for this invocation:
./cluster-config/cluster-mpi.sh run starpu "--mpi --cpu --exp 18"

# Add mpirun args (example: process binding):
./cluster-config/cluster-mpi.sh run starpu \
  --mpirun-args "--bind-to board" \
  --run-args "--mpi --cpu --exp 18"
```

`run` automatically prepares SSH from launch host to worker hosts for MPI daemon startup.
If you need to disable this behavior, pass `--no-prepare-ssh`.

8) One-shot flow:

```bash
./cluster-config/cluster-mpi.sh all starpu

# Global mpirun args applied to this invocation:
./cluster-config/cluster-mpi.sh --mpirun-args "--bind-to board" all starpu
```

9) Fetch and condense StarPU FxT traces on your local machine:

```bash
# 9.1 Fetch from each node (default remote path auto-detected).
# setup-cpu.sh/setup-cuda.sh configure STARPU_FXT_PREFIX as: ~/starpu_traces/fxt
./cluster-config/cluster-mpi.sh fetch-traces

# optional custom session folder name:
./cluster-config/cluster-mpi.sh fetch-traces 04032026-203208

# optional explicit remote trace directory override:
./cluster-config/cluster-mpi.sh fetch-traces --remote-dir "/home/ec2-user/starpu_traces/fxt"

# 9.2 Condense with starpu_fxt_tool and create <session>/condensed marker file:
./cluster-config/cluster-mpi.sh condense-traces
# or for a specific fetched session:
./cluster-config/cluster-mpi.sh condense-traces 04032026-203208
```

Local layout (git-ignored):
- `traces/<session>/prof_file_<ip>-...` / `proc_file_<ip>-...`: fetched files flattened in one folder.
- `traces/<session>/condensed`: file created after `condense-traces`.

Open traces in starvz:

```bash
# latest session under traces/
./cluster-config/cluster-mpi.sh starvz

# specific session name or path
./cluster-config/cluster-mpi.sh starvz 04032026-203208
./cluster-config/cluster-mpi.sh starvz traces/04032026-203208

# open newest generated PDF (latest session by default)
./cluster-config/cluster-mpi.sh open-starvz-pdf
./cluster-config/cluster-mpi.sh open-starvz-pdf 04032026-203208
```

## Experiment Automation

Use this runner to execute reproducible MPI vs master-slave experiments and
store results under `results/experiments/<session>`:

```bash
cp cluster-config/experiments.env.example cluster-config/experiments.env
# edit if needed

./cluster-config/run-experiments.sh --mode both
```

Outputs:
- `results.csv`: one row per repetition (`warmup` + `timed`).
- `summary.csv`: aggregate stats (mean/min/max/stddev) over successful timed rows.

Default power-equivalent pairs are:
- MPI 2 <-> MS 3
- MPI 4 <-> MS 5
- MPI 8 <-> MS 9

## Rank 0 / Master Notes

- `PREFERRED_MASTER_IP=10.0.0.10` makes that host first in generated hostfile when present.
- `CONTROL_MASTER_IP` forces where `mpirun` is launched from.
- Rank 0 placement usually has little performance impact for this workload. It mainly affects where orchestration/logging happens and can matter slightly for startup/control traffic.
