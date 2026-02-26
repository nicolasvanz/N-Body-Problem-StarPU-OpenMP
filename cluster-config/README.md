# AMI Bake Workflow

This folder contains AMI-baking scripts for CPU and CUDA nodes.

## Files

- `bake-ami.sh`: launches a temporary EC2 builder, runs setup, creates an AMI, and terminates the builder.
- `ami-bake.env.example`: template for bake parameters.
- `setup-cpu.sh`: provisions CPU image dependencies and builds StarPU/FxT.
- `setup-cuda.sh`: provisions CUDA image dependencies and builds StarPU/FxT with CUDA.

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

Keep builder instance after failure or success:

```bash
./cluster-config/bake-ami.sh --type cpu --keep-builder
```

## Common Overrides

```bash
./cluster-config/bake-ami.sh --type cpu --region us-east-1
./cluster-config/bake-ami.sh --type cpu --base-ami ami-xxxx --instance-type m5.2xlarge
./cluster-config/bake-ami.sh --type cuda --sg-ids sg-aaa,sg-bbb
```