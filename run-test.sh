#!/usr/bin/env bash
set -euo pipefail

run() {
  dir_starpu="${dir_starpu:-$HOME/N-Body-Problem-StarPU-OpenMP/src/starpu}"
  hostfile="${hostfile:-$HOME/N-Body-Problem-StarPU-OpenMP/hostfile}"

  (cd "$dir_starpu" && make clean && make)

  for ip in $(awk '{print $1}' "$hostfile"); do
    echo "Copying to $ip..."
    scp "$dir_starpu/nbody" "ec2-user@$ip:$dir_starpu/"
  done

  n="${1:-14}"
  prefix="g6.2xlarge.starpu_cpu-$n"
  run="mpirun --hostfile $hostfile -x STARPU_FXT_TRACE -x STARPU_FXT_PREFIX -map-by slot:PE=4 $dir_starpu/nbody $n"

  echo "Running: $prefix"
  echo "$run"
  eval "$run"
}

run