#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLUSTER_TOOL="${SCRIPT_DIR}/cluster-mpi.sh"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/experiments.env"

ENV_FILE="${DEFAULT_ENV_FILE}"
MODE="ms"
SESSION_NAME="$(date +%Y%m%d-%H%M%S)"
RESULTS_BASE_DIR="${REPO_ROOT}/results/experiments"
RESULTS_DIR=""

DO_SYNC=1
DO_BUILD=1
BUILD_CLEAN=1
SAVE_LOGS=1

MPI_NODE_COUNTS_RAW="2 4 8"
MS_NODE_COUNTS_RAW="3 5 9"
EXP_LIST_RAW="18 19"
WARMUP_REPS=1
TIMED_REPS=2

GLOBAL_MPIRUN_ARGS="--bind-to none --tag-output"
MPI_MPIRUN_ARGS_EXTRA=""
MS_MPIRUN_ARGS_EXTRA=""

MPI_RUN_BASE_ARGS="--mpi --classic --gpu"
MS_RUN_BASE_ARGS="--master-slave --classic --gpu"

BUILD_ARGS_MPI="USE_MPI=1 USE_CUDA=1 DEBUG=0"
BUILD_ARGS_MS="USE_MPI=0 USE_CUDA=1 DEBUG=0"

COMMON_X_EXPORTS=(
  "STARPU_WORKERS_GETBIND=0"
  "STARPU_SILENT=1"
)

POOL_CONTROL_IPS=()
POOL_MPI_IPS=()
MPI_NODE_COUNTS=()
MS_NODE_COUNTS=()
EXP_LIST=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Automate StarPU MPI vs master-slave experiment runs using cluster-config/cluster-mpi.sh.

Options:
  --mode <mpi|ms|both>       Modes to execute (default: both)
  --env <file>               Env file (default: cluster-config/experiments.env if present)
  --session <name>           Session folder name (default: timestamp)
  --results-dir <path>       Explicit output directory (overrides --session)
  --mpi-nodes "a b c"        Override MPI node list (default: "2 4 8")
  --ms-nodes "a b c"         Override master-slave node list (default: "3 5 9")
  --exp-list "e1 e2"         Override exp list (default: "18 19")
  --warmup-reps <n>          Warmup reps per case (default: 1)
  --timed-reps <n>           Timed reps per case (default: 2)
  --no-sync                  Skip sync before each node-count case
  --no-build                 Skip build before each node-count case
  --no-build-clean           Do not pass --clean to cluster-mpi.sh build
  --no-save-logs             Do not save per-run raw logs
  -h, --help                 Show this help
USAGE
}

log() {
  printf '[experiments] %s\n' "$*"
}

die() {
  printf '[experiments] ERROR: %s\n' "$*" >&2
  exit 1
}

trim_ws() {
  awk '{$1=$1; print}' <<<"$*"
}

parse_space_list() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  read -r -a out_ref <<<"$(trim_ws "$raw")"
}

join_csv() {
  local -a vals=("$@")
  local out=""
  local v
  for v in "${vals[@]}"; do
    if [[ -n "$out" ]]; then
      out+=","
    fi
    out+="$v"
  done
  printf '%s' "$out"
}

sanitize_field() {
  local s="$1"
  s="${s//$'\n'/ }"
  s="${s//$'\r'/ }"
  s="${s//,/;}"
  s="$(trim_ws "$s")"
  printf '%s' "$s"
}

n_bodies_from_exp() {
  local exp="$1"
  printf '%d' "$((1 << (exp + 1)))"
}

paired_nodes() {
  local mode="$1"
  local nodes="$2"
  if [[ "$mode" == "mpi" ]]; then
    printf '%d' "$((nodes + 1))"
  else
    printf '%d' "$((nodes - 1))"
  fi
}

gpu_lanes_for_case() {
  local mode="$1"
  local nodes="$2"
  if [[ "$mode" == "mpi" ]]; then
    printf '%d' "$nodes"
  else
    printf '%d' "$((nodes - 1))"
  fi
}

extract_time_us() {
  local output="$1"
  awk '
    {
      if (match($0, /<stdout>:[0-9]+\.[0-9]+$/)) {
        last = substr($0, RSTART + 9)
      } else if ($0 ~ /^[0-9]+\.[0-9]+$/) {
        last = $0
      }
    }
    END {
      if (last != "") {
        print last
      }
    }
  ' <<<"$output"
}

extract_failure_note() {
  local output="$1"
  local status="$2"
  local line
  line="$(awk '/ERROR|Error|error|Segmentation fault|Permission denied|No network interfaces|timed out|cannot find guid/ {print; exit}' <<<"$output")"
  if [[ -z "$line" ]]; then
    line="exit_status_${status}"
  fi
  sanitize_field "$line"
}

load_pool_ips() {
  local control_file="${SCRIPT_DIR}/.cluster/control_ips.txt"
  local mpi_file="${SCRIPT_DIR}/.cluster/mpi_ips.txt"

  [[ -f "$control_file" ]] || die "Missing ${control_file}. Run ./cluster-config/cluster-mpi.sh set-ips first."

  mapfile -t POOL_CONTROL_IPS < "$control_file"
  if [[ -f "$mpi_file" ]] && [[ -s "$mpi_file" ]]; then
    mapfile -t POOL_MPI_IPS < "$mpi_file"
  else
    POOL_MPI_IPS=("${POOL_CONTROL_IPS[@]}")
  fi

  ((${#POOL_CONTROL_IPS[@]} > 0)) || die "No control IPs found in ${control_file}."
  ((${#POOL_MPI_IPS[@]} > 0)) || die "No MPI IPs available."
  ((${#POOL_CONTROL_IPS[@]} == ${#POOL_MPI_IPS[@]})) || die "control_ips and mpi_ips count mismatch."
}

set_cluster_subset() {
  local nodes="$1"
  local -a ctl_subset=("${POOL_CONTROL_IPS[@]:0:nodes}")
  local -a mpi_subset=("${POOL_MPI_IPS[@]:0:nodes}")

  ((${#ctl_subset[@]} == nodes)) || die "Not enough control IPs for nodes=${nodes}."
  ((${#mpi_subset[@]} == nodes)) || die "Not enough MPI IPs for nodes=${nodes}."

  local ctl_csv mpi_csv
  ctl_csv="$(join_csv "${ctl_subset[@]}")"
  mpi_csv="$(join_csv "${mpi_subset[@]}")"

  log "Selecting first ${nodes} nodes from saved IP pool"
  "${CLUSTER_TOOL}" set-ips "$ctl_csv" "$mpi_csv" >/dev/null
}

build_case() {
  local mode="$1"
  local build_args=""
  if [[ "$mode" == "mpi" ]]; then
    build_args="$BUILD_ARGS_MPI"
  else
    build_args="$BUILD_ARGS_MS"
  fi

  if [[ "$DO_BUILD" -eq 0 ]]; then
    log "Skipping build for mode=${mode}"
    return
  fi

  if [[ "$BUILD_CLEAN" -eq 1 ]]; then
    "${CLUSTER_TOOL}" build starpu --clean "$build_args"
  else
    "${CLUSTER_TOOL}" build starpu "$build_args"
  fi
}

compose_mpirun_args() {
  local mode="$1"
  local np="$2"
  local args="${GLOBAL_MPIRUN_ARGS}"

  local x
  for x in "${COMMON_X_EXPORTS[@]}"; do
    args+=" -x ${x}"
  done

  if [[ "$mode" == "mpi" ]]; then
    if [[ -n "$MPI_MPIRUN_ARGS_EXTRA" ]]; then
      args+=" ${MPI_MPIRUN_ARGS_EXTRA}"
    fi
  else
    args+=" -x STARPU_NMPI_SC=$((np - 1))"
    args+=" -x STARPU_MPI_SC_NTHREADS=2"
    args+=" -x STARPU_MPI_SC_NCUDA=1"
    if [[ -n "$MS_MPIRUN_ARGS_EXTRA" ]]; then
      args+=" ${MS_MPIRUN_ARGS_EXTRA}"
    fi
  fi

  trim_ws "$args"
}

append_result_row() {
  local csv_file="$1"
  shift
  printf '%s\n' "$(join_csv "$@")" >> "$csv_file"
}

generate_summary_csv() {
  local input_csv="$1"
  local output_csv="$2"

  {
    echo "mode,nodes,paired_nodes,exp,n_bodies,parts,gpu_lanes,samples,mean_us,min_us,max_us,stddev_us"
    awk -F',' '
      NR == 1 { next }
      $10 == "timed" && $13 == "1" && $12 != "" {
        key = $2 FS $3 FS $5 FS $6 FS $7 FS $8 FS $9
        v = $12 + 0
        n[key]++
        sum[key] += v
        sumsq[key] += (v * v)
        if (!(key in min) || v < min[key]) min[key] = v
        if (!(key in max) || v > max[key]) max[key] = v
      }
      END {
        for (key in n) {
          mean = sum[key] / n[key]
          var = (sumsq[key] / n[key]) - (mean * mean)
          if (var < 0) var = 0
          stddev = sqrt(var)
          printf "%s,%d,%.6f,%.6f,%.6f,%.6f\n", key, n[key], mean, min[key], max[key], stddev
        }
      }
    ' "$input_csv" | sort -t, -k1,1 -k2,2n -k4,4n -k6,6n
  } > "$output_csv"
}

run_single_case() {
  local mode="$1"
  local nodes="$2"
  local paired="$3"
  local exp="$4"
  local n_bodies="$5"
  local parts="$6"
  local gpu_lanes="$7"
  local rep_kind="$8"
  local rep_idx="$9"
  local csv_file="${10}"
  local logs_dir="${11}"

  local run_args
  if [[ "$mode" == "mpi" ]]; then
    run_args="${MPI_RUN_BASE_ARGS} --exp ${exp} --parts ${parts}"
  else
    run_args="${MS_RUN_BASE_ARGS} --exp ${exp} --parts ${parts}"
  fi

  local mpirun_args
  mpirun_args="$(compose_mpirun_args "$mode" "$nodes")"

  log "Run mode=${mode} nodes=${nodes} exp=${exp} parts=${parts} rep=${rep_kind}:${rep_idx}"

  local output status
  set +e
  output="$("${CLUSTER_TOOL}" run starpu --mpirun-args "$mpirun_args" --run-args "$run_args" 2>&1)"
  status=$?
  set -e

  printf '%s\n' "$output"

  if [[ "$SAVE_LOGS" -eq 1 ]]; then
    local log_file="${logs_dir}/${mode}_n${nodes}_e${exp}_p${parts}_${rep_kind}${rep_idx}.log"
    printf '%s\n' "$output" > "$log_file"
  fi

  local time_us=""
  time_us="$(extract_time_us "$output")"

  local run_ok="1"
  local notes=""
  if [[ "$status" -ne 0 || -z "$time_us" ]]; then
    run_ok="0"
    notes="$(extract_failure_note "$output" "$status")"
  fi

  append_result_row "$csv_file" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$mode" \
    "$nodes" \
    "$nodes" \
    "$paired" \
    "$exp" \
    "$n_bodies" \
    "$parts" \
    "$gpu_lanes" \
    "$rep_kind" \
    "$rep_idx" \
    "$time_us" \
    "$run_ok" \
    "$notes"
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --mode)
        MODE="$2"
        shift 2
        ;;
      --env)
        ENV_FILE="$2"
        shift 2
        ;;
      --session)
        SESSION_NAME="$2"
        shift 2
        ;;
      --results-dir)
        RESULTS_DIR="$2"
        shift 2
        ;;
      --mpi-nodes)
        MPI_NODE_COUNTS_RAW="$2"
        shift 2
        ;;
      --ms-nodes)
        MS_NODE_COUNTS_RAW="$2"
        shift 2
        ;;
      --exp-list)
        EXP_LIST_RAW="$2"
        shift 2
        ;;
      --warmup-reps)
        WARMUP_REPS="$2"
        shift 2
        ;;
      --timed-reps)
        TIMED_REPS="$2"
        shift 2
        ;;
      --no-sync)
        DO_SYNC=0
        shift
        ;;
      --no-build)
        DO_BUILD=0
        shift
        ;;
      --no-build-clean)
        BUILD_CLEAN=0
        shift
        ;;
      --no-save-logs)
        SAVE_LOGS=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

load_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  elif [[ "$ENV_FILE" != "$DEFAULT_ENV_FILE" ]]; then
    die "Env file not found: $ENV_FILE"
  fi
}

validate_config() {
  [[ -x "$CLUSTER_TOOL" ]] || die "cluster tool not executable: $CLUSTER_TOOL"
  command -v awk >/dev/null 2>&1 || die "awk is required"

  case "$MODE" in
    mpi|ms|both) ;;
    *) die "--mode must be mpi, ms, or both" ;;
  esac

  [[ "$WARMUP_REPS" =~ ^[0-9]+$ ]] || die "Invalid WARMUP_REPS: $WARMUP_REPS"
  [[ "$TIMED_REPS" =~ ^[0-9]+$ ]] || die "Invalid TIMED_REPS: $TIMED_REPS"

  parse_space_list "$MPI_NODE_COUNTS_RAW" MPI_NODE_COUNTS
  parse_space_list "$MS_NODE_COUNTS_RAW" MS_NODE_COUNTS
  parse_space_list "$EXP_LIST_RAW" EXP_LIST

  ((${#EXP_LIST[@]} > 0)) || die "EXP_LIST is empty"

  local n e
  for n in "${MPI_NODE_COUNTS[@]}" "${MS_NODE_COUNTS[@]}"; do
    [[ "$n" =~ ^[0-9]+$ ]] || die "Invalid node count: $n"
  done
  for e in "${EXP_LIST[@]}"; do
    [[ "$e" =~ ^[0-9]+$ ]] || die "Invalid exp value: $e"
  done

  load_pool_ips

  local need_max=0
  if [[ "$MODE" == "mpi" || "$MODE" == "both" ]]; then
    for n in "${MPI_NODE_COUNTS[@]}"; do
      ((n > need_max)) && need_max=$n
    done
  fi
  if [[ "$MODE" == "ms" || "$MODE" == "both" ]]; then
    for n in "${MS_NODE_COUNTS[@]}"; do
      ((n > need_max)) && need_max=$n
    done
  fi

  ((${#POOL_CONTROL_IPS[@]} >= need_max)) || die "Need at least ${need_max} saved nodes; found ${#POOL_CONTROL_IPS[@]}"

  if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="${RESULTS_BASE_DIR}/${SESSION_NAME}"
  fi
}

maybe_sync() {
  if [[ "$DO_SYNC" -eq 1 ]]; then
    "${CLUSTER_TOOL}" sync
  else
    log "Skipping sync"
  fi
}

init_results_files() {
  local csv_file="$1"
  mkdir -p "$(dirname "$csv_file")"
  if [[ "$SAVE_LOGS" -eq 1 ]]; then
    mkdir -p "${RESULTS_DIR}/logs"
  fi

  echo "timestamp,mode,nodes,np,paired_nodes,exp,n_bodies,parts,gpu_lanes,rep_kind,rep_idx,time_us,run_ok,notes" > "$csv_file"
}

planned_rows_count() {
  local total=0
  local reps=$((WARMUP_REPS + TIMED_REPS))
  local n

  if [[ "$MODE" == "mpi" || "$MODE" == "both" ]]; then
    for n in "${MPI_NODE_COUNTS[@]}"; do
      total=$((total + (${#EXP_LIST[@]} * 2 * reps)))
    done
  fi
  if [[ "$MODE" == "ms" || "$MODE" == "both" ]]; then
    for n in "${MS_NODE_COUNTS[@]}"; do
      total=$((total + (${#EXP_LIST[@]} * 2 * reps)))
    done
  fi

  printf '%d' "$total"
}

run_mode_matrix() {
  local mode="$1"
  local -a node_list=()
  if [[ "$mode" == "mpi" ]]; then
    node_list=("${MPI_NODE_COUNTS[@]}")
  else
    node_list=("${MS_NODE_COUNTS[@]}")
  fi

  local nodes gpu_lanes paired exp n_bodies rep total_reps rep_kind rep_idx
  local parts_a parts_b

  for nodes in "${node_list[@]}"; do
    gpu_lanes="$(gpu_lanes_for_case "$mode" "$nodes")"
    paired="$(paired_nodes "$mode" "$nodes")"
    ((gpu_lanes > 0)) || die "Computed gpu_lanes=${gpu_lanes} for mode=${mode} nodes=${nodes}"

    set_cluster_subset "$nodes"
    "${CLUSTER_TOOL}" show
    maybe_sync
    build_case "$mode"

    parts_a="$gpu_lanes"
    parts_b="$((gpu_lanes * 4))"

    for exp in "${EXP_LIST[@]}"; do
      n_bodies="$(n_bodies_from_exp "$exp")"
      for parts in "$parts_a" "$parts_b"; do
        total_reps=$((WARMUP_REPS + TIMED_REPS))
        for ((rep = 1; rep <= total_reps; rep++)); do
          if ((rep <= WARMUP_REPS)); then
            rep_kind="warmup"
            rep_idx="$rep"
          else
            rep_kind="timed"
            rep_idx="$((rep - WARMUP_REPS))"
          fi

          run_single_case \
            "$mode" "$nodes" "$paired" "$exp" "$n_bodies" "$parts" "$gpu_lanes" \
            "$rep_kind" "$rep_idx" "$RESULTS_CSV" "$LOGS_DIR"
        done
      done
    done
  done
}

main() {
  if [[ -f "$DEFAULT_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$DEFAULT_ENV_FILE"
  fi

  local preparse=("$@")
  local i
  for ((i = 0; i < ${#preparse[@]}; i++)); do
    if [[ "${preparse[$i]}" == "--env" ]]; then
      if ((i + 1 >= ${#preparse[@]})); then
        die "--env requires a file path"
      fi
      ENV_FILE="${preparse[$((i + 1))]}"
      break
    fi
  done

  load_env_file
  parse_args "$@"
  validate_config

  RESULTS_CSV="${RESULTS_DIR}/results.csv"
  SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
  LOGS_DIR="${RESULTS_DIR}/logs"

  mkdir -p "$RESULTS_DIR"
  init_results_files "$RESULTS_CSV"

  log "Results directory: ${RESULTS_DIR}"
  log "Planned rows: $(planned_rows_count)"

  if [[ "$MODE" == "mpi" || "$MODE" == "both" ]]; then
    run_mode_matrix mpi
  fi
  if [[ "$MODE" == "ms" || "$MODE" == "both" ]]; then
    run_mode_matrix ms
  fi

  generate_summary_csv "$RESULTS_CSV" "$SUMMARY_CSV"

  log "Completed."
  log "results.csv: ${RESULTS_CSV}"
  log "summary.csv: ${SUMMARY_CSV}"
}

main "$@"
