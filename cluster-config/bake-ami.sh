#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AMI_ENV_FILE="${SCRIPT_DIR}/ami-bake.env"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
BASE_AMI_CPU=""
BASE_AMI_CUDA=""
BASE_AMI_MASTER_SLAVE=""
INSTANCE_TYPE_CPU="c7i.2xlarge"
INSTANCE_TYPE_CUDA="g5.xlarge"
INSTANCE_TYPE_MASTER_SLAVE=""
SUBNET_ID=""
declare -a SECURITY_GROUP_IDS=()
IAM_INSTANCE_PROFILE=""
KEY_NAME=""
SSH_USER="ec2-user"
SSH_PORT=22
SSH_KEY=""
SSH_OPTIONS=(
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=10
)
AUTO_DETECT_SSH_USER=1
SSH_USER_CANDIDATES=()
BAKE_SSH_ADDRESS="public"
AMI_NAME_PREFIX="nbody-starpu"
AMI_DESCRIPTION_PREFIX="N-Body StarPU/OpenMP"
INCLUDE_INSTANCE_TYPE_IN_AMI_NAME=1
ROOT_VOLUME_SIZE_GB=50
KEEP_BUILDER=0
WAIT_IMAGE=1
WAIT_INSTANCE_TIMEOUT_SECS=1800
WAIT_IMAGE_TIMEOUT_SECS=3600
WAIT_POLL_SECONDS=10
LOCAL_STARPU_SRC="${HOME}/code/starpu"
REMOTE_STARPU_SRC="\$HOME/code/starpu"
STARPU_REPO_URL="https://github.com/nicolasvanz/starpu-ms-gpu.git"
STARPU_REPO_REF="enhancement/ms-cuda-support"
STARPU_REPO_TOKEN="${GITHUB_TOKEN:-}"
STARPU_MAXMPIDEV="16"

NODE_TYPE=""
AMI_NAME=""
AMI_DESCRIPTION=""
BASE_AMI=""
INSTANCE_TYPE=""
BUILDER_INSTANCE_ID=""
ACTIVE_SSH_USER=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] --type cpu|cuda|master-slave

Options:
  -t, --type <cpu|cuda|master-slave>
                              Node type to bake. Aliases: ms, master_slave.
  --ami-env <file>           AMI bake env file (default: cluster-config/ami-bake.env).
  --region <region>          AWS region override.
  --base-ami <ami-id>        Override base AMI for this bake.
  --instance-type <type>     Override builder instance type.
  --starpu-repo-url <url>    Override StarPU git URL for --type master-slave.
  --starpu-repo-ref <ref>    Override StarPU git ref for --type master-slave.
  --starpu-maxmpidev <n>     Override StarPU --enable-maxmpidev for --type master-slave.
  --subnet-id <subnet-id>    Override subnet id.
  --sg-ids <sg1,sg2>         Override security group ids (comma-separated).
  --key-name <keypair-name>  Override EC2 key pair name.
  --ssh-user <user>          Override SSH username.
  --ssh-key <path>           Override SSH private key path.
  --ami-name <name>          Explicit AMI name.
  --ami-description <text>   Explicit AMI description.
  --keep-builder             Do not terminate builder instance after bake.
  --no-wait-image            Return after create-image (do not wait for availability).
  -h, --help                 Show this help.

Examples:
  $(basename "$0") --type cpu
  $(basename "$0") --type cuda --base-ami ami-abc123 --instance-type g5.xlarge
  $(basename "$0") --type master-slave
USAGE
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    # shellcheck disable=SC1090
    source "$file"
  fi
}

parse_sg_ids_csv() {
  local csv="$1"
  SECURITY_GROUP_IDS=()
  local piece
  local -a pieces=()
  IFS=',' read -r -a pieces <<<"$csv"
  for piece in "${pieces[@]}"; do
    piece="$(trim "$piece")"
    [[ -z "$piece" ]] && continue
    SECURITY_GROUP_IDS+=("$piece")
  done
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      -t|--type)
        NODE_TYPE="$2"
        shift 2
        ;;
      --ami-env)
        AMI_ENV_FILE="$2"
        shift 2
        ;;
      --region)
        REGION="$2"
        shift 2
        ;;
      --base-ami)
        BASE_AMI="$2"
        shift 2
        ;;
      --instance-type)
        INSTANCE_TYPE="$2"
        shift 2
        ;;
      --starpu-repo-url)
        STARPU_REPO_URL="$2"
        shift 2
        ;;
      --starpu-repo-ref)
        STARPU_REPO_REF="$2"
        shift 2
        ;;
      --starpu-maxmpidev)
        STARPU_MAXMPIDEV="$2"
        shift 2
        ;;
      --subnet-id)
        SUBNET_ID="$2"
        shift 2
        ;;
      --sg-ids)
        parse_sg_ids_csv "$2"
        shift 2
        ;;
      --key-name)
        KEY_NAME="$2"
        shift 2
        ;;
      --ssh-user)
        SSH_USER="$2"
        shift 2
        ;;
      --ssh-key)
        SSH_KEY="$2"
        shift 2
        ;;
      --ami-name)
        AMI_NAME="$2"
        shift 2
        ;;
      --ami-description)
        AMI_DESCRIPTION="$2"
        shift 2
        ;;
      --keep-builder)
        KEEP_BUILDER=1
        shift
        ;;
      --no-wait-image)
        WAIT_IMAGE=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

preparse_env_files() {
  while (($# > 0)); do
    case "$1" in
      --ami-env)
        AMI_ENV_FILE="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
}

resolve_defaults() {
  NODE_TYPE="$(printf '%s' "$NODE_TYPE" | tr '[:upper:]' '[:lower:]')"

  case "$NODE_TYPE" in
    ms|master_slave)
      NODE_TYPE="master-slave"
      ;;
  esac

  if [[ "$NODE_TYPE" != "cpu" && "$NODE_TYPE" != "cuda" && "$NODE_TYPE" != "master-slave" ]]; then
    echo "--type must be cpu, cuda, or master-slave" >&2
    exit 1
  fi

  if [[ -z "$BASE_AMI" ]]; then
    case "$NODE_TYPE" in
      cpu)
        BASE_AMI="$BASE_AMI_CPU"
        ;;
      cuda)
        BASE_AMI="$BASE_AMI_CUDA"
        ;;
      master-slave)
        BASE_AMI="${BASE_AMI_MASTER_SLAVE:-$BASE_AMI_CUDA}"
        ;;
    esac
  fi

  if [[ -z "$INSTANCE_TYPE" ]]; then
    case "$NODE_TYPE" in
      cpu)
        INSTANCE_TYPE="$INSTANCE_TYPE_CPU"
        ;;
      cuda)
        INSTANCE_TYPE="$INSTANCE_TYPE_CUDA"
        ;;
      master-slave)
        INSTANCE_TYPE="${INSTANCE_TYPE_MASTER_SLAVE:-$INSTANCE_TYPE_CUDA}"
        ;;
    esac
  fi

  local instance_type_label
  instance_type_label="$(printf '%s' "$INSTANCE_TYPE" | sed 's/[^[:alnum:]._:-]/-/g')"

  if [[ -z "$AMI_NAME" ]]; then
    if [[ "$INCLUDE_INSTANCE_TYPE_IN_AMI_NAME" -eq 1 ]]; then
      AMI_NAME="${AMI_NAME_PREFIX}-${NODE_TYPE}-${instance_type_label}-$(date +%Y%m%d-%H%M%S)"
    else
      AMI_NAME="${AMI_NAME_PREFIX}-${NODE_TYPE}-$(date +%Y%m%d-%H%M%S)"
    fi
  fi

  if [[ -z "$AMI_DESCRIPTION" ]]; then
    AMI_DESCRIPTION="${AMI_DESCRIPTION_PREFIX} ${NODE_TYPE} instance=${INSTANCE_TYPE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi

  if [[ -z "$REGION" ]]; then
    echo "AWS region is required. Set REGION in ${AMI_ENV_FILE} or pass --region." >&2
    exit 1
  fi

  if [[ -z "$BASE_AMI" ]]; then
    local base_key="BASE_AMI_${NODE_TYPE^^}"
    if [[ "$NODE_TYPE" == "master-slave" ]]; then
      base_key="BASE_AMI_MASTER_SLAVE (or BASE_AMI_CUDA fallback)"
    fi
    echo "Base AMI is required. Set ${base_key} in ${AMI_ENV_FILE} or pass --base-ami." >&2
    exit 1
  fi

  if [[ -z "$SUBNET_ID" ]]; then
    echo "SUBNET_ID is required. Set it in ${AMI_ENV_FILE} or pass --subnet-id." >&2
    exit 1
  fi

  if ((${#SECURITY_GROUP_IDS[@]} == 0)); then
    echo "At least one SECURITY_GROUP_ID is required." >&2
    exit 1
  fi

  if [[ -z "$KEY_NAME" ]]; then
    echo "KEY_NAME is required for SSH provisioning." >&2
    exit 1
  fi

  if [[ -z "$SSH_KEY" ]]; then
    echo "SSH_KEY is required for SSH provisioning." >&2
    exit 1
  fi

  if [[ ! -f "$SSH_KEY" ]]; then
    echo "SSH key file not found: $SSH_KEY" >&2
    exit 1
  fi

  if [[ "$BAKE_SSH_ADDRESS" != "public" && "$BAKE_SSH_ADDRESS" != "private" ]]; then
    echo "BAKE_SSH_ADDRESS must be 'public' or 'private'" >&2
    exit 1
  fi

  if [[ -z "${STARPU_REPO_TOKEN}" && -n "${GITHUB_TOKEN:-}" ]]; then
    STARPU_REPO_TOKEN="${GITHUB_TOKEN}"
  fi

  if ! [[ "${STARPU_MAXMPIDEV}" =~ ^[0-9]+$ ]]; then
    echo "STARPU_MAXMPIDEV must be a non-negative integer (got: ${STARPU_MAXMPIDEV})." >&2
    exit 1
  fi
}

build_ssh_args() {
  SSH_ARGS=(-p "$SSH_PORT")
  SCP_ARGS=(-P "$SSH_PORT")

  if [[ -n "$SSH_KEY" ]]; then
    SSH_ARGS+=( -i "$SSH_KEY" )
    SCP_ARGS+=( -i "$SSH_KEY" )
  fi

  if ((${#SSH_OPTIONS[@]} > 0)); then
    SSH_ARGS+=( "${SSH_OPTIONS[@]}" )
    SCP_ARGS+=( "${SSH_OPTIONS[@]}" )
  fi

  # Never prompt for password in automation loops.
  SSH_ARGS+=( -o BatchMode=yes -o PreferredAuthentications=publickey )
  SCP_ARGS+=( -o BatchMode=yes )

  ACTIVE_SSH_USER="$SSH_USER"
}

candidate_ssh_users() {
  local users=("$SSH_USER")
  local defaults=("${SSH_USER_CANDIDATES[@]}")

  if [[ "$AUTO_DETECT_SSH_USER" -eq 1 ]]; then
    if ((${#defaults[@]} == 0)); then
      defaults=(ec2-user ubuntu admin debian centos fedora rocky almalinux)
    fi
  else
    defaults=()
  fi

  local u
  for u in "${defaults[@]}"; do
    [[ -z "$u" || "$u" == "$SSH_USER" ]] && continue
    users+=("$u")
  done

  printf '%s\n' "${users[@]}"
}

wait_for_instance_ssh() {
  local host="$1"
  local deadline=$((SECONDS + WAIT_INSTANCE_TIMEOUT_SECS))
  local attempt=0
  local last_err=""

  while ((SECONDS < deadline)); do
    attempt=$((attempt + 1))
    echo "Waiting for SSH on ${host} (attempt ${attempt}, ${WAIT_POLL_SECONDS}s poll interval)..."

    local user
    while IFS= read -r user; do
      [[ -z "$user" ]] && continue
      echo "  trying user: ${user}"
      if last_err="$(ssh "${SSH_ARGS[@]}" "${user}@${host}" 'true' 2>&1)"; then
        ACTIVE_SSH_USER="$user"
        echo "SSH ready on ${host} with user '${ACTIVE_SSH_USER}'."
        return 0
      fi
    done < <(candidate_ssh_users)

    local remaining=$((deadline - SECONDS))
    if ((remaining < 0)); then
      remaining=0
    fi
    echo "  not ready yet, ~${remaining}s remaining before timeout"
    sleep "$WAIT_POLL_SECONDS"
  done

  echo "Timed out waiting for SSH on ${host}" >&2
  if [[ -n "$last_err" ]]; then
    echo "Last SSH error: $(printf '%s\n' "$last_err" | tail -n 1)" >&2
  fi
  return 1
}

cleanup() {
  local code="$1"

  if [[ "$code" -ne 0 ]]; then
    echo "Bake failed." >&2
    if [[ -n "$BUILDER_INSTANCE_ID" ]]; then
      echo "Builder instance left running for debugging: ${BUILDER_INSTANCE_ID}" >&2
    fi
  fi
}

trap 'cleanup $?' EXIT

launch_builder_instance() {
  echo "Launching ${NODE_TYPE} builder instance in ${REGION}..."

  local -a cmd=(
    aws ec2 run-instances
    --region "$REGION"
    --image-id "$BASE_AMI"
    --instance-type "$INSTANCE_TYPE"
    --key-name "$KEY_NAME"
    --subnet-id "$SUBNET_ID"
    --count 1
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${ROOT_VOLUME_SIZE_GB},VolumeType=gp3,DeleteOnTermination=true}"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${AMI_NAME}-builder},{Key=Project,Value=nbody},{Key=Purpose,Value=ami-bake},{Key=NodeType,Value=${NODE_TYPE}}]"
    --query 'Instances[0].InstanceId'
    --output text
  )

  if ((${#SECURITY_GROUP_IDS[@]} > 0)); then
    cmd+=(--security-group-ids "${SECURITY_GROUP_IDS[@]}")
  fi

  if [[ -n "$IAM_INSTANCE_PROFILE" ]]; then
    cmd+=(--iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}")
  fi

  BUILDER_INSTANCE_ID="$("${cmd[@]}")"
  echo "Builder instance id: ${BUILDER_INSTANCE_ID}"

  aws ec2 wait instance-running --region "$REGION" --instance-ids "$BUILDER_INSTANCE_ID"
  aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$BUILDER_INSTANCE_ID"
}

get_builder_host() {
  local query_field
  if [[ "$BAKE_SSH_ADDRESS" == "private" ]]; then
    query_field='Reservations[0].Instances[0].PrivateIpAddress'
  else
    query_field='Reservations[0].Instances[0].PublicIpAddress'
  fi

  local host
  host="$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$BUILDER_INSTANCE_ID" \
    --query "$query_field" \
    --output text)"

  if [[ -z "$host" || "$host" == "None" ]]; then
    echo "Could not resolve ${BAKE_SSH_ADDRESS} IP for builder instance ${BUILDER_INSTANCE_ID}." >&2
    exit 1
  fi

  printf '%s' "$host"
}

run_setup_script() {
  local host="$1"
  local setup_local setup_remote
  local master_slave_src_dir="${REMOTE_STARPU_SRC}"

  case "$NODE_TYPE" in
    cpu)
      setup_local="${SCRIPT_DIR}/setup-cpu.sh"
      ;;
    cuda)
      setup_local="${SCRIPT_DIR}/setup-cuda.sh"
      ;;
    master-slave)
      setup_local="${SCRIPT_DIR}/setup-ms.sh"
      ;;
    *)
      echo "Unsupported NODE_TYPE for setup script: ${NODE_TYPE}" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$setup_local" ]]; then
    echo "Setup script not found: $setup_local" >&2
    exit 1
  fi

  if [[ "$NODE_TYPE" == "master-slave" ]]; then
    local remote_src="${REMOTE_STARPU_SRC}"

    # Common pitfall: REMOTE_STARPU_SRC accidentally set with local $HOME
    # (e.g. /home/nvanz/...) while remote user is ubuntu/ec2-user.
    # Rewrite that into remote-home-relative path.
    if [[ "$remote_src" == "${HOME}"* ]]; then
      local suffix="${remote_src#${HOME}}"
      remote_src="\$HOME${suffix}"
      echo "Rewriting REMOTE_STARPU_SRC to remote user home: ${remote_src}"
    fi
    master_slave_src_dir="${remote_src}"

    if [[ -n "${STARPU_REPO_URL}" ]]; then
      echo "Using remote StarPU repo for master-slave bake:"
      echo "  URL: ${STARPU_REPO_URL}"
      echo "  REF: ${STARPU_REPO_REF}"
    else
      local local_src="${LOCAL_STARPU_SRC}"
      if [[ ! -d "$local_src" ]]; then
        echo "Local StarPU source not found: ${local_src}" >&2
        echo "Set LOCAL_STARPU_SRC in ${AMI_ENV_FILE} or export it before running bake-ami.sh." >&2
        exit 1
      fi

      local local_src_human
      local_src_human="$(du -sh "$local_src" | awk '{print $1}')"
      echo "STARPU_REPO_URL is empty; uploading local StarPU source (${local_src_human}) to builder (${remote_src})..."
      ssh "${SSH_ARGS[@]}" "${ACTIVE_SSH_USER}@${host}" \
        "mkdir -p \"\$(dirname ${remote_src})\" && rm -rf \"${remote_src}\""

      local base_name
      base_name="$(basename "$local_src")"
      local base_parent
      base_parent="$(dirname "$local_src")"

      local -a tar_cmd=(
        tar -C "$base_parent" -czf -
        --exclude="${base_name}/.git"
        --exclude="${base_name}/autom4te.cache"
        --exclude="${base_name}/build"
        --exclude="${base_name}/build-*"
        --exclude="${base_name}/src/.libs"
        --exclude="${base_name}/**/.libs"
        "$base_name"
      )

      if command -v pv >/dev/null 2>&1; then
        local total_bytes
        total_bytes="$(du -sb "$local_src" | awk '{print $1}')"
        "${tar_cmd[@]}" \
          | pv -s "$total_bytes" \
          | ssh "${SSH_ARGS[@]}" "${ACTIVE_SSH_USER}@${host}" \
            "tar -xzf - -C \"\$(dirname ${remote_src})\""
      else
        "${tar_cmd[@]}" \
          | ssh "${SSH_ARGS[@]}" "${ACTIVE_SSH_USER}@${host}" \
            "tar -xzf - -C \"\$(dirname ${remote_src})\""
      fi
    fi
  fi

  setup_remote="~/setup-${NODE_TYPE}.sh"
  echo "Uploading setup script to ${host}..."
  scp "${SCP_ARGS[@]}" "$setup_local" "${ACTIVE_SSH_USER}@${host}:${setup_remote}"

  echo "Running setup script on builder..."
  if [[ "$NODE_TYPE" == "master-slave" ]]; then
    local setup_cmd
    setup_cmd="chmod +x ${setup_remote} && STARPU_SRC_DIR=${master_slave_src_dir}"
    setup_cmd+=" STARPU_MAXMPIDEV=$(printf '%q' "${STARPU_MAXMPIDEV}")"
    if [[ -n "${STARPU_REPO_URL}" ]]; then
      setup_cmd+=" STARPU_REPO_URL=$(printf '%q' "${STARPU_REPO_URL}")"
      setup_cmd+=" STARPU_REPO_REF=$(printf '%q' "${STARPU_REPO_REF}")"
      if [[ -n "${STARPU_REPO_TOKEN}" ]]; then
        setup_cmd+=" STARPU_REPO_TOKEN=$(printf '%q' "${STARPU_REPO_TOKEN}")"
      fi
    fi
    setup_cmd+=" ${setup_remote}"

    ssh "${SSH_ARGS[@]}" "${ACTIVE_SSH_USER}@${host}" \
      "bash -lc $(printf '%q' "${setup_cmd}")"
  else
    ssh "${SSH_ARGS[@]}" "${ACTIVE_SSH_USER}@${host}" \
      "bash -lc $(printf '%q' "chmod +x ${setup_remote} && ${setup_remote}")"
  fi
}

create_ami() {
  echo "Creating AMI ${AMI_NAME} from ${BUILDER_INSTANCE_ID}..."

  local image_id
  image_id="$(aws ec2 create-image \
    --region "$REGION" \
    --instance-id "$BUILDER_INSTANCE_ID" \
    --name "$AMI_NAME" \
    --description "$AMI_DESCRIPTION" \
    --query 'ImageId' \
    --output text)"

  echo "AMI requested: ${image_id}"

  if [[ "$WAIT_IMAGE" -eq 1 ]]; then
    local deadline=$((SECONDS + WAIT_IMAGE_TIMEOUT_SECS))
    while ((SECONDS < deadline)); do
      local state
      state="$(aws ec2 describe-images --region "$REGION" --image-ids "$image_id" --query 'Images[0].State' --output text)"
      if [[ "$state" == "available" ]]; then
        echo "AMI available: ${image_id}"
        break
      fi
      if [[ "$state" == "failed" ]]; then
        echo "AMI creation failed for ${image_id}" >&2
        exit 1
      fi
      sleep "$WAIT_POLL_SECONDS"
    done

    local final_state
    final_state="$(aws ec2 describe-images --region "$REGION" --image-ids "$image_id" --query 'Images[0].State' --output text)"
    if [[ "$final_state" != "available" ]]; then
      echo "Timed out waiting for AMI ${image_id} to become available (state=${final_state})." >&2
      exit 1
    fi
  fi

  echo
  echo "Bake complete"
  echo "  Type: ${NODE_TYPE}"
  echo "  AMI:  ${image_id}"
  echo "  Name: ${AMI_NAME}"

  if [[ "$KEEP_BUILDER" -eq 0 ]]; then
    echo "Terminating builder instance ${BUILDER_INSTANCE_ID}..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$BUILDER_INSTANCE_ID" >/dev/null
    BUILDER_INSTANCE_ID=""
  else
    echo "Keeping builder instance: ${BUILDER_INSTANCE_ID}"
  fi
}

main() {
  preparse_env_files "$@"
  load_env_file "$AMI_ENV_FILE"
  parse_args "$@"
  resolve_defaults

  require_cmd aws
  require_cmd ssh
  require_cmd scp
  build_ssh_args

  launch_builder_instance

  local host
  host="$(get_builder_host)"
  echo "Builder ${BAKE_SSH_ADDRESS} IP: ${host}"

  wait_for_instance_ssh "$host"
  run_setup_script "$host"
  create_ami
}

main "$@"
