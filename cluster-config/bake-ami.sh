#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AMI_ENV_FILE="${SCRIPT_DIR}/ami-bake.env"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
BASE_AMI_CPU=""
BASE_AMI_CUDA=""
INSTANCE_TYPE_CPU="c7i.2xlarge"
INSTANCE_TYPE_CUDA="g5.xlarge"
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

NODE_TYPE=""
AMI_NAME=""
AMI_DESCRIPTION=""
BASE_AMI=""
INSTANCE_TYPE=""
BUILDER_INSTANCE_ID=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] --type cpu|cuda

Options:
  -t, --type <cpu|cuda>      Node type to bake.
  --ami-env <file>           AMI bake env file (default: cluster-config/ami-bake.env).
  --region <region>          AWS region override.
  --base-ami <ami-id>        Override base AMI for this bake.
  --instance-type <type>     Override builder instance type.
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
  if [[ "$NODE_TYPE" != "cpu" && "$NODE_TYPE" != "cuda" ]]; then
    echo "--type must be cpu or cuda" >&2
    exit 1
  fi

  if [[ -z "$BASE_AMI" ]]; then
    if [[ "$NODE_TYPE" == "cuda" ]]; then
      BASE_AMI="$BASE_AMI_CUDA"
    else
      BASE_AMI="$BASE_AMI_CPU"
    fi
  fi

  if [[ -z "$INSTANCE_TYPE" ]]; then
    if [[ "$NODE_TYPE" == "cuda" ]]; then
      INSTANCE_TYPE="$INSTANCE_TYPE_CUDA"
    else
      INSTANCE_TYPE="$INSTANCE_TYPE_CPU"
    fi
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
    echo "Base AMI is required. Set BASE_AMI_${NODE_TYPE^^} in ${AMI_ENV_FILE} or pass --base-ami." >&2
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
}

wait_for_instance_ssh() {
  local host="$1"
  local deadline=$((SECONDS + WAIT_INSTANCE_TIMEOUT_SECS))

  while ((SECONDS < deadline)); do
    if ssh "${SSH_ARGS[@]}" "${SSH_USER}@${host}" 'true' >/dev/null 2>&1; then
      return 0
    fi
    sleep "$WAIT_POLL_SECONDS"
  done

  echo "Timed out waiting for SSH on ${host}" >&2
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

  if [[ "$NODE_TYPE" == "cuda" ]]; then
    setup_local="${SCRIPT_DIR}/setup-cuda.sh"
  else
    setup_local="${SCRIPT_DIR}/setup-cpu.sh"
  fi

  if [[ ! -f "$setup_local" ]]; then
    echo "Setup script not found: $setup_local" >&2
    exit 1
  fi

  setup_remote="~/setup-${NODE_TYPE}.sh"
  echo "Uploading setup script to ${host}..."
  scp "${SCP_ARGS[@]}" "$setup_local" "${SSH_USER}@${host}:${setup_remote}"

  echo "Running setup script on builder..."
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "chmod +x ${setup_remote} && ${setup_remote}")"
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
