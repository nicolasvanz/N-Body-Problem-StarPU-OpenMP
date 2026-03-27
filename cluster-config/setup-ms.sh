#!/usr/bin/env bash
set -euo pipefail

FXT_VER="0.3.15"
PREFIX="/usr/local"
BUILD_DIR="${HOME}/build-starpu-ms"
NPROC="$(nproc)"

STARPU_SRC_DIR="${STARPU_SRC_DIR:-${HOME}/code/starpu}"
STARPU_REPO_URL="${STARPU_REPO_URL:-}"
STARPU_REPO_REF="${STARPU_REPO_REF:-master}"
STARPU_REPO_TOKEN="${STARPU_REPO_TOKEN:-}"
STARPU_MAXMPIDEV="${STARPU_MAXMPIDEV:-16}"
ENV_FILE="/etc/profile.d/starpu.sh"
TRACE_DIR="${HOME}/starpu_traces"
TRACE_PREFIX="${TRACE_DIR}/fxt"

FXT_TARBALL="fxt-${FXT_VER}.tar.gz"
FXT_URL_PRIMARY="https://download.savannah.gnu.org/releases/fkt/${FXT_TARBALL}"
FXT_URL_MIRROR="https://download-mirror.savannah.gnu.org/releases/fkt/${FXT_TARBALL}"

download_tarball() {
  local output="$1"
  shift

  local url
  for url in "$@"; do
    echo "Downloading ${url} ..."
    if command -v curl >/dev/null 2>&1; then
      if curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 -o "${output}" "${url}"; then
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget --tries=5 --timeout=30 -O "${output}" "${url}"; then
        return 0
      fi
    fi
    echo "WARN: failed to download ${url}"
  done

  echo "ERROR: unable to download ${output}"
  return 1
}

install_base_deps() {
  echo "==> [1/10] Installing base dependencies..."

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf -y groupinstall "Development Tools" || true
    sudo dnf -y install \
      gcc gcc-c++ make automake autoconf libtool \
      wget tar gzip bzip2 xz pkgconf-pkg-config \
      git \
      hwloc hwloc-devel numactl numactl-devel \
      openmpi openmpi-devel \
      nmap
  elif command -v yum >/dev/null 2>&1; then
    sudo yum -y groupinstall "Development Tools" || true
    sudo yum -y install \
      gcc gcc-c++ make automake autoconf libtool \
      wget tar gzip bzip2 xz pkgconfig \
      git \
      hwloc hwloc-devel numactl numactl-devel \
      openmpi openmpi-devel \
      nmap
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y \
      build-essential autoconf automake libtool \
      libtool-bin \
      wget tar gzip bzip2 xz-utils pkg-config \
      git \
      libhwloc-dev libnuma-dev \
      openmpi-bin libopenmpi-dev \
      nmap
  else
    echo "ERROR: No supported package manager found (dnf/yum/apt-get)."
    exit 1
  fi

  if ! command -v libtool >/dev/null 2>&1; then
    # Debian/Ubuntu sometimes split the runtime binary into libtool-bin.
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get install -y libtool-bin
    fi
  fi

  if ! command -v libtool >/dev/null 2>&1; then
    echo "ERROR: libtool command not found after dependency installation."
    exit 1
  fi
}

ensure_mpi_wrappers_on_path() {
  echo "==> [2/10] Ensuring MPI wrappers are on PATH..."

  local mpi_bin=""
  if command -v mpicc >/dev/null 2>&1; then
    mpi_bin="$(dirname "$(command -v mpicc)")"
  else
    for p in /usr/lib64/openmpi/bin /usr/lib/x86_64-linux-gnu/openmpi/bin /usr/local/openmpi/bin; do
      if [[ -x "${p}/mpicc" ]]; then
        mpi_bin="${p}"
        break
      fi
    done
  fi

  if [[ -z "${mpi_bin}" ]]; then
    echo "WARN: mpicc not found after package install; continuing."
    return 0
  fi

  export PATH="${mpi_bin}:${PATH}"

  sudo tee /etc/profile.d/openmpi-path.sh >/dev/null <<EOF
# Ensure OpenMPI compiler/runtime wrappers are on PATH for all users.
if [ -d "${mpi_bin}" ]; then
  case ":\$PATH:" in
    *:"${mpi_bin}":*) ;;
    *) export PATH="${mpi_bin}:\$PATH" ;;
  esac
fi
EOF

  for wrapper in mpicc mpicxx mpic++ mpirun mpiexec; do
    if [[ -x "${mpi_bin}/${wrapper}" ]]; then
      sudo ln -sfn "${mpi_bin}/${wrapper}" "/usr/local/bin/${wrapper}"
    fi
  done

  echo "OK: MPI wrappers available from ${mpi_bin}"
}

ensure_cuda_toolkit() {
  echo "==> [3/10] Checking CUDA toolkit..."

  if command -v nvcc >/dev/null 2>&1; then
    echo "Found nvcc: $(command -v nvcc)"
    return
  fi

  echo "nvcc not found, attempting toolkit install..."

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf -y install cuda-toolkit || true
  elif command -v yum >/dev/null 2>&1; then
    sudo yum -y install cuda-toolkit || true
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y nvidia-cuda-toolkit || true
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not available after install attempt."
    echo "Install CUDA toolkit on this node and re-run this script."
    exit 1
  fi

  echo "Found nvcc: $(command -v nvcc)"
}

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Using CUDA_HOME=${CUDA_HOME}"
    return
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export CUDA_HOME
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME="/usr/local/cuda"
    export CUDA_HOME
  else
    echo "ERROR: Unable to resolve CUDA_HOME."
    exit 1
  fi

  echo "Using CUDA_HOME=${CUDA_HOME}"
}

install_fxt() {
  echo "==> [4/10] Building and installing FxT ${FXT_VER}..."
  rm -rf "${BUILD_DIR}/fxt-${FXT_VER}" "${BUILD_DIR}/${FXT_TARBALL}"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}"

  download_tarball "${FXT_TARBALL}" "${FXT_URL_PRIMARY}" "${FXT_URL_MIRROR}"
  tar -xzf "${FXT_TARBALL}"
  cd "fxt-${FXT_VER}"
  ./configure --prefix="${PREFIX}"
  make -j"${NPROC}"
  sudo env PATH="$PATH" make install
}

prepare_starpu_source() {
  echo "==> [5/10] Preparing custom StarPU source..."

  if [[ -n "${STARPU_REPO_URL}" ]]; then
    echo "Cloning StarPU from ${STARPU_REPO_URL} (ref=${STARPU_REPO_REF}) into ${STARPU_SRC_DIR} ..."
    rm -rf "${STARPU_SRC_DIR}"
    mkdir -p "$(dirname "${STARPU_SRC_DIR}")"

    if [[ -n "${STARPU_REPO_TOKEN}" && "${STARPU_REPO_URL}" == https://github.com/* ]]; then
      local auth_header
      auth_header="$(printf 'x-access-token:%s' "${STARPU_REPO_TOKEN}" | base64 | tr -d '\n')"
      git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
        clone --recursive --branch "${STARPU_REPO_REF}" "${STARPU_REPO_URL}" "${STARPU_SRC_DIR}"
    else
      git clone --recursive --branch "${STARPU_REPO_REF}" "${STARPU_REPO_URL}" "${STARPU_SRC_DIR}"
    fi
  else
    echo "Checking custom StarPU source at ${STARPU_SRC_DIR}..."
    if [[ ! -d "${STARPU_SRC_DIR}" ]]; then
      echo "ERROR: StarPU source directory not found: ${STARPU_SRC_DIR}"
      echo "Expected your custom StarPU checkout to be present there."
      echo "Tip: set STARPU_SRC_DIR=/path/to/starpu when running this script."
      exit 1
    fi
  fi

  if [[ ! -f "${STARPU_SRC_DIR}/configure" ]]; then
    if [[ -x "${STARPU_SRC_DIR}/autogen.sh" ]]; then
      echo "Running autogen.sh in ${STARPU_SRC_DIR} ..."
      (cd "${STARPU_SRC_DIR}" && ./autogen.sh)
    else
      echo "ERROR: configure not found and autogen.sh not executable in ${STARPU_SRC_DIR}"
      exit 1
    fi
  fi
}

write_env() {
  echo "==> [6/10] Writing environment exports to ${ENV_FILE}..."
  sudo tee "${ENV_FILE}" >/dev/null <<EOF
# StarPU/FxT/CUDA environment (installed under ${PREFIX})
export CUDA_HOME="${CUDA_HOME}"
export PATH="${PREFIX}/bin:${CUDA_HOME}/bin:\$PATH"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:\${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:\${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:\${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:\${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:\${CPLUS_INCLUDE_PATH:-}"

# Enable FxT tracing by default
export STARPU_FXT_TRACE=1
export STARPU_FXT_PREFIX="\$HOME/starpu_traces/fxt"
EOF

  # shellcheck disable=SC1090
  source "${ENV_FILE}"
}

configure_linker() {
  echo "==> [7/10] Updating dynamic linker cache..."
  sudo tee /etc/ld.so.conf.d/starpu-local.conf >/dev/null <<EOF
${PREFIX}/lib
${PREFIX}/lib64
${CUDA_HOME}/lib64
${CUDA_HOME}/lib
EOF
  sudo ldconfig
}

build_starpu_ms() {
  echo "==> [8/10] Building and installing custom StarPU (CUDA + FxT + MPI server-client)..."
  echo "Using STARPU_MAXMPIDEV=${STARPU_MAXMPIDEV}"
  cd "${STARPU_SRC_DIR}"
  mkdir -p build-ms
  cd build-ms

  if ! [[ "${STARPU_MAXMPIDEV}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STARPU_MAXMPIDEV must be a non-negative integer (got: ${STARPU_MAXMPIDEV})"
    exit 1
  fi

  if command -v mpicc >/dev/null 2>&1; then
    export CC=mpicc
  fi
  if command -v mpicxx >/dev/null 2>&1; then
    export CXX=mpicxx
  fi

  ../configure \
    --prefix="${PREFIX}" \
    --enable-cuda \
    --disable-opencl \
    --disable-starpupy \
    --enable-fxt \
    --enable-mpi-server-client \
    --enable-maxmpidev="${STARPU_MAXMPIDEV}" \
    CPPFLAGS="-I${CUDA_HOME}/include" \
    LDFLAGS="-L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib"

  make -j"${NPROC}"
  sudo env PATH="$PATH" make install
  sudo ldconfig
}

prepare_traces() {
  echo "==> [9/10] Creating trace folders..."
  mkdir -p "${TRACE_DIR}" "${TRACE_PREFIX}"
}

verify() {
  echo "==> [10/10] Verifying StarPU pkg-config registration..."
  if pkg-config --exists starpu-1.4; then
    echo "OK: starpu-1.4 version: $(pkg-config --modversion starpu-1.4)"
  else
    echo "ERROR: pkg-config cannot find starpu-1.4"
    exit 1
  fi

  echo
  echo "DONE."
  echo "Custom StarPU source built from: ${STARPU_SRC_DIR}"
  echo "Open a new shell or run: source ${ENV_FILE}"
}

main() {
  install_base_deps
  ensure_mpi_wrappers_on_path
  ensure_cuda_toolkit
  detect_cuda_home
  install_fxt
  prepare_starpu_source
  write_env
  configure_linker
  build_starpu_ms
  prepare_traces
  verify
}

main "$@"
