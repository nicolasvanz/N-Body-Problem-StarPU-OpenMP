#!/usr/bin/env bash
set -euo pipefail

STARPU_VER="1.4.7"
FXT_VER="0.3.15"
PREFIX="/usr/local"
BUILD_DIR="${HOME}/build-starpu-${STARPU_VER}"
NPROC="$(nproc)"

STARPU_TARBALL="starpu-${STARPU_VER}.tar.gz"
STARPU_URL="https://files.inria.fr/starpu/starpu-${STARPU_VER}/${STARPU_TARBALL}"

FXT_TARBALL="fxt-${FXT_VER}.tar.gz"
FXT_URL_PRIMARY="https://download.savannah.gnu.org/releases/fkt/${FXT_TARBALL}"
FXT_URL_MIRROR="https://download-mirror.savannah.gnu.org/releases/fkt/${FXT_TARBALL}"

ENV_FILE="/etc/profile.d/starpu.sh"
TRACE_DIR="${HOME}/starpu_traces"
TRACE_PREFIX="${TRACE_DIR}/fxt"

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
  echo "==> [1/9] Installing base dependencies..."

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf -y groupinstall "Development Tools" || true
    sudo dnf -y install \
      gcc gcc-c++ make automake autoconf libtool \
      wget tar gzip bzip2 xz pkgconf-pkg-config \
      hwloc hwloc-devel numactl numactl-devel \
      openmpi openmpi-devel \
      nmap
  elif command -v yum >/dev/null 2>&1; then
    sudo yum -y groupinstall "Development Tools" || true
    sudo yum -y install \
      gcc gcc-c++ make automake autoconf libtool \
      wget tar gzip bzip2 xz pkgconfig \
      hwloc hwloc-devel numactl numactl-devel \
      openmpi openmpi-devel \
      nmap
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y \
      build-essential autoconf automake libtool \
      wget tar gzip bzip2 xz-utils pkg-config \
      libhwloc-dev libnuma-dev \
      openmpi-bin libopenmpi-dev \
      nmap
  else
    echo "ERROR: No supported package manager found (dnf/yum/apt-get)."
    exit 1
  fi
}

ensure_cuda_toolkit() {
  echo "==> [2/9] Checking CUDA toolkit..."

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
  echo "==> [3/9] Building and installing FxT ${FXT_VER}..."
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

write_env() {
  echo "==> [4/9] Writing environment exports to ${ENV_FILE}..."
  sudo tee "${ENV_FILE}" >/dev/null <<EOF_INNER
# StarPU/FxT/CUDA environment (installed under ${PREFIX})
export PATH="${PREFIX}/bin:${CUDA_HOME}/bin:\$PATH"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:\${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:\${LD_LIBRARY_PATH:-}"

# Enable FxT tracing by default
export STARPU_FXT_TRACE=1
export STARPU_FXT_PREFIX="\$HOME/starpu_traces/fxt"
EOF_INNER

  # shellcheck disable=SC1090
  source "${ENV_FILE}"
}

configure_linker() {
  echo "==> [5/9] Updating dynamic linker cache..."
  sudo tee /etc/ld.so.conf.d/starpu-local.conf >/dev/null <<EOF_INNER
${PREFIX}/lib
${PREFIX}/lib64
${CUDA_HOME}/lib64
${CUDA_HOME}/lib
EOF_INNER
  sudo ldconfig
}

prepare_traces() {
  echo "==> [6/9] Creating trace folders..."
  mkdir -p "${TRACE_DIR}" "${TRACE_PREFIX}"
}

build_starpu_cuda() {
  echo "==> [7/9] Building and installing StarPU ${STARPU_VER} with CUDA..."
  cd "${BUILD_DIR}"
  rm -rf "starpu-${STARPU_VER}" "${STARPU_TARBALL}"

  download_tarball "${STARPU_TARBALL}" "${STARPU_URL}"
  tar -xzf "${STARPU_TARBALL}"
  cd "starpu-${STARPU_VER}"

  mkdir -p build
  cd build

  if ! command -v mpicc >/dev/null 2>&1; then
    for p in /usr/lib64/openmpi/bin /usr/lib/x86_64-linux-gnu/openmpi/bin /usr/local/bin; do
      if [[ -x "${p}/mpicc" ]]; then
        export PATH="${p}:${PATH}"
        break
      fi
    done
  fi

  if command -v mpicc >/dev/null 2>&1; then export CC=mpicc; fi
  if command -v mpicxx >/dev/null 2>&1; then export CXX=mpicxx; fi

  ../configure \
    --prefix="${PREFIX}" \
    --enable-cuda \
    --disable-opencl \
    --enable-fxt \
    --enable-mpi \
    CPPFLAGS="-I${CUDA_HOME}/include" \
    LDFLAGS="-L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib"

  make -j"${NPROC}"
  sudo env PATH="$PATH" make install
  sudo ldconfig
}

verify() {
  echo "==> [8/9] Verifying StarPU pkg-config registration..."

  if pkg-config --exists starpu-1.4; then
    echo "OK: starpu-1.4 version: $(pkg-config --modversion starpu-1.4)"
  else
    echo "ERROR: pkg-config cannot find starpu-1.4"
    exit 1
  fi

  echo "==> [9/9] Checking GPU visibility (non-fatal if nvidia-smi is unavailable)..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found; skipping GPU runtime check."
  fi

  echo
  echo "DONE."
  echo "Open a new shell or run: source ${ENV_FILE}"
  echo "FxT traces will be written under: \$HOME/starpu_traces/fxt"
}

install_base_deps
ensure_cuda_toolkit
detect_cuda_home
install_fxt
write_env
configure_linker
prepare_traces
build_starpu_cuda
verify
