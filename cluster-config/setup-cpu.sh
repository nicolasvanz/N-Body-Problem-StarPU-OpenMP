#!/bin/bash
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

# Where FxT traces should go (per-user, per-machine).
# StarPU will typically create files like: ${STARPU_FXT_PREFIX}/prof_file_*
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

ensure_clang18_openmp() {
  echo "==> [2/10] Installing clang-18 + OpenMP runtime..."

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf -y install clang18 clang18-devel clang18-tools-extra lld18
    # On AL2023, libomp15 and libomp18 conflict; ensure clang18's runtime is selected.
    sudo dnf -y remove libomp libomp-devel || true
    sudo dnf -y install libomp18 libomp18-devel --allowerasing
  elif command -v yum >/dev/null 2>&1; then
    sudo yum -y install clang clang-tools-extra lld libomp libomp-devel
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y clang-18 lld-18 libomp-18-dev
  else
    echo "ERROR: No supported package manager found (dnf/yum/apt-get)."
    exit 1
  fi

  local clang18_bin
  if command -v clang-18 >/dev/null 2>&1; then
    clang18_bin="clang-18"
  elif command -v clang18 >/dev/null 2>&1; then
    clang18_bin="clang18"
  elif command -v clang >/dev/null 2>&1; then
    clang18_bin="clang"
  else
    echo "ERROR: clang was not installed."
    exit 1
  fi

  if ! printf '#include <omp.h>\nint main(void){return 0;}\n' | \
       "${clang18_bin}" -fopenmp -x c - -fsyntax-only; then
    echo "ERROR: clang OpenMP probe failed (omp.h/runtime not usable)."
    exit 1
  fi

  echo "OK: ${clang18_bin} OpenMP toolchain is available."
}

ensure_mpi_wrappers_on_path() {
  echo "==> [3/10] Ensuring MPI wrappers are on PATH..."

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

echo "==> [1/10] Installing dependencies..."

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

ensure_clang18_openmp
ensure_mpi_wrappers_on_path

echo "==> [4/10] Creating trace folders..."
mkdir -p "${TRACE_DIR}"
mkdir -p "${TRACE_PREFIX}"

echo "==> [5/10] Creating build dir: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "==> [6/10] Building and installing FxT ${FXT_VER}..."
rm -rf "fxt-${FXT_VER}" "${FXT_TARBALL}"
download_tarball "${FXT_TARBALL}" "${FXT_URL_PRIMARY}" "${FXT_URL_MIRROR}"
tar -xzf "${FXT_TARBALL}"
cd "fxt-${FXT_VER}"
./configure --prefix="${PREFIX}"
make -j"${NPROC}"
sudo env PATH="$PATH" make install
cd "${BUILD_DIR}"

echo "==> [7/10] Writing environment exports to ${ENV_FILE} (system-wide)..."
sudo tee "${ENV_FILE}" >/dev/null <<EOF
# StarPU/FxT environment (installed under ${PREFIX})
export PATH="${PREFIX}/bin:\$PATH"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:\${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:\${LD_LIBRARY_PATH:-}"

# Enable FxT tracing by default
export STARPU_FXT_TRACE=1

# Write FxT traces under the user's HOME (recommended for per-node runs)
export STARPU_FXT_PREFIX="\$HOME/starpu_traces/fxt"
EOF

# Load env in *this* script run (so pkg-config works immediately)
# shellcheck disable=SC1090
source "${ENV_FILE}"

echo "==> [8/10] Ensuring /usr/local libs are visible to the dynamic linker..."
sudo tee /etc/ld.so.conf.d/starpu-local.conf >/dev/null <<EOF
${PREFIX}/lib
${PREFIX}/lib64
EOF
sudo ldconfig

echo "==> [9/10] Downloading and building StarPU ${STARPU_VER} (CPU-only, FxT, MPI)..."
rm -rf "starpu-${STARPU_VER}" "${STARPU_TARBALL}"
download_tarball "${STARPU_TARBALL}" "${STARPU_URL}"
tar -xzf "${STARPU_TARBALL}"
cd "starpu-${STARPU_VER}"

mkdir -p build
cd build

# Help find mpicc on distros that don't put it on PATH by default
if ! command -v mpicc >/dev/null 2>&1; then
  for p in /usr/lib64/openmpi/bin /usr/lib/x86_64-linux-gnu/openmpi/bin /usr/local/bin; do
    if [ -x "${p}/mpicc" ]; then
      export PATH="${p}:${PATH}"
      break
    fi
  done
fi

# Prefer MPI compiler wrappers when building StarPU with --enable-mpi
if command -v mpicc >/dev/null 2>&1; then
  export OMPI_CC=gcc
  export CC=mpicc
fi
if command -v mpicxx >/dev/null 2>&1; then
  export OMPI_CXX=g++
  export CXX=mpicxx
fi

../configure \
  --prefix="${PREFIX}" \
  --disable-cuda \
  --disable-opencl \
  --enable-fxt \
  --enable-mpi

make -j"${NPROC}"
sudo env PATH="$PATH" make install
sudo ldconfig

echo "==> [10/10] Verifying pkg-config sees StarPU and FxT env is set..."
echo "PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"

if pkg-config --exists starpu-1.4; then
  echo "OK: starpu-1.4 version: $(pkg-config --modversion starpu-1.4)"
else
  echo "ERROR: pkg-config still can't find starpu-1.4."
  echo "Try locating the .pc files:"
  echo "  sudo find ${PREFIX} -name 'starpu-1.4.pc' -o -name 'starpumpi-1.4.pc'"
  exit 1
fi

echo
echo "DONE."
echo "Open a new shell or run: source ${ENV_FILE}"
echo "FxT tracing is now enabled by default:"
echo "  STARPU_FXT_TRACE=${STARPU_FXT_TRACE}"
echo "  STARPU_FXT_PREFIX=${STARPU_FXT_PREFIX}"
echo "Traces will be written under: \$HOME/starpu_traces/fxt (on each node)."
echo "clang OpenMP check passed during setup. Use CC=clang-18 for OpenMP builds."
