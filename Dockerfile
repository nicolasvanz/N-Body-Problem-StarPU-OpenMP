FROM registry.gitlab.inria.fr/starpu/starpu-docker/starpu:1.4.7

# starpu container does not use root as default user
USER root
RUN sudo adduser starpu sudo
USER starpu

# cuda device is not recognized for some reason. The steps reinstall hwloc

RUN sudo apt-get update && sudo apt-get install -y \
    build-essential \
    wget \
    libpciaccess-dev \
    libxml2-dev \
    libnuma-dev 

ENV HWLOC_VERSION=2.9.0

RUN sudo wget https://download.open-mpi.org/release/hwloc/v2.9/hwloc-${HWLOC_VERSION}.tar.gz && \
    sudo tar -xvzf hwloc-${HWLOC_VERSION}.tar.gz && \
    cd hwloc-${HWLOC_VERSION} && \
    ./configure --enable-cuda && \
    sudo make -j$(nproc) && \
    sudo make install && \
    sudo ldconfig && \
    cd .. && sudo rm -rf hwloc-${HWLOC_VERSION}*

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /home

WORKDIR /home/src/starpu