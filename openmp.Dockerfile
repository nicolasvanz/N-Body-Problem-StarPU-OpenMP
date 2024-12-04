FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

COPY ./src /src

WORKDIR /src/openmp