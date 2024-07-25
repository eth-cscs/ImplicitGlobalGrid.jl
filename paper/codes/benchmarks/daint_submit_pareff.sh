#!/bin/bash

RUN="diff3D"

for ie in {1..20}; do

    # RDMA CUDA-aware
    LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project --check-bounds=no -O3  "$RUN".jl

    # no RDMA
    # julia --project --check-bounds=no -O3  "$RUN".jl

done
