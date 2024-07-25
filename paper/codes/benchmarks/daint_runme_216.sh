#!/bin/bash -l
#SBATCH --job-name="diff3D_216"
#SBATCH --output=diff3D_216.%j.o
#SBATCH --error=diff3D_216.%j.e
#SBATCH --time=00:45:00
#SBATCH --nodes=216
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account c23

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
module load cray-hdf5-parallel

export JULIA_HDF5_PATH=$HDF5_ROOT
export JULIA_CUDA_MEMORY_POOL=none

export IGG_CUDAAWARE_MPI=1
export MPICH_RDMA_ENABLED_CUDA=1

scp diff3D.jl daint_submit_pareff.sh $SCRATCH/diff3D

pushd $SCRATCH/diff3D

chmod +x *.sh

srun daint_submit_pareff.sh

scp out_diff3D_pareff* $HOME/diff3D
