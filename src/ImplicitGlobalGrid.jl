"""
Module ImplicitGlobalGrid

Renders the distributed parallelization of stencil-based GPU and CPU applications on a regular staggered grid almost trivial and enables close to ideal weak scaling of real-world applications on thousands of GPUs.

# General overview and examples
https://github.com/eth-cscs/ImplicitGlobalGrid.jl

# Functions
- [`init_global_grid`](@ref)
- [`finalize_global_grid`](@ref)
- [`update_halo!`](@ref)
- [`gather!`](@ref)
- [`select_device`](@ref)
- [`nx_g`](@ref)
- [`ny_g`](@ref)
- [`nz_g`](@ref)
- [`x_g`](@ref)
- [`y_g`](@ref)
- [`z_g`](@ref)
- [`tic`](@ref)
- [`toc`](@ref)

To see a description of a function type `?<functionname>`.

!!! note "Activation of device support"
    The support for a device type (CUDA or AMDGPU) is activated by importing the corresponding module (CUDA or AMDGPU) before importing ImplicitGlobalGrid (the corresponding extension will be loaded).

!!! note "Performance note"
    If the system supports CUDA-aware MPI (for Nvidia GPUs) or ROCm-aware MPI (for AMD GPUs), it may be activated for ImplicitGlobalGrid by setting one of the following environment variables (at latest before the call to `init_global_grid`):
    ```shell
    shell> export IGG_CUDAAWARE_MPI=1
    ```
    ```shell
    shell> export IGG_ROCMAWARE_MPI=1
    ```
"""
module ImplicitGlobalGrid

## Include of exception module
include("Exceptions.jl");
using .Exceptions

## Include of shared constant parameters, types and syntax sugar
include("shared.jl")

## Alphabetical include of defaults for extensions
include("defaults_shared.jl")
include(joinpath("AMDGPUExt", "defaults.jl"))
include(joinpath("CUDAExt", "defaults.jl"))
include(joinpath("LoopVectorizationExt", "memcopy_LV_default.jl"))

## Alphabetical include of files
include("finalize_global_grid.jl")
include("gather.jl")
include("init_global_grid.jl")
include("select_device.jl")
include("tools.jl")
include("update_halo.jl")

end
