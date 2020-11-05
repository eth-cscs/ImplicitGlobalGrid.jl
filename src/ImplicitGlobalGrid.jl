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

!!! note "Performance note"
    If the system supports CUDA-aware MPI, it may be activated for ImplicitGlobalGrid by setting the following environment variable (at latest before the first call to `init_global_grid`:
    ```shell
    shell> export IGG_CUDAAWARE_MPI=1
    ```
"""
module ImplicitGlobalGrid

## Include of shared constant parameters, types and syntax sugar
include("shared.jl")

## Alphabetical include of files
include("finalize_global_grid.jl")
include("gather.jl")
include("init_global_grid.jl")
include("select_device.jl")
include("tools.jl")
include("update_halo.jl")

end
