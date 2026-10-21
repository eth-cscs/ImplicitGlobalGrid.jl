# Usage

## [Multi-GPU with three functions](@id usage-three-functions)
Only three functions are required to perform halo updates close to hardware limit:
- [`init_global_grid`](@ref)
- [`update_halo!`](@ref)
- [`finalize_global_grid`](@ref)

The [50-lines Multi-GPU example](@ref example-50-lines) illustrates how these functions enable the creation of massively parallel applications.


## [Global coordinates, sizes, indices and extents](@id usage-grid-queries)
Three functions are provided to query Cartesian coordinates with respect to the global computational grid if required:
- [`x_g`](@ref)
- [`y_g`](@ref)
- [`z_g`](@ref)

Moreover, the following three functions allow to query the size of the global grid:
- [`nx_g`](@ref)
- [`ny_g`](@ref)
- [`nz_g`](@ref)

Further tools: global indices ([`ix_g`](@ref), [`iy_g`](@ref), [`iz_g`](@ref)), local and global extents of arrays, e.g. for parallel VTK output ([`extents`](@ref), [`extents_g`](@ref), [`metagrid`](@ref)) and timing ([`tic`](@ref), [`toc`](@ref)).


## [GPU support and device selection](@id usage-gpu)
Support for Nvidia or AMD GPUs is activated by importing [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) before ImplicitGlobalGrid:

:::code-group

```julia [CPUs]
using ImplicitGlobalGrid
```

```julia [Nvidia GPUs]
using CUDA
using ImplicitGlobalGrid
```

```julia [AMD GPUs]
using AMDGPU
using ImplicitGlobalGrid
```

:::

By default, [`init_global_grid`](@ref) selects the GPU corresponding to the node-local MPI rank (see [`select_device`](@ref)); every rank must then see all GPUs of its node. If the job launcher already binds one GPU per rank (e.g. Slurm `--gpus-per-task=1`), set `select_device=false`. To run on CPUs only while a GPU package is loaded, set `device_type="none"`.


## [CUDA-/ROCm-aware MPI support](@id usage-gpu-aware-mpi)
If the system supports CUDA-aware MPI (for Nvidia GPUs) or ROCm-aware MPI (for AMD GPUs), it can be activated for ImplicitGlobalGrid by setting the following environment variable (at latest before the call to [`init_global_grid`](@ref)):

:::code-group

```bash [Nvidia GPUs]
export IGG_CUDAAWARE_MPI=1
```

```bash [AMD GPUs]
export IGG_ROCMAWARE_MPI=1
```

:::

It can also be activated per dimension, see [Environment variables](@ref).


## [Multi-threaded CPU halo copies](@id usage-polyester)
Halo copies of CPU arrays can be multi-threaded with [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) by importing it before ImplicitGlobalGrid, setting `IGG_USE_POLYESTER=1` and starting Julia with multiple threads (e.g. `julia -t 4`):
```julia
using Polyester
using ImplicitGlobalGrid
```


## [CellArrays](@id usage-cellarrays)
[`update_halo!`](@ref) also accepts `CellArray`s of [CellArrays.jl](https://github.com/omlins/CellArrays.jl) (with blocklength `0` or `1`).


## [Multiple grids](@id usage-multiple-grids)
[`init_global_grid`](@ref) creates and activates one grid. Additional grids are created with [`create_global_grid`](@ref) and selected with [`activate_global_grid`](@ref) (or passed to [`update_halo!`](@ref) with the keyword `active_global_grid`); [`get_global_grid`](@ref) returns the active grid:
```julia
init_global_grid()                          # Initialize the package without creating a grid.
gg1 = create_global_grid(nx, ny, nz)
gg2 = create_global_grid(2nx, 2ny, 2nz)
activate_global_grid(gg1)
update_halo!(A1)                            # Uses gg1.
update_halo!(A2; active_global_grid=gg2)    # Uses gg2.
```


## [In-situ visualization / monitoring](@id in-situ-usage)
ImplicitGlobalGrid provides a function to gather an array from each process into a one large array on a single process, assembled according to the global grid:
- [`gather!`](@ref)

This enables straightforward in-situ visualization or monitoring of Multi-GPU/CPU applications as shown in [this example](@ref in-situ-example).


## [Seamless interoperability with MPI.jl](@id usage-mpi)
ImplicitGlobalGrid is seamlessly interoperable with [MPI.jl](https://github.com/JuliaParallel/MPI.jl). The Cartesian MPI communicator it uses is created by default when calling [`init_global_grid`](@ref) and can then be obtained as follows (variable `comm_cart`):
```julia
me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz);
```
Moreover, the automatic initialization and finalization of MPI can be deactivated in order to replace them with direct calls to [MPI.jl](https://github.com/JuliaParallel/MPI.jl):
```julia
init_global_grid(nx, ny, nz; init_MPI=false);
```
```julia
finalize_global_grid(;finalize_MPI=false)
```
Besides, [`init_global_grid`](@ref) makes every argument it passes to an [MPI.jl](https://github.com/JuliaParallel/MPI.jl) function customizable via its keyword arguments.


## Environment variables
The following environment variables are read when a grid is created ([`init_global_grid`](@ref) or [`create_global_grid`](@ref)); a value `> 0` activates the feature:

| Variable                                                                      | Description                                                                                           |
|:----------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------- |
| `IGG_CUDAAWARE_MPI`                                                           | Use CUDA-aware MPI in all dimensions.                                                                 |
| `IGG_CUDAAWARE_MPI_DIMX`, `IGG_CUDAAWARE_MPI_DIMY`, `IGG_CUDAAWARE_MPI_DIMZ`  | Use CUDA-aware MPI in the given dimension (only read if `IGG_CUDAAWARE_MPI` is not activated).        |
| `IGG_ROCMAWARE_MPI`                                                           | Use ROCm-aware MPI in all dimensions.                                                                 |
| `IGG_ROCMAWARE_MPI_DIMX`, `IGG_ROCMAWARE_MPI_DIMY`, `IGG_ROCMAWARE_MPI_DIMZ`  | Use ROCm-aware MPI in the given dimension (only read if `IGG_ROCMAWARE_MPI` is not activated).        |
| `IGG_USE_POLYESTER`                                                           | Use Polyester for the CPU halo copies in all dimensions (requires importing Polyester).               |
| `IGG_USE_POLYESTER_DIMX`, `IGG_USE_POLYESTER_DIMY`, `IGG_USE_POLYESTER_DIMZ`  | Override `IGG_USE_POLYESTER` in the given dimension (only read if `IGG_USE_POLYESTER` is activated).  |
