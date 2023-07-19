# Usage

## Installation
ImplicitGlobalGrid can be installed directly with the [Julia package manager] from the [Julia REPL]:
```julia-repl
julia>]
  pkg> add ImplicitGlobalGrid
```


## Multi-GPU with three functions
Only three functions are required to perform halo updates close to hardware limit:
- [`init_global_grid`](@ref)
- [`update_halo!`](@ref)
- [`finalize_global_grid`](@ref)

Three additional functions are provided to query Cartesian coordinates with respect to the global computational grid if required:
- [`x_g`](@ref)
- [`y_g`](@ref)
- [`z_g`](@ref)

Moreover, the following three functions allow to query the size of the global grid:
- [`nx_g`](@ref)
- [`ny_g`](@ref)
- [`nz_g`](@ref)

The [50-lines Multi-GPU example](@ref example-50-lines) in the section [Examples](@ref) illustrates how these functions enable the creation of massively parallel applications.


## [Straightforward in-situ visualization / monitoring](@id in-situ-usage)

ImplicitGlobalGrid provides a function to gather an array from each process into a one large array on a single process, assembled according to the global grid:
- [`gather!`](@ref)

This enables straightforward in-situ visualization or monitoring of Multi-GPU/CPU applications as shown in [this example](@ref in-situ-example).


## Seamless interoperability with [MPI.jl]
ImplicitGlobalGrid is seamlessly interoperable with [MPI.jl]. The Cartesian MPI communicator it uses is created by default when calling [`init_global_grid`](@ref) and can then be obtained as follows (variable `comm_cart`):
```julia
me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz);
```
Moreover, the automatic initialization and finalization of MPI can be deactivated in order to replace them with direct calls to [MPI.jl]:
```julia
init_global_grid(nx, ny, nz; init_MPI=false);
```
```julia
finalize_global_grid(;finalize_MPI=false)
```
Besides, [`init_global_grid`](@ref) makes every argument it passes to an [MPI.jl] function customizable via its keyword arguments.


## CUDA-/ROCm-aware MPI support
If the system supports CUDA-aware MPI (for Nvidia GPUs) or ROCm-aware MPI (for AMD GPUs), it can be activated for ImplicitGlobalGrid by setting one of the following environment variables (at latest before the call to [`init_global_grid`](@ref)):
```
shell> export IGG_CUDAAWARE_MPI=1
```
```
shell> export IGG_ROCMAWARE_MPI=1
```
