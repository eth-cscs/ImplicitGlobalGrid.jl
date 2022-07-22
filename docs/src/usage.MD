# Usage

## Multi-GPU with three functions
Only three functions are required to perform halo updates close to hardware limit:
- `init_global_grid`
- `update_halo!`
- `finalize_global_grid`

Three additional functions are provided to query Cartesian coordinates with respect to the global computational grid if required:
- `x_g`
- `y_g`
- `z_g`

Moreover, the following three functions allow to query the size of the global grid:
- `nx_g`
- `ny_g`
- `nz_g`

The [50-lines Multi-GPU example](@ref) illustrates how these functions enable the creation of massively parallel applications.


## Seamless interoperability with MPI.jl
ImplicitGlobalGrid is seamlessly interoperable with [MPI.jl]. The Cartesian MPI communicator it uses is created by default when calling `init_global_grid` and can then be obtained as follows (variable `comm_cart`):
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
Besides, `init_global_grid` makes every argument it passes to an [MPI.jl] function customizable via its keyword arguments.


## CUDA-aware/ROCm-aware MPI support
If the system supports CUDA-aware MPI (for Nvidia GPUs) or ROCm-aware MPI (for AMD GPUs), it can be activated for ImplicitGlobalGrid by setting one of the following environment variables (at latest before the call to `init_global_grid`):
```
shell> export IGG_CUDAAWARE_MPI=1
```
```
shell> export IGG_ROCMAWARE_MPI=1
```

## Installation
ImplicitGlobalGrid can be installed directly with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html) from the REPL:
```julia-repl
julia>]
  pkg> add ImplicitGlobalGrid
  pkg> test ImplicitGlobalGrid
```
