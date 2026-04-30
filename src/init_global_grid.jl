export init_global_grid

function init_package()
    check_not_initialized();
    # Set bools
    set_cuda_loaded()
    set_cuda_functional()
    set_amdgpu_loaded()
    set_amdgpu_functional()
    if haskey(ENV, "IGG_LOOPVECTORIZATION") error("Environment variable IGG_LOOPVECTORIZATION is not supported anymore. Use IGG_USE_POLYESTER instead."); end
end

function init_global_grid(;save_kwarg_defaults::Bool=false, device_type::String=DEVICE_TYPE_AUTO, select_device::Bool=true, dimx::Integer=0, dimy::Integer=0, dimz::Integer=0, periodx::Union{Bool,Integer}=0, periody::Union{Bool,Integer}=0, periodz::Union{Bool,Integer}=0, origin::Union{Tuple,AbstractFloat}=(0.0, 0.0, 0.0), origin_on_vertex::Bool=false, centerx::Bool=false, centery::Bool=false, centerz::Bool=false, overlaps::Tuple{Int,Int,Int}=(2, 2, 2), halowidths::Tuple{Int,Int,Int}=max.(1, overlaps .÷ 2), disp::Integer=1, reorder::Integer=1, comm::MPI.Comm=MPI.COMM_WORLD, init_MPI::Bool=true, quiet::Bool=false)
    init_package()
    # Validate and normalize, in some situations even defaults can be invalid so it has to be always checked
    dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet  = normalize_input(dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
    # Set the device type to be used for any possible grid, (!) keep in mind (!) bypasses the set default flag 
    set_default_args(device_type=device_type, select_device=select_device)
    # Check no default updates are passed if save_kwarg_defaults is false, device checks are exempt
    if differ_default_args(dimx=dimx, dimy=dimy, dimz=dimz, periodx=periodx, periody=periody, periodz=periodz, origin=origin, origin_on_vertex=origin_on_vertex, centerx=centerx, centery=centery, centerz=centerz, overlaps=overlaps, halowidths=halowidths, disp=disp, reorder=reorder, comm=comm, quiet=quiet)
        if save_kwarg_defaults
            set_default_args(dimx=dimx, dimy=dimy, dimz=dimz, periodx=periodx, periody=periody, periodz=periodz, origin=origin, origin_on_vertex=origin_on_vertex, centerx=centerx, centery=centery, centerz=centerz, overlaps=overlaps, halowidths=halowidths, disp=disp, reorder=reorder, comm=comm, quiet=quiet)
        else
            error("Different defaults for grid arguments have been passed with save_kwarg_defaults=false, set to true to change default grid creation parameters")
        end
    end
    if (init_MPI)  # NOTE: init MPI only, once the input arguments have been checked.
        if (MPI.Initialized()) error("MPI is already initialized. Set the argument 'init_MPI=false'."); end
        MPI.Init()
    else
        if (!MPI.Initialized()) error("MPI has not been initialized beforehand. Remove the argument 'init_MPI=false'."); end  # Ensure that MPI is always initialized after init_global_grid().
    end
    set_initialized();
    return nothing;
end

"""
    init_global_grid(nx, ny, nz)
    init_global_grid()
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; <keyword arguments>)
    me, dims, nprocs, coords, comm_cart = init_global_grid(<keyword arguments>)

Initialize the package and create a Cartesian grid of MPI processes (and also MPI itself by default) defining implicitly a global grid. 

For use cases needing multiple grid arrangements, the dispatch without (`nx`|`ny`|`nz`) can be used to do a package initialization without creating a grid, several grids can then be created by the create_global_grid function and activated via activate_global_grid.

# Arguments
- {`nx`|`ny`|`nz`}`::Integer`: the number of elements of the local grid in dimension {x|y|z}.
- {`dimx`|`dimy`|`dimz`}`::Integer=0`: the desired number of processes in dimension {x|y|z}. By default, (value `0`) the process topology is created as compact as possible with the given constraints. This is handled by the MPI implementation which is installed on your system. For more information, refer to the specifications of `MPI_Dims_create` in the corresponding documentation.
- {`periodx`|`periody`|`periodz`}`::Bool|Integer=false`: whether the grid is periodic (`true`) or not (`false`) in dimension {x|y|z}. The argument is also accepted as an integer as traditionally in the MPI standard (`1` for `true` and `0` for `false`).
- `origin::Tuple|AbstractFloat`: the origin of the global grid. By default, it is set to `(0.0, 0.0, 0.0)` for 3D, `(0.0, 0.0)` for 2D and `0.0` for 1D.
- `origin_on_vertex::Bool=false`: whether the origin is on the cell vertex; else, it is on the cell center (default). The default implies that the space step `dx` is computed in the user code as `dx=lx/(nx-1)`, where `lx` is the length of the global grid in dimension x. Setting the origin on the vertex implies that the space step is computed as `dx=lx/nx`, instead. The analog applies for the dimensions y and z.
- {`centerx`|`centery`|`centerz`}`::Bool=false`: whether to center the grid on the origin (`true`) or not (`false`) in dimension {x|y|z}. By default, the grid is extends from `origin` in the positive direction of the corresponding dimension.
- `quiet::Bool=false`: whether to suppress printing information like the size of the global grid (`true`) or not (`false`).
- `save_kwarg_defaults::Bool=false`: whether to update the default values of the keyword arguments of `create_global_grid` with the ones passed to `init_global_grid`. If `false` and no positional arguments are given (`nx`, `ny`, `nz`) and keyword arguments are passed with values different from the current defaults, an error is thrown. The parameters `select_device` and `device_type` are not affected by this argument, they can only and always be set on initialization.
!!! note "Advanced keyword arguments"
    - `overlaps::Tuple{Int,Int,Int}=(2,2,2)`: the number of elements adjacent local grids overlap in dimension x, y and z. By default (value `(2,2,2)`), an array `A` of size (`nx`, `ny`, `nz`) on process 1 (`A_1`) overlaps the corresponding array `A` on process 2 (`A_2`) by `2` indices if the two processes are adjacent. E.g., if `overlaps[1]=2` and process 2 is the right neighbor of process 1 in dimension x, then `A_1[end-1:end,:,:]` overlaps `A_2[1:2,:,:]`. That means, after every call `update_halo!(A)`, we have `all(A_1[end-1:end,:,:] .== A_2[1:2,:,:])` (`A_1[end,:,:]` is the halo of process 1 and `A_2[1,:,:]` is the halo of process 2). The analog applies for the dimensions y and z.
    - `halowidths::Tuple{Int,Int,Int}=max.(1,overlaps.÷2)`: the default width of an array's halo in dimension x, y and z (must be greater or equal to 1). The default can be overwritten per array in the function [`update_halo`](@ref).
    - `disp::Integer=1`:  the displacement argument to `MPI.Cart_shift` in order to determine the neighbors.
    - `reorder::Integer=1`: the reorder argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    - `comm::MPI.Comm=MPI.COMM_WORLD`: the input communicator argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    - `init_MPI::Bool=true`: whether to initialize MPI (`true`) or not (`false`).
    - `device_type::String="auto"`: the type of the device to be used if available: `"CUDA"`, `"AMDGPU"`, `"none"` or `"auto"`. Set `device_type="none"` if you want to use only CPUs on a system having also GPUs. If `device_type` is `"auto"` (default), it is automatically determined, depending on which of the modules used for programming the devices (CUDA.jl or AMDGPU.jl) was imported before ImplicitGlobalGrid; if both were imported, an error will be given if `device_type` is set as `"auto"`. This can only be set up on initialization.
    - `select_device::Bool=true`: whether to automatically select the device (GPU) (`true`) or not (`false`) if CUDA or AMDGPU was imported and `device_type` is not `"none"`. If `true`, it selects the device corresponding to the node-local MPI rank. This method of device selection suits both single and multi-device compute nodes and is recommended in general. It is also the default method of device selection of the *function* [`select_device`](@ref). This can only be set up on initialization.
    For more information, refer to the documentation of MPI.jl / MPI.

# Return values
- `me`: the MPI rank of the process.
- `dims`: the number of processes in each dimension.
- `nprocs`: the number of processes.
- `coords`: the Cartesian coordinates of the process.
- `comm_cart`: the MPI communicator of the created Cartesian process topology.

# Typical use cases
    init_global_grid(nx, ny, nz)                  # Basic call (no optional in and output arguments).
    me, = init_global_grid(nx, ny, nz)            # Capture 'me' (note the ','!).
    me, dims = init_global_grid(nx, ny, nz)       # Capture 'me' and 'dims'.
    init_global_grid(nx, ny, nz; dimx=2, dimy=2)  # Fix the number of processes in the dimensions x and y of the Cartesian grid of MPI processes to 2 (the number of processes can vary only in the dimension z).
    init_global_grid(nx, ny, nz; periodz=1)       # Make the boundaries in dimension z periodic.

See also: [`finalize_global_grid`](@ref), [`select_device`](@ref)
"""
function init_global_grid(nx::Integer, ny::Integer=1, nz::Integer=1; dimx::Integer=0, dimy::Integer=0, dimz::Integer=0, periodx::Union{Bool,Integer}=0, periody::Union{Bool,Integer}=0, periodz::Union{Bool,Integer}=0, origin::Union{Tuple,AbstractFloat}=(0.0, 0.0, 0.0), origin_on_vertex::Bool=false, centerx::Bool=false, centery::Bool=false, centerz::Bool=false, overlaps::Tuple{Int,Int,Int}=(2, 2, 2), halowidths::Tuple{Int,Int,Int}=max.(1, overlaps .÷ 2), disp::Integer=1, reorder::Integer=1, comm::MPI.Comm=MPI.COMM_WORLD, init_MPI::Bool=true, device_type::String=DEVICE_TYPE_AUTO, select_device::Bool=true, quiet::Bool=false, save_kwarg_defaults::Bool=false)
    init_package()
    gg = nothing
    try
        set_initialized()
        nx, ny, nz, dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet = normalize_input(nx,ny,nz, dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
        # Set the device type to be used for any possible grid, (!) keep in mind (!) bypasses the set default flag 
        set_default_args(device_type=device_type, select_device=select_device)
        if save_kwarg_defaults
            set_default_args(dimx=dimx, dimy=dimy, dimz=dimz, periodx=periodx, periody=periody, periodz=periodz, origin=origin, origin_on_vertex=origin_on_vertex, centerx=centerx, centery=centery, centerz=centerz, overlaps=overlaps, halowidths=halowidths, disp=disp, reorder=reorder, comm=comm, quiet=quiet)
        end
        if (init_MPI)  # NOTE: init MPI only, once the input arguments have been checked.
            if (MPI.Initialized()) error("MPI is already initialized. Set the argument 'init_MPI=false'."); end
            MPI.Init()
        else
            if (!MPI.Initialized()) error("MPI has not been initialized beforehand. Remove the argument 'init_MPI=false'."); end  # Ensure that MPI is always initialized after init_global_grid().
        end
        gg = create_global_grid(nx, ny, nz,
            dimx=dimx, dimy=dimy, dimz=dimz,
            periodx=periodx, periody=periody, periodz=periodz,
            origin=origin, origin_on_vertex=origin_on_vertex,
            centerx=centerx, centery=centery, centerz=centerz,
            overlaps=overlaps, halowidths=halowidths, disp=disp,
            reorder=reorder, comm=comm, quiet=quiet)
        activate_global_grid(gg)
    catch 
        set_initialized(false)
        rethrow()  
    end
    return gg.me, gg.dims, gg.nprocs, gg.coords, gg.comm # The typical use case requires only these variables; the remaining can be obtained calling active_global_grid() if needed.
end

# Make sure that timing functions which must be fast at the first user call are already compiled now.
function init_timing_functions()
    tic();
    toc();
end
