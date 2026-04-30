export create_global_grid

"""
    gg = create_global_grid(nx, ny, nz; <keyword arguments>)

Construct and return a Cartesian grid configuration of MPI processes, defining implicitly a global grid. This grid configuration can then be activated by `activate_global_grid`.

# Arguments
- {`nx`|`ny`|`nz`}`::Integer`: the number of elements of the local grid in dimension {x|y|z}.
- {`dimx`|`dimy`|`dimz`}`::Integer=0`: the desired number of processes in dimension {x|y|z}. By default, (value `0`) the process topology is created as compact as possible with the given constraints. This is handled by the MPI implementation which is installed on your system. For more information, refer to the specifications of `MPI_Dims_create` in the corresponding documentation.
- {`periodx`|`periody`|`periodz`}`::Union{Bool,Integer}=0`: whether the grid is periodic (`true` or `1`) or not (`false` or `0`) in dimension {x|y|z}.
- `origin::Tuple|AbstractFloat`: the origin of the global grid. By default, it is set to `(0.0, 0.0, 0.0)` for 3D, `(0.0, 0.0)` for 2D and `0.0` for 1D.
- `origin_on_vertex::Bool=false`: whether the origin is on the cell vertex; else, it is on the cell center (default). The default implies that the space step `dx` is computed in the user code as `dx=lx/(nx-1)`, where `lx` is the length of the global grid in dimension x. Setting the origin on the vertex implies that the space step is computed as `dx=lx/nx`, instead. The analog applies for the dimensions y and z.
- {`centerx`|`centery`|`centerz`}`::Bool=false`: whether to center the grid on the origin (`true`) or not (`false`) in dimension {x|y|z}. By default, the grid is extends from `origin` in the positive direction of the corresponding dimension.
- `quiet::Bool=false`: whether to suppress printing information like the size of the global grid (`true`) or not (`false`).
!!! note "Advanced keyword arguments"
    - `overlaps::Tuple{Int,Int,Int}=(2,2,2)`: the number of elements adjacent local grids overlap in dimension x, y and z. By default (value `(2,2,2)`), an array `A` of size (`nx`, `ny`, `nz`) on process 1 (`A_1`) overlaps the corresponding array `A` on process 2 (`A_2`) by `2` indices if the two processes are adjacent. E.g., if `overlaps[1]=2` and process 2 is the right neighbor of process 1 in dimension x, then `A_1[end-1:end,:,:]` overlaps `A_2[1:2,:,:]`. That means, after every call `update_halo!(A)`, we have `all(A_1[end-1:end,:,:] .== A_2[1:2,:,:])` (`A_1[end,:,:]` is the halo of process 1 and `A_2[1,:,:]` is the halo of process 2). The analog applies for the dimensions y and z.
    - `halowidths::Tuple{Int,Int,Int}=max.(1,overlaps.÷2)`: the default width of an array's halo in dimension x, y and z (must be greater or equal to 1). The default can be overwritten per array in the function [`update_halo`](@ref).
    - `disp::Integer=1`:  the displacement argument to `MPI.Cart_shift` in order to determine the neighbors.
    - `reorder::Integer=1`: the reorder argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    - `comm::MPI.Comm=MPI.COMM_WORLD`: the input communicator argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    For more information, refer to the documentation of MPI.jl / MPI. `select_device` and `device_type` cannot be changed during grid creation; they must be set during initialization.

# Return values
- `gg`: the newly defined global grid

# Usage example
    init_global_grid()                  # Basic call no dimensions are provided thus no grid is defined.
    gg = create_global_grid(nx, ny, nz) # A global grid specification is created
    activate_global_grid(gg)            # The new global grid specification is active
    update_halo!(array)                 # The halo exchange is made following the active global grid specification

See also: [`finalize_global_grid`](@ref), [`init_global_grid`](@ref), [`activate_global_grid`](@ref)
"""
function create_global_grid(nx::Integer, ny::Integer=1, nz::Integer=1;
    dimx::Integer                      = default(:dimx),
    dimy::Integer                      = default(:dimy),
    dimz::Integer                      = default(:dimz),
    periodx::Union{Bool,Integer}       = default(:periodx),
    periody::Union{Bool,Integer}       = default(:periody),
    periodz::Union{Bool,Integer}       = default(:periodz),
    origin::Union{Tuple,AbstractFloat} = default(:origin),
    origin_on_vertex::Bool             = default(:origin_on_vertex),
    centerx::Bool                      = default(:centerx),
    centery::Bool                      = default(:centery),
    centerz::Bool                      = default(:centerz),
    overlaps::Tuple{Int,Int,Int}       = default(:overlaps),
    halowidths::Tuple{Int,Int,Int}     = default(:halowidths),
    disp                               = default(:disp),
    reorder                            = default(:reorder),
    comm                               = default(:comm),
    quiet                              = default(:quiet),)
    
    check_initialized()
    # These have been moved out of the argument list since they are not expected to change in different grids and the default is updated by the preceding call to init_global_grid if the user set the corresponding keywords.
    device_type     = default(:device_type)
    select_device   = default(:select_device)
    nx, ny, nz, dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet = normalize_input(nx,ny,nz, dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
    nxyz            = [nx, ny, nz]
    dims            = [dimx, dimy, dimz]
    periods         = [periodx, periody, periodz]
    centerxyz       = [centerx, centery, centerz]
    origin          = [o for o in origin]
    overlaps        = [overlaps...]
    halowidths      = [halowidths...]
    cuda_enabled    = false
    amdgpu_enabled  = false
    cudaaware_MPI   = [false, false, false]
    amdgpuaware_MPI = [false, false, false]
    use_polyester   = [false, false, false]
    if haskey(ENV, "IGG_CUDAAWARE_MPI") cudaaware_MPI   .= (parse(Int64, ENV["IGG_CUDAAWARE_MPI"]) > 0); end
    if haskey(ENV, "IGG_ROCMAWARE_MPI") amdgpuaware_MPI .= (parse(Int64, ENV["IGG_ROCMAWARE_MPI"]) > 0); end
    if haskey(ENV, "IGG_USE_POLYESTER") use_polyester   .= (parse(Int64, ENV["IGG_USE_POLYESTER"]) > 0); end
    if none(cudaaware_MPI)
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMX") cudaaware_MPI[1] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMX"]) > 0); end
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMY") cudaaware_MPI[2] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMY"]) > 0); end
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMZ") cudaaware_MPI[3] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMZ"]) > 0); end
    end
    if none(amdgpuaware_MPI)
        if haskey(ENV, "IGG_ROCMAWARE_MPI_DIMX") amdgpuaware_MPI[1] = (parse(Int64, ENV["IGG_ROCMAWARE_MPI_DIMX"]) > 0); end
        if haskey(ENV, "IGG_ROCMAWARE_MPI_DIMY") amdgpuaware_MPI[2] = (parse(Int64, ENV["IGG_ROCMAWARE_MPI_DIMY"]) > 0); end
        if haskey(ENV, "IGG_ROCMAWARE_MPI_DIMZ") amdgpuaware_MPI[3] = (parse(Int64, ENV["IGG_ROCMAWARE_MPI_DIMZ"]) > 0); end
    end
    if all(use_polyester)
        if haskey(ENV, "IGG_USE_POLYESTER_DIMX") use_polyester[1] = (parse(Int64, ENV["IGG_USE_POLYESTER_DIMX"]) > 0); end
        if haskey(ENV, "IGG_USE_POLYESTER_DIMY") use_polyester[2] = (parse(Int64, ENV["IGG_USE_POLYESTER_DIMY"]) > 0); end
        if haskey(ENV, "IGG_USE_POLYESTER_DIMZ") use_polyester[3] = (parse(Int64, ENV["IGG_USE_POLYESTER_DIMZ"]) > 0); end
    end
    if (device_type != DEVICE_TYPE_NONE)
        if (device_type in [DEVICE_TYPE_CUDA,   DEVICE_TYPE_AUTO]) cuda_enabled   = cuda_loaded() && cuda_functional()  end # NOTE: cuda could be enabled/disabled depending on some additional criteria.
        if (device_type in [DEVICE_TYPE_AMDGPU, DEVICE_TYPE_AUTO]) amdgpu_enabled = amdgpu_loaded() && amdgpu_functional() end # NOTE: amdgpu could be enabled/disabled depending on some additional criteria.
    end
    nprocs         = MPI.Comm_size(comm)
    dims          .= MPI.Dims_create!(nprocs, dims)
    comm_cart      = MPI.Cart_create(comm, dims, periods, reorder)
    me             = MPI.Comm_rank(comm_cart)
    coords         = MPI.Cart_coords(comm_cart)
    neighbors      = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI)
    for i = 1:NDIMS_MPI
        neighbors[:, i] .= MPI.Cart_shift(comm_cart, i - 1, disp)
    end
    nxyz_g                = dims .* (nxyz .- overlaps) .+ overlaps .* (periods .== 0) # E.g. for dimension x with ol=2 and periodx=0: dimx*(nx-2)+2
    gg                    = GlobalGrid(nxyz_g, nxyz, dims, overlaps, halowidths, origin, origin_on_vertex, centerxyz, nprocs, me, coords, neighbors, periods, disp, reorder, comm_cart, cuda_enabled, amdgpu_enabled, cudaaware_MPI, amdgpuaware_MPI, use_polyester, quiet)
    cuda_support_string   = (cuda_enabled && all(cudaaware_MPI)) ? "CUDA-aware" : (cuda_enabled && any(cudaaware_MPI)) ? "CUDA(-aware)" : (cuda_enabled) ? "CUDA" : ""
    amdgpu_support_string = (amdgpu_enabled && all(amdgpuaware_MPI)) ? "AMDGPU-aware" : (amdgpu_enabled && any(amdgpuaware_MPI)) ? "AMDGPU(-aware)" : (amdgpu_enabled) ? "AMDGPU" : ""
    gpu_support_string    = join(filter(!isempty, [cuda_support_string, amdgpu_support_string]), ", ")
    support_string        = isempty(gpu_support_string) ? "none" : gpu_support_string
    if (!quiet && me == 0) println("Global grid: $(nxyz_g[1])x$(nxyz_g[2])x$(nxyz_g[3]) (nprocs: $nprocs, dims: $(dims[1])x$(dims[2])x$(dims[3]); device support: $support_string)"); end
    if ((cuda_enabled || amdgpu_enabled) && select_device) _select_device(gg); end
    return gg
end
