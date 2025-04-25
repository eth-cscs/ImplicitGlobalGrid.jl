export init_global_grid

"""
    init_global_grid(nx)
    init_global_grid(nx, ny)
    init_global_grid(nx, ny, nz)
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; <keyword arguments>)

Initialize a Cartesian grid of MPI processes (and also MPI itself by default) defining implicitely a global grid.

# Arguments
- {`nx`|`ny`|`nz`}`::Integer`: the number of elements of the local grid in dimension {x|y|z}. For 2D problems, setting `nz` to 1 is equivalent to ommitting the argument; for 1D problems, setting `ny` and `nz` to 1 is equivalent to ommitting the arguments.

# Keyword arguments
- {`dimx`|`dimy`|`dimz`}`::Integer=0`: the desired number of processes in dimension {x|y|z}. By default, (value `0`) the process topology is created as compact as possible with the given constraints. This is handled by the MPI implementation which is installed on your system. For more information, refer to the specifications of `MPI_Dims_create` in the corresponding documentation.
- {`periodx`|`periody`|`periodz`}`::Bool|Integer=false`: whether the grid is periodic (`true`) or not (`false`) in dimension {x|y|z}. The argument is also accepted as an integer as traditionally in the MPI standard (`1` for `true` and `0` for `false`).
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
    - `init_MPI::Bool=true`: whether to initialize MPI (`true`) or not (`false`).
    - `device_type::String="auto"`: the type of the device to be used if available: `"CUDA"`, `"AMDGPU"`, `"none"` or `"auto"`. Set `device_type="none"` if you want to use only CPUs on a system having also GPUs. If `device_type` is `"auto"` (default), it is automatically determined, depending on which of the modules used for programming the devices (CUDA.jl or AMDGPU.jl) was imported before ImplicitGlobalGrid; if both were imported, an error will be given if `device_type` is set as `"auto"`.
    - `select_device::Bool=true`: whether to automatically select the device (GPU) (`true`) or not (`false`) if CUDA or AMDGPU was imported and `device_type` is not `"none"`. If `true`, it selects the device corresponding to the node-local MPI rank. This method of device selection suits both single and multi-device compute nodes and is recommended in general. It is also the default method of device selection of the *function* [`select_device`](@ref).
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
function init_global_grid(nx::Integer, ny::Integer=1, nz::Integer=1; dimx::Integer=0, dimy::Integer=0, dimz::Integer=0, periodx::Union{Bool,Integer}=0, periody::Union{Bool,Integer}=0, periodz::Union{Bool,Integer}=0, origin::Union{Tuple,AbstractFloat}, origin_on_vertex::Bool=false, centerx::Bool=false, centery::Bool=false, centerz::Bool=false, overlaps::Tuple{Int,Int,Int}=(2,2,2), halowidths::Tuple{Int,Int,Int}=max.(1,overlaps.÷2), disp::Integer=1, reorder::Integer=1, comm::MPI.Comm=MPI.COMM_WORLD, init_MPI::Bool=true, device_type::String=DEVICE_TYPE_AUTO, select_device::Bool=true, quiet::Bool=false)
    if grid_is_initialized() error("The global grid has already been initialized.") end
    set_cuda_loaded()
    set_cuda_functional()
    set_amdgpu_loaded()
    set_amdgpu_functional()
    nxyz              = [nx, ny, nz];
    dims              = [dimx, dimy, dimz];
    periods           = Int64.([periodx, periody, periodz]);
    origin            = Float64.([((length((origin...,)) == 1) ?  (origin, 0, 0) : ((length(origin) == 2) ? (origin..., 0) : origin))...]);
    centerxyz         = [centerx, centery, centerz];
    overlaps          = [overlaps...];
    halowidths        = [halowidths...];
    cuda_enabled      = false
    amdgpu_enabled    = false
    cudaaware_MPI     = [false, false, false]
    amdgpuaware_MPI   = [false, false, false]
    use_polyester     = [false, false, false]
    if haskey(ENV, "IGG_LOOPVECTORIZATION") error("Environment variable IGG_LOOPVECTORIZATION is not supported anymore. Use IGG_USE_POLYESTER instead.") end
    if haskey(ENV, "IGG_CUDAAWARE_MPI") cudaaware_MPI .= (parse(Int64, ENV["IGG_CUDAAWARE_MPI"]) > 0); end
    if haskey(ENV, "IGG_ROCMAWARE_MPI") amdgpuaware_MPI .= (parse(Int64, ENV["IGG_ROCMAWARE_MPI"]) > 0); end
    if haskey(ENV, "IGG_USE_POLYESTER") use_polyester .= (parse(Int64, ENV["IGG_USE_POLYESTER"]) > 0); end
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
    if !(device_type in [DEVICE_TYPE_NONE, DEVICE_TYPE_AUTO, DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU]) error("Argument `device_type`: invalid value obtained ($device_type). Valid values are: $DEVICE_TYPE_CUDA, $DEVICE_TYPE_AMDGPU, $DEVICE_TYPE_NONE, $DEVICE_TYPE_AUTO") end
    if ((device_type == DEVICE_TYPE_AUTO) && cuda_loaded() && cuda_functional() && amdgpu_loaded() && amdgpu_functional()) error("Automatic detection of the device type to be used not possible: both CUDA and AMDGPU extensions are loaded and functional. Set keyword argument `device_type` to $DEVICE_TYPE_CUDA or $DEVICE_TYPE_AMDGPU.") end
    if (device_type != DEVICE_TYPE_NONE)
        if (device_type in [DEVICE_TYPE_CUDA,   DEVICE_TYPE_AUTO]) cuda_enabled   = cuda_loaded() && cuda_functional()  end # NOTE: cuda could be enabled/disabled depending on some additional criteria.
        if (device_type in [DEVICE_TYPE_AMDGPU, DEVICE_TYPE_AUTO]) amdgpu_enabled = amdgpu_loaded() && amdgpu_functional() end # NOTE: amdgpu could be enabled/disabled depending on some additional criteria.
    end
    if (any(nxyz .< 1)) error("Invalid arguments: nx, ny, and nz cannot be less than 1."); end
    if (any(dims .< 0)) error("Invalid arguments: dimx, dimy, and dimz cannot be negative."); end
    if (any(periods .∉ ((0,1),))) error("Invalid arguments: periodx, periody, and periodz must be either 0 or 1."); end
    if length(origin) != 3 error("Invalid argument: the length of the origin tuple must be at most 3.") end
    if (centerx && origin_on_vertex && isodd(nx)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and nx being odd; set either `origin_on_vertex=false` or make nx even."); end
    if (centery && origin_on_vertex && isodd(ny)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and ny being odd; set either `origin_on_vertex=false` or make ny even."); end
    if (centerz && origin_on_vertex && isodd(nz)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and nz being odd; set either `origin_on_vertex=false` or make nz even."); end
    if (centerx && !origin_on_vertex && iseven(nx)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and nx being even; set either `origin_on_vertex=true` or make nx odd."); end
    if (centery && !origin_on_vertex && iseven(ny)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and ny being even; set either `origin_on_vertex=true` or make ny odd."); end
    if (centerz && !origin_on_vertex && iseven(nz)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and nz being even; set either `origin_on_vertex=true` or make nz odd."); end
    if (any(halowidths .< 1)) error("Invalid arguments: halowidths cannot be less than 1."); end
    if (nx==1) error("Invalid arguments: nx can never be 1.") end
    if (ny==1 && nz>1) error("Invalid arguments: ny cannot be 1 if nz is greater than 1.") end
    if (any((nxyz .== 1) .& (dims .>1 ))) error("Incoherent arguments: if nx, ny, or nz is 1, then the corresponding dimx, dimy or dimz must not be set (or set 0 or 1)."); end
    if (any((nxyz .< 2 .* overlaps .- 1) .& (periods .> 0))) error("Incoherent arguments: if nx, ny, or nz is smaller than 2*overlaps[1]-1, 2*overlaps[2]-1 or 2*overlaps[3]-1, respectively, then the corresponding periodx, periody or periodz must not be set (or set 0)."); end
    if (any((overlaps .> 0) .& (halowidths .> overlaps.÷2))) error("Incoherent arguments: if overlap is greater than 0, then halowidth cannot be greater than overlap÷2, in each dimension."); end
    dims[(nxyz.==1).&(dims.==0)] .= 1;   # Setting any of nxyz to 1, means that the corresponding dimension must also be 1 in the global grid. Thus, the corresponding dims entry must be 1.
    if (init_MPI)  # NOTE: init MPI only, once the input arguments have been checked.
        if (MPI.Initialized()) error("MPI is already initialized. Set the argument 'init_MPI=false'."); end
        MPI.Init();
    else
        if (!MPI.Initialized()) error("MPI has not been initialized beforehand. Remove the argument 'init_MPI=false'."); end  # Ensure that MPI is always initialized after init_global_grid().
    end
    nprocs    = MPI.Comm_size(comm);
    dims     .= MPI.Dims_create!(nprocs, dims);
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder);
    me        = MPI.Comm_rank(comm_cart);
    coords    = MPI.Cart_coords(comm_cart);
    neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    nxyz_g = dims.*(nxyz.-overlaps) .+ overlaps.*(periods.==0); # E.g. for dimension x with ol=2 and periodx=0: dimx*(nx-2)+2
    set_global_grid(GlobalGrid(nxyz_g, nxyz, dims, overlaps, halowidths, origin, origin_on_vertex, centerxyz, nprocs, me, coords, neighbors, periods, disp, reorder, comm_cart, cuda_enabled, amdgpu_enabled, cudaaware_MPI, amdgpuaware_MPI, use_polyester, quiet));
    cuda_support_string   = (cuda_enabled && all(cudaaware_MPI))     ? "CUDA-aware"   : (cuda_enabled && any(cudaaware_MPI))     ? "CUDA(-aware)"   : (cuda_enabled)   ? "CUDA"   : "";
    amdgpu_support_string = (amdgpu_enabled && all(amdgpuaware_MPI)) ? "AMDGPU-aware" : (amdgpu_enabled && any(amdgpuaware_MPI)) ? "AMDGPU(-aware)" : (amdgpu_enabled) ? "AMDGPU" : "";
    gpu_support_string    = join(filter(!isempty, [cuda_support_string, amdgpu_support_string]), ", ");
    support_string        = isempty(gpu_support_string) ? "none" : gpu_support_string;
    if (!quiet && me==0) println("Global grid: $(nxyz_g[1])x$(nxyz_g[2])x$(nxyz_g[3]) (nprocs: $nprocs, dims: $(dims[1])x$(dims[2])x$(dims[3]); device support: $support_string)"); end
    if ((cuda_enabled || amdgpu_enabled) && select_device) _select_device() end
    init_timing_functions();
    return me, dims, nprocs, coords, comm_cart; # The typical use case requires only these variables; the remaining can be obtained calling get_global_grid() if needed.
end

# Make sure that timing functions which must be fast at the first user call are already compiled now.
function init_timing_functions()
    tic();
    toc();
end
