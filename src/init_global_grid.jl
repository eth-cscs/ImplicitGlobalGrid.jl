export init_global_grid

import MPI


"""
    init_global_grid(nx, ny, nz)
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; <keyword arguments>)

Initialize a Cartesian grid of MPI processes (and also MPI itself by default) defining implicitely a global grid.

# Arguments
- {`nx`|`ny`|`nz`}`::Integer`: the number of elements of the local grid in dimension {x|y|z}.
- {`dimx`|`dimy`|`dimz`}`::Integer=0`: the desired number of processes in dimension {x|y|z}. By default, (value `0`) the process topology is created as compact as possible with the given constraints. This is handled by the MPI implementation which is installed on your system. For more information, refer to the specifications of `MPI_Dims_create` in the corresponding documentation.
- {`periodx`|`periody`|`periodz`}`::Integer=0`: whether the grid is periodic (`1`) or not (`0`) in dimension {x|y|z}.
- `quiet::Bool=false`: whether to suppress printing information like the size of the global grid (`true`) or not (`false`).
!!! note "Advanced keyword arguments"
    - {`overlapx`|`overlapy`|`overlapz`}`::Integer=2`: the number of elements adjacent local grids overlap in dimension {x|y|z}. By default (value `2`), an array `A` of size (`nx`, `ny`, `nz`) on process 1 (`A_1`) overlaps the corresponding array `A` on process 2 (`A_2`) by `2` indices if the two processes are adjacent. E.g., if `overlapx=2` and process 2 is the right neighbor of process 1 in dimension x, then `A_1[end-1:end,:,:]` overlaps `A_2[1:2,:,:]`. That means, after every call `update_halo!(A)`, we have `all(A_1[end-1:end,:,:] .== A_2[1:2,:,:])` (`A_1[end,:,:]` is the halo of process 1 and `A_2[1,:,:]` is the halo of process 2). The analog applies for the dimensions y and z.
    - `disp::Integer=1`:  the displacement argument to `MPI.Cart_shift` in order to determine the neighbors.
    - `reorder::Integer=1`: the reorder argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    - `comm::MPI.Comm=MPI.COMM_WORLD`: the input communicator argument to `MPI.Cart_create` in order to create the Cartesian process topology.
    - `init_MPI::Bool=true`: whether to initialize MPI (`true`) or not (`false`).
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

See also: [`finalize_global_grid`](@ref)
"""
function init_global_grid(nx::Integer, ny::Integer, nz::Integer; dimx::Integer=0, dimy::Integer=0, dimz::Integer=0, periodx::Integer=0, periody::Integer=0, periodz::Integer=0, overlapx::Integer=2, overlapy::Integer=2, overlapz::Integer=2, disp::Integer=1, reorder::Integer=1, comm::MPI.Comm=MPI.COMM_WORLD, init_MPI::Bool=true, quiet::Bool=false)
    nxyz              = [nx, ny, nz];
    dims              = [dimx, dimy, dimz];
    periods           = [periodx, periody, periodz];
    overlaps          = [overlapx, overlapy, overlapz];
    cudaaware_MPI     = [false, false, false]
    loopvectorization = [false, false, false]
    if haskey(ENV, "IGG_CUDAAWARE_MPI") cudaaware_MPI .= (parse(Int64, ENV["IGG_CUDAAWARE_MPI"]) > 0); end
    if haskey(ENV, "IGG_LOOPVECTORIZATION") loopvectorization .= (parse(Int64, ENV["IGG_LOOPVECTORIZATION"]) > 0); end
    if none(cudaaware_MPI)
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMX") cudaaware_MPI[1] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMX"]) > 0); end
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMY") cudaaware_MPI[2] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMY"]) > 0); end
        if haskey(ENV, "IGG_CUDAAWARE_MPI_DIMZ") cudaaware_MPI[3] = (parse(Int64, ENV["IGG_CUDAAWARE_MPI_DIMZ"]) > 0); end
    end
    if all(loopvectorization)
        if haskey(ENV, "IGG_LOOPVECTORIZATION_DIMX") loopvectorization[1] = (parse(Int64, ENV["IGG_LOOPVECTORIZATION_DIMX"]) > 0); end
        if haskey(ENV, "IGG_LOOPVECTORIZATION_DIMY") loopvectorization[2] = (parse(Int64, ENV["IGG_LOOPVECTORIZATION_DIMY"]) > 0); end
        if haskey(ENV, "IGG_LOOPVECTORIZATION_DIMZ") loopvectorization[3] = (parse(Int64, ENV["IGG_LOOPVECTORIZATION_DIMZ"]) > 0); end
    end
    if (nx==1) error("Invalid arguments: nx can never be 1.") end
    if (ny==1 && nz>1) error("Invalid arguments: ny cannot be 1 if nz is greater than 1.") end
    if (any((nxyz .== 1) .& (dims .>1 ))) error("Incoherent arguments: if nx, ny, or nz is 1, then the corresponding dimx, dimy or dimz must not be set (or set 0 or 1)."); end
    if (any((nxyz .< 2 .* overlaps .- 1) .& (periods .> 0))) error("Incoherent arguments: if nx, ny, or nz is smaller than 2*overlapx-1, 2*overlapy-1 or 2*overlapz-1, respectively, then the corresponding periodx, periody or periodz must not be set (or set 0)."); end
    dims[(nxyz.==1).&(dims.==0)] .= 1;   # Setting any of nxyz to 1, means that the corresponding dimension must also be 1 in the global grid. Thus, the corresponding dims entry must be 1.
    if (init_MPI)  # NOTE: init MPI only, once the input arguments have been checked.
        if (MPI.Initialized()) error("MPI is already initialized. Set the argument 'init_MPI=false'."); end
        MPI.Init();
    else
        if (!MPI.Initialized()) error("MPI has not been initialized beforehand. Remove the argument 'init_MPI=false'."); end  # Ensure that MPI is always initialized after init_global_grid().
    end
    nprocs    = MPI.Comm_size(comm);
    MPI.Dims_create!(nprocs, dims);
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder);
    me        = MPI.Comm_rank(comm_cart);
    coords    = MPI.Cart_coords(comm_cart);
    neighbors = fill(MPI.MPI_PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    nxyz_g = dims.*(nxyz.-overlaps) .+ overlaps.*(periods.==0); # E.g. for dimension x with ol=2 and periodx=0: dimx*(nx-2)+2
    set_global_grid(GlobalGrid(nxyz_g, nxyz, dims, overlaps, nprocs, me, coords, neighbors, periods, disp, reorder, comm_cart, cudaaware_MPI, loopvectorization, quiet));
    if (!quiet && me==0) println("Global grid: $(nxyz_g[1])x$(nxyz_g[2])x$(nxyz_g[3]) (nprocs: $nprocs, dims: $(dims[1])x$(dims[2])x$(dims[3]))"); end
    init_timing_functions();
    return me, dims, nprocs, coords, comm_cart; # The typical use case requires only these variables; the remaining can be obtained calling get_global_grid() if needed.
end

# Make sure that timing functions which must be fast at the first user call are already compiled now.
function init_timing_functions()
    tic();
    toc();
end
