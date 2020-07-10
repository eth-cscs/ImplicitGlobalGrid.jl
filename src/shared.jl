# Enable CUDA if the required packages are installed and functional (enables to use the package for CPU-only without requiring the CUDA packages functional - or even not at all if the installation procedure allows it). NOTE: it cannot be precompiled for GPU on a node without GPU.
const CUDA_IS_INSTALLED = (Base.find_package("CUDA")!==nothing && import CUDA==nothing && CUDA.functional())
const ENABLE_CUDA = CUDA_IS_INSTALLED # Can of course be set to false even if CUDA_IS_INSTALLED.
macro enable_if_cuda(block) # Macro intended to put one-liners depending on ENABLE_CUDA (Note an alternative would be to create always a function for CPU and GPU and rely on multiple dispatch).
    esc(
        quote
            @static if ENABLE_CUDA
                $block
            end
        end
    )
end

import MPI
@static if ENABLE_CUDA
    using CUDA
    __init__() = @assert CUDA.functional(true)
end


##--------------------
## CONSTANT PARAMETERS

const NDIMS_MPI = 3              # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2     # Number of neighbors per dimension (left neighbor + right neighbor).
const GG_ALLOC_GRANULARITY = 32  # Internal buffers are allocated with a granulariy of GG_ALLOC_GRANULARITY elements in order to ensure correct reinterpretation when used for different types and to reduce amount of re-allocations.


##------
## TYPES

const GGInt    = Cint
const GGNumber = Real

@static if ENABLE_CUDA
    const GGArray{T,N} = Union{Array{T,N}, CuArray{T,N}}
    const cuzeros = CUDA.zeros
else
    const GGArray{T,N} = Union{Array{T,N}}
end

"An GlobalGrid struct contains information on the grid and the corresponding MPI communicator." # Note: type GlobalGrid is immutable, i.e. users can only read, but not modify it (except the actual entries of arrays can be modified, e.g. dims .= dims - useful for writing tests)
struct GlobalGrid
    nxyz_g::Vector{GGInt}
    nxyz::Vector{GGInt}
    dims::Vector{GGInt}
    overlaps::Vector{GGInt}
    nprocs::GGInt
    me::GGInt
    coords::Vector{GGInt}
    neighbors::Array{GGInt, NNEIGHBORS_PER_DIM}
    periods::Vector{GGInt}
    disp::GGInt
    reorder::GGInt
    comm::MPI.Comm
    cudaaware_MPI::Vector{Bool}
    quiet::Bool
end
const GLOBAL_GRID_NULL = GlobalGrid(GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], -1, -1, GGInt[-1,-1,-1], GGInt[-1 -1 -1; -1 -1 -1], GGInt[-1,-1,-1], -1, -1, MPI.COMM_NULL, [false,false,false], false)

# Macro to switch on/off check_initialized() for performance reasons (potentially relevant for tools.jl).
macro check_initialized() :(check_initialized();) end  #FIXME: Alternative: macro check_initialized() end
let
    global global_grid, set_global_grid, grid_is_initialized, check_initialized, get_global_grid

    _global_grid::GlobalGrid           = GLOBAL_GRID_NULL
    global_grid()::GlobalGrid          = (@check_initialized(); _global_grid::GlobalGrid) # Thanks to the call to check_initialized, we can be sure that _global_grid is defined and therefore must be of type GlobalGrid.
    set_global_grid(gg::GlobalGrid)    = (_global_grid = gg;)
    grid_is_initialized()              = (_global_grid.nprocs > 0)
    check_initialized()                = if !grid_is_initialized() error("No function of the module can be called before init_global_grid() or after finalize_global_grid().") end

    "Return a deep copy of the global grid."
    get_global_grid()                  = deepcopy(_global_grid)
end


##-------------
## SYNTAX SUGAR

macro require(condition) esc(:( if !($condition) error("Pre-test requirement not met: $condition") end )) end  # Verify a condition required for a unit test (in the unit test results, this should not be treated as a unit test).
longnameof(f)                          = "$(parentmodule(f)).$(nameof(f))"
isnothing(x::Any)                      = x === nothing ? true : false # To ensure compatibility for Julia >=v1
none(x::AbstractArray{Bool})           = all(x.==false)
me()                                   = global_grid().me
comm()                                 = global_grid().comm
ol(dim::Integer)                       = global_grid().overlaps[dim]
ol(dim::Integer, A::GGArray)           = global_grid().overlaps[dim] + (size(A,dim) - global_grid().nxyz[dim])
neighbors(dim::Integer)                = global_grid().neighbors[:,dim]
neighbor(n::Integer, dim::Integer)     = global_grid().neighbors[n,dim]
cudaaware_MPI()                        = global_grid().cudaaware_MPI
cudaaware_MPI(dim::Integer)            = global_grid().cudaaware_MPI[dim]
has_neighbor(n::Integer, dim::Integer) = neighbor(n, dim) != MPI.MPI_PROC_NULL
any_array(fields::GGArray...)          = any([is_array(A) for A in fields])
any_cuarray(fields::GGArray...)        = any([is_cuarray(A) for A in fields])
is_array(A::GGArray)                   = typeof(A) <: Array
@static if ENABLE_CUDA
    is_cuarray(A::GGArray)             = typeof(A) <: CuArray  #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.
else
    is_cuarray(A::GGArray)             = false
end


##---------------
## CUDA functions

@static if ENABLE_CUDA
    function register(buf::Array{T}) where T <: GGNumber
        rbuf = Mem.register(Mem.Host, pointer(buf), sizeof(buf), Mem.HOSTREGISTER_DEVICEMAP);
        rbuf_d = convert(CuPtr{T}, rbuf);
        return unsafe_wrap(CuArray, rbuf_d, size(buf)), rbuf;
    end
end
