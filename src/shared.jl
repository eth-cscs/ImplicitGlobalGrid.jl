import MPI
using CUDA
using AMDGPU
using Base.Threads
using LoopVectorization


##-------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT
let
    global cuda_functional, amdgpu_functional, set_cuda_functional, set_amdgpu_functional
    _cuda_functional::Bool           = false
    _amdgpu_functional::Bool         = false
    cuda_functional()::Bool          = _cuda_functional
    amdgpu_functional()::Bool        = _amdgpu_functional
    set_cuda_functional(val::Bool)   = (_cuda_functional = val;)
    set_amdgpu_functional(val::Bool) = (_amdgpu_functional = val;)
end

function __init__()
    set_cuda_functional(CUDA.functional())
    set_amdgpu_functional(AMDGPU.functional())
end


##--------------------
## CONSTANT PARAMETERS

const NDIMS_MPI = 3                    # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2           # Number of neighbors per dimension (left neighbor + right neighbor).
const GG_ALLOC_GRANULARITY = 32        # Internal buffers are allocated with a granulariy of GG_ALLOC_GRANULARITY elements in order to ensure correct reinterpretation when used for different types and to reduce amount of re-allocations.
const GG_THREADCOPY_THRESHOLD = 32768  # When LoopVectorization is deactivated, then the GG_THREADCOPY_THRESHOLD defines the size in bytes upon which memory copy is performed with multiple threads.
const DEVICE_TYPE_AUTO = "auto"
const DEVICE_TYPE_CUDA = "CUDA"
const DEVICE_TYPE_AMDGPU = "AMDGPU"


##------
## TYPES

const GGInt                           = Cint
const GGNumber                        = Number
const GGArray{T,N}                    = Union{Array{T,N}, CuArray{T,N}, ROCArray{T,N}}
const GGField{T,N,T_array}            = NamedTuple{(:A, :halowidths), Tuple{T_array, Tuple{GGInt,GGInt,GGInt}}} where {T_array<:GGArray{T,N}}
const GGFieldConvertible{T,N,T_array} = NamedTuple{(:A, :halowidths), <:Tuple{T_array, Tuple{T2,T2,T2}}} where {T_array<:GGArray{T,N}, T2<:Integer}
const GGField{}(t::NamedTuple)        = GGField{eltype(t.A),ndims(t.A),typeof(t.A)}((t.A, GGInt.(t.halowidths)))
const CPUField{T,N}                   = GGField{T,N,Array{T,N}}
const CuField{T,N}                    = GGField{T,N,CuArray{T,N}}
const ROCField{T,N}                   = GGField{T,N,ROCArray{T,N}}

"An GlobalGrid struct contains information on the grid and the corresponding MPI communicator." # Note: type GlobalGrid is immutable, i.e. users can only read, but not modify it (except the actual entries of arrays can be modified, e.g. dims .= dims - useful for writing tests)
struct GlobalGrid
    nxyz_g::Vector{GGInt}
    nxyz::Vector{GGInt}
    dims::Vector{GGInt}
    overlaps::Vector{GGInt}
    halowidths::Vector{GGInt}
    nprocs::GGInt
    me::GGInt
    coords::Vector{GGInt}
    neighbors::Array{GGInt, NNEIGHBORS_PER_DIM}
    periods::Vector{GGInt}
    disp::GGInt
    reorder::GGInt
    comm::MPI.Comm
    cuda_enabled::Bool
    amdgpu_enabled::Bool
    cudaaware_MPI::Vector{Bool}
    amdgpuaware_MPI::Vector{Bool}
    loopvectorization::Vector{Bool}
    quiet::Bool
end
const GLOBAL_GRID_NULL = GlobalGrid(GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], -1, -1, GGInt[-1,-1,-1], GGInt[-1 -1 -1; -1 -1 -1], GGInt[-1,-1,-1], -1, -1, MPI.COMM_NULL, false, false, [false,false,false], [false,false,false], [false,false,false], false)

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
ol(A::GGArray)                         = (ol(dim, A) for dim in 1:ndims(A))
hw_default()                           = global_grid().halowidths
neighbors(dim::Integer)                = global_grid().neighbors[:,dim]
neighbor(n::Integer, dim::Integer)     = global_grid().neighbors[n,dim]
cuda_enabled()                         = global_grid().cuda_enabled
amdgpu_enabled()                       = global_grid().amdgpu_enabled
cudaaware_MPI()                        = global_grid().cudaaware_MPI
cudaaware_MPI(dim::Integer)            = global_grid().cudaaware_MPI[dim]
amdgpuaware_MPI()                      = global_grid().amdgpuaware_MPI
amdgpuaware_MPI(dim::Integer)          = global_grid().amdgpuaware_MPI[dim]
loopvectorization()                    = global_grid().loopvectorization
loopvectorization(dim::Integer)        = global_grid().loopvectorization[dim]
has_neighbor(n::Integer, dim::Integer) = neighbor(n, dim) != MPI.PROC_NULL
any_array(fields::GGField...)          = any([is_array(A.A) for A in fields])
any_cuarray(fields::GGField...)        = any([is_cuarray(A.A) for A in fields])
any_rocarray(fields::GGField...)       = any([is_rocarray(A.A) for A in fields])
is_array(A::GGArray)                   = typeof(A) <: Array
is_cuarray(A::GGArray)                 = typeof(A) <: CuArray  #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.
is_rocarray(A::GGArray)                = typeof(A) <: ROCArray  #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

wrap_field(A::GGField)                 = A
wrap_field(A::GGFieldConvertible)      = GGField(A)
wrap_field(A::Array, hw::Tuple)        = CPUField{eltype(A), ndims(A)}((A, hw))
wrap_field(A::CuArray, hw::Tuple)      = CuField{eltype(A), ndims(A)}((A, hw))
wrap_field(A::ROCArray, hw::Tuple)     = ROCField{eltype(A), ndims(A)}((A, hw))
wrap_field(A::GGArray, hw::Integer...) = wrap_field(A, hw)
wrap_field(A::GGArray)                 = wrap_field(A, hw_default()...)

Base.size(A::Union{GGField, CPUField, CuField, ROCField})          = Base.size(A.A)
Base.size(A::Union{GGField, CPUField, CuField, ROCField}, args...) = Base.size(A.A, args...)
Base.length(A::Union{GGField, CPUField, CuField, ROCField})        = Base.length(A.A)
Base.ndims(A::Union{GGField, CPUField, CuField, ROCField})         = Base.ndims(A.A)
Base.eltype(A::Union{GGField, CPUField, CuField, ROCField})        = Base.eltype(A.A)


##---------------
## CUDA functions

function register(::Type{<:CuArray},buf::Array{T}) where T <: GGNumber
    rbuf = CUDA.Mem.register(CUDA.Mem.Host, pointer(buf), sizeof(buf), CUDA.Mem.HOSTREGISTER_DEVICEMAP);
    rbuf_d = convert(CuPtr{T}, rbuf);
    return unsafe_wrap(CuArray, rbuf_d, size(buf)), rbuf;
end


##---------------
## AMDGPU functions

function register(::Type{<:ROCArray},buf::Array{T}) where T <: GGNumber
    return unsafe_wrap(ROCArray, pointer(buf), size(buf))
end
