import MPI
import LinearAlgebra
using Base.Threads


##------------------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT

let
    global cuda_loaded, cuda_functional, amdgpu_loaded, amdgpu_functional, set_cuda_loaded, set_cuda_functional, set_amdgpu_loaded, set_amdgpu_functional
    _cuda_loaded::Bool        = false
    _cuda_functional::Bool    = false
    _amdgpu_loaded::Bool      = false
    _amdgpu_functional::Bool  = false
    cuda_loaded()::Bool       = _cuda_loaded
    cuda_functional()::Bool   = _cuda_functional
    amdgpu_loaded()::Bool     = _amdgpu_loaded
    amdgpu_functional()::Bool = _amdgpu_functional
    set_cuda_loaded()         = (_cuda_loaded = is_loaded(Val(:ImplicitGlobalGrid_CUDAExt)))
    set_cuda_functional()     = (_cuda_functional = is_functional(Val(:CUDA)))
    set_amdgpu_loaded()       = (_amdgpu_loaded = is_loaded(Val(:ImplicitGlobalGrid_AMDGPUExt)))
    set_amdgpu_functional()   = (_amdgpu_functional = is_functional(Val(:AMDGPU)))
end


##--------------------
## CONSTANT PARAMETERS

const NDIMS_MPI = 3                    # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2           # Number of neighbors per dimension (left neighbor + right neighbor).
const GG_ALLOC_GRANULARITY = 32        # Internal buffers are allocated with a granulariy of GG_ALLOC_GRANULARITY elements in order to ensure correct reinterpretation when used for different types and to reduce amount of re-allocations.
const GG_THREADCOPY_THRESHOLD = 32768  # When Polyester is deactivated, then the GG_THREADCOPY_THRESHOLD defines the size in bytes upon which memory copy is performed with multiple threads.
const DEVICE_TYPE_NONE = "none"
const DEVICE_TYPE_AUTO = "auto"
const DEVICE_TYPE_CUDA = "CUDA"
const DEVICE_TYPE_AMDGPU = "AMDGPU"
const SUPPORTED_DEVICE_TYPES = [DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU]


##------
## TYPES

const AnyCPUArray{T,N}                = Union{Base.LogicalIndex{T, <:Array}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:Array, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:Array, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:Array, Array{T, N}, LinearAlgebra.Adjoint{T, <:Array{T, N}}, LinearAlgebra.Diagonal{T, <:Array{T, N}}, LinearAlgebra.LowerTriangular{T, <:Array{T, N}}, LinearAlgebra.Symmetric{T, <:Array{T, N}}, LinearAlgebra.Transpose{T, <:Array{T, N}}, LinearAlgebra.Tridiagonal{T, <:Array{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:Array{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:Array{T, N}}, LinearAlgebra.UpperTriangular{T, <:Array{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:Array}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
const AnyDenseArray{T,N}              = Union{Base.LogicalIndex{T, <:DenseArray}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:DenseArray, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:DenseArray, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:DenseArray, DenseArray{T, N}, LinearAlgebra.Adjoint{T, <:DenseArray{T, N}}, LinearAlgebra.Diagonal{T, <:DenseArray{T, N}}, LinearAlgebra.LowerTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.Symmetric{T, <:DenseArray{T, N}}, LinearAlgebra.Transpose{T, <:DenseArray{T, N}}, LinearAlgebra.Tridiagonal{T, <:DenseArray{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.UpperTriangular{T, <:DenseArray{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:DenseArray}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
const GGInt                           = Cint
const GGNumber                        = Any # NOTE: In every user facing function, it must be verified that it is bits (union) type # was Number
const GGArray{T,N}                    = AnyDenseArray{T,N} # was Union{Array{T,N}, CuArray{T,N}, ROCArray{T,N}}
const GGField{T,N,T_array}            = NamedTuple{(:A, :halowidths), Tuple{T_array, Tuple{GGInt,GGInt,GGInt}}} where {T_array<:GGArray{T,N}}
const GGFieldConvertible{T,N,T_array} = NamedTuple{(:A, :halowidths), <:Tuple{T_array, Tuple{T2,T2,T2}}} where {T_array<:GGArray{T,N}, T2<:Integer}
const GGField{}(t::NamedTuple)        = GGField{eltype(t.A),ndims(t.A),typeof(t.A)}((t.A, GGInt.(t.halowidths)))
const CPUField{T,N}                   = GGField{T,N,<:AnyCPUArray{T,N}}

"A GlobalGrid struct contains information on the grid and the corresponding MPI communicator." # Note: type GlobalGrid is immutable, i.e. users can only read, but not modify it (except the actual entries of arrays can be modified, e.g. dims .= dims - useful for writing tests)
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
    use_polyester::Vector{Bool}
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
use_polyester()                        = global_grid().use_polyester
use_polyester(dim::Integer)            = global_grid().use_polyester[dim]
has_neighbor(n::Integer, dim::Integer) = neighbor(n, dim) != MPI.PROC_NULL
any_array(fields::GGField...)          = any([is_array(A.A) for A in fields])
any_cuarray(fields::GGField...)        = any([is_cuarray(A.A) for A in fields])
any_rocarray(fields::GGField...)       = any([is_rocarray(A.A) for A in fields])
all_arrays(fields::GGField...)         = all([is_array(A.A) for A in fields])
all_cuarrays(fields::GGField...)       = all([is_cuarray(A.A) for A in fields])
all_rocarrays(fields::GGField...)      = all([is_rocarray(A.A) for A in fields])
is_array(A::GGArray)                   = typeof(A) <: AnyCPUArray


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

wrap_field(A::GGField)                 = A
wrap_field(A::GGFieldConvertible)      = GGField(A)
wrap_field(A::GGArray, hw::Integer...) = wrap_field(A, hw)
wrap_field(A::GGArray)                 = wrap_field(A, hw_default()...)
wrap_field(A::T_array, hw::Tuple) where {T_array <: GGArray} = GGField{eltype(A),ndims(A),T_array}((A, hw))

Base.size(A::Union{GGField, CPUField}, args...) = Base.size(A.A, args...)
Base.length(A::Union{GGField, CPUField})        = Base.length(A.A)
Base.ndims(A::Union{GGField, CPUField})         = Base.ndims(A.A)
Base.eltype(A::Union{GGField, CPUField})        = Base.eltype(A.A)

Base.iscontiguous(A::DenseArray) = true


##------------------------------------------
## CUDA AND AMDGPU COMMON EXTENSION DEFAULTS
# TODO: this should not be required as only called from the extensions #function register end