import MPI
using Base.Threads
using CellArrays
export get_global_grid

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

const AnyCPUArray{T,N}                    = AbstractArray{T,N} # NOTE: In every user facing function, it must be verified that its elements are bits (union) type and views contiguous. # was Union{Base.LogicalIndex{T, <:Array}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:Array, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:Array, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:Array, Array{T, N}, LinearAlgebra.Adjoint{T, <:Array{T, N}}, LinearAlgebra.Diagonal{T, <:Array{T, N}}, LinearAlgebra.LowerTriangular{T, <:Array{T, N}}, LinearAlgebra.Symmetric{T, <:Array{T, N}}, LinearAlgebra.Transpose{T, <:Array{T, N}}, LinearAlgebra.Tridiagonal{T, <:Array{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:Array{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:Array{T, N}}, LinearAlgebra.UpperTriangular{T, <:Array{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:Array}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
const AnyDenseArray{T,N}                  = AbstractArray{T,N} # NOTE: In every user facing function, it must be verified that its elements are bits (union) type and views contiguous. # was Union{Base.LogicalIndex{T, <:DenseArray}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:DenseArray, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:DenseArray, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:DenseArray, DenseArray{T, N}, LinearAlgebra.Adjoint{T, <:DenseArray{T, N}}, LinearAlgebra.Diagonal{T, <:DenseArray{T, N}}, LinearAlgebra.LowerTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.Symmetric{T, <:DenseArray{T, N}}, LinearAlgebra.Transpose{T, <:DenseArray{T, N}}, LinearAlgebra.Tridiagonal{T, <:DenseArray{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:DenseArray{T, N}}, LinearAlgebra.UpperTriangular{T, <:DenseArray{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:DenseArray}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
const AnyCellArray{T,N}                   = Union{Base.LogicalIndex{T, <:CellArray}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:CellArray, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:CellArray, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:CellArray, CellArray{T, N}, PermutedDimsArray{T, N, <:Any, <:Any, <:CellArray}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray. #was with LinearAlgebra: Union{Base.LogicalIndex{T, <:CellArray}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:CellArray, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:CellArray, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:CellArray, CellArray{T, N}, LinearAlgebra.Adjoint{T, <:CellArray{T, N}}, LinearAlgebra.Diagonal{T, <:CellArray{T, N}}, LinearAlgebra.LowerTriangular{T, <:CellArray{T, N}}, LinearAlgebra.Symmetric{T, <:CellArray{T, N}}, LinearAlgebra.Transpose{T, <:CellArray{T, N}}, LinearAlgebra.Tridiagonal{T, <:CellArray{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:CellArray{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:CellArray{T, N}}, LinearAlgebra.UpperTriangular{T, <:CellArray{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:CellArray}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
const GGInt                               = Int64 # NOTE: was Cint which is Int32 and may overflow for large numbers
const GGNumber                            = Any # NOTE: In every user facing function, it must be verified that it is bits (union) type # was Number
const GGArray{T,N}                        = AnyDenseArray{T,N} # was Union{Array{T,N}, CuArray{T,N}, ROCArray{T,N}}
const GGCellArray{T,N}                    = AnyCellArray{T,N}
const GGField{T,N,T_array}                = NamedTuple{(:A, :halowidths), Tuple{T_array, Tuple{GGInt,GGInt,GGInt}}} where {T_array<:GGArray{T,N}}
const GGCellField{T,N,T_array}            = NamedTuple{(:A, :halowidths), Tuple{T_array, Tuple{GGInt,GGInt,GGInt}}} where {T_array<:GGCellArray{T,N}}
const GGFieldConvertible{T,N,T_array}     = NamedTuple{(:A, :halowidths), <:Tuple{T_array, Tuple{T2,T2,T2}}} where {T_array<:GGArray{T,N}, T2<:Integer}
const GGCellFieldConvertible{T,N,T_array} = NamedTuple{(:A, :halowidths), <:Tuple{T_array, Tuple{T2,T2,T2}}} where {T_array<:GGCellArray{T,N}, T2<:Integer}
const GGField{}(t::NamedTuple)            = GGField{eltype(t.A),ndims(t.A),typeof(t.A)}((t.A, GGInt.(t.halowidths)))
const CPUField{T,N}                       = GGField{T,N,<:AnyCPUArray{T,N}}

"A GlobalGrid struct contains information on the grid and the corresponding MPI communicator." # Note: type GlobalGrid is immutable, i.e. users can only read, but not modify it (except the actual entries of arrays can be modified, e.g. dims .= dims - useful for writing tests)
struct GlobalGrid
    nxyz_g::Vector{GGInt}
    nxyz::Vector{GGInt}
    dims::Vector{GGInt}
    overlaps::Vector{GGInt}
    halowidths::Vector{GGInt}
    origin::Vector{Float64}
    origin_on_vertex::Bool
    centerxyz::Vector{Bool}
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
const GLOBAL_GRID_NULL = GlobalGrid(GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], GGInt[-1,-1,-1], Float64[-1,-1,-1], false, [false,false,false], -1, -1, GGInt[-1,-1,-1], GGInt[-1 -1 -1; -1 -1 -1], GGInt[-1,-1,-1], -1, -1, MPI.COMM_NULL, false, false, [false,false,false], [false,false,false], [false,false,false], false)

"""
    get_global_grid() :: GlobalGrid

    Returns a deep copy of the currently active global grid. If no grid is active a GLOBAL_GRID_NULL will be returned, with negative nprocs. 
"""
function get_global_grid end

# Macro to switch on/off check_initialized() for performance reasons (potentially relevant for tools.jl).
macro check_initialized() :(check_initialized();) end  #FIXME: Alternative: macro check_initialized() end
let
    global global_grid, set_global_grid, grid_is_initialized, check_initialized, check_not_initialized, check_grid_is_initialized, get_global_grid, set_initialized, _unsafe_get_comm

    _global_grid::GlobalGrid           = GLOBAL_GRID_NULL
    _init_::Bool                        = false
    global_grid()::GlobalGrid          = (check_grid_is_initialized(); _global_grid::GlobalGrid) # Protected access for internal use
    set_global_grid(gg::GlobalGrid)    = (_global_grid = gg;)
    set_initialized(val::Bool = true)  = (_init_ = val; nothing)
    grid_is_initialized()              = (_global_grid.nprocs > 0)
    check_initialized()                = if !_init_ error("No function of the module can be called before init_global_grid().") end
    check_not_initialized()            = if _init_ error("init_global_grid() can only be called once before finalize_global_grid().") end
    check_grid_is_initialized()        = (if !grid_is_initialized() error("No global grid has been created and activated yet, or the package has not been initialized.") end)

    function get_global_grid() :: GlobalGrid
        check_initialized();
        return deepcopy(_global_grid)
    end

    _unsafe_get_comm()         = _global_grid.comm # For toc so there is no overhead of init checks
end


let 
    global  differ_default_args, set_default_args, reset_default_args, default

    _default_args::Dict{Symbol,Any}    = Dict([
        :dimx          => 0,
        :dimy          => 0,
        :dimz          => 0,
        :periodx       => 0,
        :periody       => 0,
        :periodz       => 0,
        :origin        => (0.0, 0.0, 0.0),
        :origin_on_vertex => false,
        :centerx       => false,
        :centery       => false,
        :centerz       => false,
        :overlaps      => (2, 2, 2),
        :halowidths    => max.(1, (2,2,2) .÷ 2),
        :disp          => 1,
        :reorder       => 1,
        :comm          => MPI.COMM_WORLD,
        :device_type   => DEVICE_TYPE_AUTO, # as of now changing this between grids is disabled
        :select_device => true,             # as of now changing this between grids is disabled
        :quiet         => false,        
    ])
    DEFAULT_OF_DEFAULT_ARGS::Dict{Symbol,Any} = copy(_default_args) # This is used in the corner case of finalizing and reinitializing to reset the default arguments to their original values.
    # Check if the value set is not the same as the current default.
    differ_default_args(;kwargs...)    = any([haskey(kwargs, key) && kwargs[key] != value for (key, value) in _default_args])
    # Set the new argument set as default
    set_default_args(;kwargs...)       = ([_default_args[key] = kwargs[key] for key in keys(kwargs)];nothing)
    reset_default_args()               = (_default_args = copy(DEFAULT_OF_DEFAULT_ARGS); nothing)
    # For an argument get its default value
    default(key::Symbol)               = _default_args[key]
end


"""
    normalize_input(nx,ny,nz,...)
    normalize_input(...)

    Validates and normalizes the input arguments for implicit global grid creation.
"""
function normalize_input(dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
    # Signature includes all params for easy extensibility of checks
    dims              = [dimx, dimy, dimz];
    periods           = Int64.([periodx, periody, periodz]);
    # origin: In the GG is a vector but in the arguments is a tuple
    origin            = Float64.((((length((origin...,)) == 1) ?  (origin, 0, 0) : ((length(origin) == 2) ? (origin..., 0) : origin))))
    # Value checks
    if !(device_type in [DEVICE_TYPE_NONE, DEVICE_TYPE_AUTO, DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU]) error("Argument `device_type`: invalid value obtained ($device_type). Valid values are: $DEVICE_TYPE_CUDA, $DEVICE_TYPE_AMDGPU, $DEVICE_TYPE_NONE, $DEVICE_TYPE_AUTO") end
    if ((device_type == DEVICE_TYPE_AUTO) && cuda_loaded() && cuda_functional() && amdgpu_loaded() && amdgpu_functional()) error("Automatic detection of the device type to be used not possible: both CUDA and AMDGPU extensions are loaded and functional. Set keyword argument `device_type` to $DEVICE_TYPE_CUDA or $DEVICE_TYPE_AMDGPU.") end
    if (any(dims .< 0)) error("Invalid arguments: dimx, dimy, and dimz cannot be negative."); end
    if (any(periods .∉ ((0,1),))) error("Invalid arguments: periodx, periody, and periodz must be either 0 or 1."); end
    if length(origin) != 3 error("Invalid argument: the length of the origin tuple must be at most 3.") end
    if (any(halowidths .< 1)) error("Invalid arguments: halowidths cannot be less than 1."); end
    if (any((overlaps .> 0) .& (halowidths .> overlaps.÷2))) error("Incoherent arguments: if overlap is greater than 0, then halowidth cannot be greater than overlap÷2, in each dimension."); end
    return dimx, dimy, dimz, 
        periodx, periody, periodz, 
        origin, origin_on_vertex, 
        centerx, centery, centerz, 
        overlaps, halowidths, disp, 
        reorder, comm, 
        device_type, select_device, quiet 
end

function normalize_input(nx, ny, nz, dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
    # Signature includes all params for easy extensibility of checks
    # Checks without grid size
    dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet = normalize_input(dimx, dimy, dimz, periodx, periody, periodz, origin, origin_on_vertex, centerx, centery, centerz, overlaps, halowidths, disp, reorder, comm, device_type, select_device, quiet)
    dims              = [dimx, dimy, dimz];
    nxyz              = [nx, ny, nz];
    periods           = [periodx, periody, periodz];
    # Value checks
    if (any(nxyz .< 1)) error("Invalid arguments: nx, ny, and nz cannot be less than 1."); end
    if (centerx && origin_on_vertex && isodd(nx)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and nx being odd; set either `origin_on_vertex=false` or make nx even."); end
    if (centery && origin_on_vertex && isodd(ny)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and ny being odd; set either `origin_on_vertex=false` or make ny even."); end
    if (centerz && origin_on_vertex && isodd(nz)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin on the cell vertex and nz being odd; set either `origin_on_vertex=false` or make nz even."); end
    if (centerx && !origin_on_vertex && iseven(nx)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and nx being even; set either `origin_on_vertex=true` or make nx odd."); end
    if (centery && !origin_on_vertex && iseven(ny)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and ny being even; set either `origin_on_vertex=true` or make ny odd."); end
    if (centerz && !origin_on_vertex && iseven(nz)) error("Incoherent arguments: the grid cannot be centered on the origin with the constraint to have the origin the cell center and nz being even; set either `origin_on_vertex=true` or make nz odd."); end
    if (nx==1) error("Invalid arguments: nx can never be 1.") end
    if (ny==1 && nz>1) error("Invalid arguments: ny cannot be 1 if nz is greater than 1.") end
    if (any((nxyz .== 1) .& (dims .>1 ))) error("Incoherent arguments: if nx, ny, or nz is 1, then the corresponding dimx, dimy or dimz must not be set (or set 0 or 1)."); end
    if (any((nxyz .< 2 .* overlaps .- 1) .& (periods .> 0))) error("Incoherent arguments: if nx, ny, or nz is smaller than 2*overlaps[1]-1, 2*overlaps[2]-1 or 2*overlaps[3]-1, respectively, then the corresponding periodx, periody or periodz must not be set (or set 0)."); end
    dims[(nxyz.==1).&(dims.==0)] .= 1;   # Setting any of nxyz to 1, means that the corresponding dimension must also be 1 in the global grid. Thus, the corresponding dims entry must be 1.
    return nx,ny,nz, 
        dimx, dimy, dimz, 
        periodx, periody, periodz, 
        origin, origin_on_vertex,
        centerx, centery, centerz, 
        overlaps, halowidths, disp, 
        reorder, comm, 
        device_type, select_device, quiet
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
is_array(A::GGArray)                   = !(is_cuarray(A) || is_rocarray(A)) # TODO: should later be AnyCPUArray/AbstractCPUArray for clear error messages


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

extract(A::GGField)                = (A,)
extract(A::GGFieldConvertible)     = (A,)
extract(A::GGArray)                = (A,)
extract(A::GGCellFieldConvertible) = (NamedTuple{keys(A)}((array, A.halowidths)) for array in bitsarrays(A.A))
extract(A::CellArray)              = bitsarrays(A)

wrap_field(A::GGField)                 = A
wrap_field(A::GGFieldConvertible)      = GGField(A)
wrap_field(A::GGArray, hw::Integer...) = wrap_field(A, hw)
wrap_field(A::GGArray)                 = wrap_field(A, hw_default()...)
wrap_field(A::T_array, hw::Tuple) where {T_array <: GGArray} = GGField{eltype(A),ndims(A),T_array}((A, hw))

Base.size(A::Union{GGField, CPUField}, args...) = Base.size(A.A, args...)
Base.length(A::Union{GGField, CPUField})        = Base.length(A.A)
Base.ndims(A::Union{GGField, CPUField})         = Base.ndims(A.A)
Base.eltype(A::Union{GGField, CPUField})        = Base.eltype(A.A)

Base.iscontiguous(A::DenseArray)                        = true
Base.iscontiguous(A::Base.ReinterpretArray)             = Base.iscontiguous(A.parent)
Base.iscontiguous(A::Base.ReshapedArray)                = Base.iscontiguous(A.parent)
Base.iscontiguous(A::PermutedDimsArray)                 = Base.iscontiguous(A.parent)
#NOTE: with LinearAlgebra as a dependency one could do something similar:
# Base.iscontiguous(A::LinearAlgebra.Adjoint)             = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.Diagonal)            = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.LowerTriangular)     = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.Symmetric)           = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.Transpose)           = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.Tridiagonal)         = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.UnitLowerTriangular) = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.UnitUpperTriangular) = Base.iscontiguous(A.parent)
# Base.iscontiguous(A::LinearAlgebra.UpperTriangular)     = Base.iscontiguous(A.parent)


# CellArray helper function (This could be moved to CellArrays.jl if useful also for other purposes)
#
# """
#     bitsarrays(A)
#
# Return a (named) tuple containing the fields of CellArray `A` as `N`-dimensional bits type array view(s). The views' dimensionality and size are equal to `A`'s. The operation is not supported if parameter `B` of `A` is neither `0` nor `1`.
#
# """
@inline bitsarrays(A::CellArray{T,N,0,T_array}) where {T,N,  T_array} = (field(A, i) for i=1:celllength(A))
@inline bitsarrays(A::CellArray{T,N,1,T_array}) where {T,N,  T_array} = (reshape(reinterpret(T, view(A.data,:)), size(A)),)
@inline bitsarrays(A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = error("only CellArrays with B=0 or B=1 are supported")

##------------------------------------------
## CUDA AND AMDGPU COMMON EXTENSION DEFAULTS
# TODO: this should not be required as only called from the extensions #function register end
