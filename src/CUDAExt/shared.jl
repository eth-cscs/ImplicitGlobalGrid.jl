import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber
import ImplicitGlobalGrid: NNEIGHBORS_PER_DIM, GG_ALLOC_GRANULARITY
using CUDA


##------
## TYPES

const CuField{T,N} = GGField{T,N,CuArray{T,N}}


##------------------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT

ImplicitGlobalGrid.is_loaded(::Val{:ImplicitGlobalGrid_CUDAExt}) = (@assert CUDA.functional(true); return true)


##-------------
## SYNTAX SUGAR

ImplicitGlobalGrid.is_cuarray(A::GGArray) = typeof(A) <: CuArray   #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

ImplicitGlobalGrid.wrap_field(A::CuArray, hw::Tuple) = CuField{eltype(A), ndims(A)}((A, hw))

Base.size(A::CuField)          = Base.size(A.A)
Base.size(A::CuField, args...) = Base.size(A.A, args...)
Base.length(A::CuField)        = Base.length(A.A)
Base.ndims(A::CuField)         = Base.ndims(A.A)
Base.eltype(A::CuField)        = Base.eltype(A.A)


##---------------
## CUDA functions

function register(::Type{<:CuArray},buf::Array{T}) where T <: GGNumber
    rbuf = CUDA.Mem.register(CUDA.Mem.Host, pointer(buf), sizeof(buf), CUDA.Mem.HOSTREGISTER_DEVICEMAP);
    rbuf_d = convert(CuPtr{T}, rbuf);
    return unsafe_wrap(CuArray, rbuf_d, size(buf)), rbuf;
end
