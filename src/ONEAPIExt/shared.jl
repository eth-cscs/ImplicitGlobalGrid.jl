import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber, halosize, ol, oneapiaware_MPI, sendranges, recvranges, sendbuf_flat, recvbuf_flat, write_d2x!, read_x2d!, write_d2h_async!, read_h2d_async!, register, is_cuarray
import ImplicitGlobalGrid: NNEIGHBORS_PER_DIM, GG_ALLOC_GRANULARITY
using oneAPI


##------
## TYPES

const oneField{T,N} = GGField{T,N,oneArray{T,N}}


##------------------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT

ImplicitGlobalGrid.is_loaded(::Val{:ImplicitGlobalGrid_ONEAPIExt}) = true
ImplicitGlobalGrid.is_functional(::Val{:oneAPI})                   = oneAPI.functional()


##-------------
## SYNTAX SUGAR

ImplicitGlobalGrid.is_onearray(A::oneArray) = true   #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

ImplicitGlobalGrid.wrap_field(A::oneArray, hw::Tuple) = oneField{eltype(A), ndims(A)}((A, hw))

Base.size(A::oneField)          = Base.size(A.A)
Base.size(A::oneField, args...) = Base.size(A.A, args...)
Base.length(A::oneField)        = Base.length(A.A)
Base.ndims(A::oneField)         = Base.ndims(A.A)
Base.eltype(A::oneField)        = Base.eltype(A.A)


##---------------
## oneAPI functions

function ImplicitGlobalGrid.register(::Type{<:oneArray},buf::Array{T}) where T <: GGNumber
    rbuf = oneAPI.Mem.register(oneAPI.Mem.Host, pointer(buf), sizeof(buf), oneAPI.Mem.HOSTREGISTER_DEVICEMAP);
    rbuf_d = convert(onePtr{T}, rbuf);
    return unsafe_wrap(oneArray, rbuf_d, size(buf)), rbuf;
end
