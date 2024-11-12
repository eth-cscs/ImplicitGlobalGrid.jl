import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber, halosize, ol, cudaaware_MPI, sendranges, recvranges, sendbuf_flat, recvbuf_flat, write_d2x!, read_x2d!, write_d2h_async!, read_h2d_async!, register, is_cuarray
import ImplicitGlobalGrid: NNEIGHBORS_PER_DIM, GG_ALLOC_GRANULARITY
using CUDA


##------
## TYPES

const CuField{T,N} = GGField{T,N,CuArray{T,N}}


##------------------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT

ImplicitGlobalGrid.is_loaded(::Val{:ImplicitGlobalGrid_CUDAExt}) = true
ImplicitGlobalGrid.is_functional(::Val{:CUDA})                   = CUDA.functional()


##-------------
## SYNTAX SUGAR

ImplicitGlobalGrid.is_cuarray(A::CuArray) = true   #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


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

function ImplicitGlobalGrid.register(::Type{<:CuArray},buf::Array{T}) where T <: GGNumber
    rbuf = CUDA.register(CUDA.HostMemory, pointer(buf), sizeof(buf), CUDA.MEMHOSTREGISTER_DEVICEMAP);
    rbuf_d = convert(CuPtr{T}, rbuf);
    return unsafe_wrap(CuArray, rbuf_d, size(buf)), rbuf;
end
