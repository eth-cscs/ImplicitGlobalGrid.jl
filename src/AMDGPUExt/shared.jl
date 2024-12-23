import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber, halosize, ol, amdgpuaware_MPI, sendranges, recvranges, sendbuf_flat, recvbuf_flat, write_d2x!, read_x2d!, write_d2h_async!, read_h2d_async!, register, is_rocarray
import ImplicitGlobalGrid: NNEIGHBORS_PER_DIM, GG_ALLOC_GRANULARITY
using AMDGPU
import AMDGPU: AnyROCArray


##------
## TYPES

const ROCField{T,N} = GGField{T,N,<:AnyROCArray{T,N}}


##------------------------------------
## HANDLING OF CUDA AND AMDGPU SUPPORT

ImplicitGlobalGrid.is_loaded(::Val{:ImplicitGlobalGrid_AMDGPUExt}) = true
ImplicitGlobalGrid.is_functional(::Val{:AMDGPU})                   = AMDGPU.functional()


##-------------
## SYNTAX SUGAR

ImplicitGlobalGrid.is_rocarray(A::AnyROCArray) = true  #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

Base.size(A::ROCField, args...) = Base.size(A.A, args...)
Base.length(A::ROCField)        = Base.length(A.A)
Base.ndims(A::ROCField)         = Base.ndims(A.A)
Base.eltype(A::ROCField)        = Base.eltype(A.A)

##---------------
## AMDGPU functions

function ImplicitGlobalGrid.register(::Type{<:ROCArray},buf::Array{T}) where T <: GGNumber
    return unsafe_wrap(ROCArray, pointer(buf), size(buf))
end
