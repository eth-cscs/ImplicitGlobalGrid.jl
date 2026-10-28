import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber, halosize, ol, amdgpuaware_MPI, sendranges, recvranges, sendbuf_flat, recvbuf_flat, write_d2x!, read_x2d!, write_d2x_async!, read_x2d_async!, register, is_rocarray
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

function ImplicitGlobalGrid.register(::Type{<:ROCArray},buf::AbstractArray{T}) where T <: GGNumber  # NOTE: buf is a host buffer, i.e. an Array or a reinterpretation of it (see reinterpret_bufs).
    if hostbuf_pinned(buf) error("the host buffer to be registered for the GPU is already page-locked (pinned) host memory, which is not supported.") end  # NOTE: the wrapping ROCArray would not track such a registration and could hipHostFree the memory when freed (own=true).
    return unsafe_wrap(ROCArray, pointer(buf), size(buf); own=true) # NOTE: the host memory is page-locked (hipHostRegister) and unregistered when the wrapping ROCArray is freed (own=true); AMDGPU may release the registration only once the last wrapper of the memory is finalized and the device is done with it, i.e. the caller must keep buf alive until then (see unregister_rocbufs).
end

hostbuf_pinned(buf::AbstractArray)     = isdefined(AMDGPU.Mem, :is_pinned)     && AMDGPU.Mem.is_pinned(Ptr{Cvoid}(pointer(buf)))      # NOTE: the memory state can only be queried with AMDGPU internals; if they are not available, no check is done.
hostbuf_registered(buf::AbstractArray) = !isdefined(AMDGPU.Mem, :is_registered) || AMDGPU.Mem.is_registered(Ptr{Cvoid}(pointer(buf)))  # NOTE: if the registration state cannot be queried, a buffer is considered registered (i.e. it is kept alive until the end of the session).
