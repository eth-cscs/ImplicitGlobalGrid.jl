import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber, halosize, ol, amdgpuaware_MPI, sendranges, recvranges, sendbuf_flat, recvbuf_flat, write_d2x!, read_x2d!, write_d2h_async!, read_h2d_async!, register, is_rocarray
import ImplicitGlobalGrid: NNEIGHBORS_PER_DIM, GG_ALLOC_GRANULARITY
import LinearAlgebra
using AMDGPU


##------
## TYPES

const AnyROCArray{T,N} = Union{Base.LogicalIndex{T, <:ROCArray}, Base.ReinterpretArray{T, N, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s14"}, var"#s14"}} where var"#s14"<:ROCArray, Base.ReshapedArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}}, SubArray{<:Any, <:Any, var"#s15"}, var"#s15"}} where var"#s15"<:ROCArray, SubArray{T, N, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, Base.ReshapedArray{<:Any, <:Any, <:Union{Base.ReinterpretArray{<:Any, <:Any, <:Any, <:Union{SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, SubArray{<:Any, <:Any, var"#s16"}, var"#s16"}}, var"#s16"}} where var"#s16"<:ROCArray, ROCArray{T, N}, LinearAlgebra.Adjoint{T, <:ROCArray{T, N}}, LinearAlgebra.Diagonal{T, <:ROCArray{T, N}}, LinearAlgebra.LowerTriangular{T, <:ROCArray{T, N}}, LinearAlgebra.Symmetric{T, <:ROCArray{T, N}}, LinearAlgebra.Transpose{T, <:ROCArray{T, N}}, LinearAlgebra.Tridiagonal{T, <:ROCArray{T, N}}, LinearAlgebra.UnitLowerTriangular{T, <:ROCArray{T, N}}, LinearAlgebra.UnitUpperTriangular{T, <:ROCArray{T, N}}, LinearAlgebra.UpperTriangular{T, <:ROCArray{T, N}}, PermutedDimsArray{T, N, <:Any, <:Any, <:ROCArray}} where {T, N}  # NOTE: This is done in analogy with CUDA.AnyCuArray.
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
