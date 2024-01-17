import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGArray, GGField, GGNumber
using AMDGPU


##------
## TYPES

const ROCField{T,N} = GGField{T,N,ROCArray{T,N}}


##-------------
## SYNTAX SUGAR

is_rocarray(A::GGArray) = typeof(A) <: ROCArray  #NOTE: this function is only to be used when multiple dispatch on the type of the array seems an overkill (in particular when only something needs to be done for the GPU case, but nothing for the CPU case) and as long as performance does not suffer.


##--------------------------------------------------------------------------------
## FUNCTIONS FOR WRAPPING ARRAYS AND FIELDS AND DEFINE ARRAY PROPERTY BASE METHODS

wrap_field(A::ROCArray, hw::Tuple) = ROCField{eltype(A), ndims(A)}((A, hw))

Base.size(A::ROCField)          = Base.size(A.A)
Base.size(A::ROCField, args...) = Base.size(A.A, args...)
Base.length(A::ROCField)        = Base.length(A.A)
Base.ndims(A::ROCField)         = Base.ndims(A.A)
Base.eltype(A::ROCField)        = Base.eltype(A.A)

##---------------
## AMDGPU functions

function register(::Type{<:ROCArray},buf::Array{T}) where T <: GGNumber
    return unsafe_wrap(ROCArray, pointer(buf), size(buf))
end
