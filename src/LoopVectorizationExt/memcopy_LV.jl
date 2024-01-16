import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGNumber
using LoopVectorization

function ImplicitGlobalGrid.memcopy_loopvect!(dst::AbstractArray{T}, src::AbstractArray{T}) where T <: GGNumber
    @tturbo for i ∈ eachindex(dst, src)  # NOTE: tturbo will use maximally Threads.nthreads() threads. Set the number of threads e.g. as: export JULIA_NUM_THREADS=12. NOTE: tturbo fails if src_flat and dst_flat are used due to an issue in ArrayInterface : https://github.com/JuliaArrays/ArrayInterface.jl/issues/228
        @inbounds dst[i] = src[i]        # NOTE: We fix here exceptionally the use of @inbounds (currently anyways done by LoopVectorization) as this copy between two flat vectors (which must have the right length) is considered safe.
    end
end