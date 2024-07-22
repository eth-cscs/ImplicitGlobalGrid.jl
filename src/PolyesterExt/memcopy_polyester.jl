import ImplicitGlobalGrid
import ImplicitGlobalGrid: GGNumber
using Polyester

function ImplicitGlobalGrid.memcopy_polyester!(dst::AbstractArray{T}, src::AbstractArray{T}) where T <: GGNumber
    @info "Using PolyesterExt for memory copy"
    @batch for i ∈ eachindex(dst, src)  # NOTE: @batch will use maximally Threads.nthreads() threads / #cores threads. Set the number of threads e.g. as: export JULIA_NUM_THREADS=12. NOTE on previous implementation with LoopVectorization: tturbo fails if src_flat and dst_flat are used due to an issue in ArrayInterface : https://github.com/JuliaArrays/ArrayInterface.jl/issues/228 TODO: once the package has matured check again if there is any benefit with: per=core stride=true
        @inbounds dst[i] = src[i]       # NOTE: We fix here exceptionally the use of @inbounds as this copy between two flat vectors (which must have the right length) is considered safe.
    end
end