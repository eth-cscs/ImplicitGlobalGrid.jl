module ImplicitGlobalGrid_CUDAExt
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "update_halo.jl"))
end