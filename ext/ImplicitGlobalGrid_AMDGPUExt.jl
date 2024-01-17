module ImplicitGlobalGrid_AMDGPUExt
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "update_halo.jl"))
end