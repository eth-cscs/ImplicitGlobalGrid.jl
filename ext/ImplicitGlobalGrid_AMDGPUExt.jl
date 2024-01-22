module ImplicitGlobalGrid_AMDGPUExt
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "select_device.jl"))
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "update_halo.jl"))
end