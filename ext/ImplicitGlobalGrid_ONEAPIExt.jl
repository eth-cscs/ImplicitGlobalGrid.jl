module ImplicitGlobalGrid_INTELExt
    include(joinpath(@__DIR__, "..", "src", "ONEAPIExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "ONEAPIExt", "select_device.jl"))
    include(joinpath(@__DIR__, "..", "src", "ONEAPIExt", "update_halo.jl"))
end
