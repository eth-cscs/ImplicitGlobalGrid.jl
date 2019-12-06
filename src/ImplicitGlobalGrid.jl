module ImplicitGlobalGrid

## Include of shared constant parameters, types and syntax sugar
include("shared.jl")

## Alphabetical include of files:
include("finalize_global_grid.jl")
include("gather.jl")
include("init_global_grid.jl")
include("tools.jl")
include("update_halo.jl")

end
