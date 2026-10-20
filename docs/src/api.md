```@meta
CurrentModule = ImplicitGlobalGrid
```

# API reference

This is the official API reference of ImplicitGlobalGrid. Note that it can also be queried interactively from the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) using the [help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode):
```julia-repl
julia> using ImplicitGlobalGrid
julia>?
help?> ImplicitGlobalGrid
```

```@index
Pages = ["api.md"]
```

## Module
```@docs
ImplicitGlobalGrid
```

## Initialization and finalization
```@docs
init_global_grid
finalize_global_grid
```

## Halo update
```@docs
update_halo!
```

## Multiple grids
```@docs
create_global_grid
activate_global_grid
get_global_grid
```

## Device selection
```@docs
select_device
```

## Global grid size, coordinates and indices
```@docs
nx_g
ny_g
nz_g
x_g
y_g
z_g
ix_g
iy_g
iz_g
```

## Extents and metadata grids
```@docs
extents
extents_g
metagrid
```

## Gathering
```@docs
gather!
```

## Timing
```@docs
tic
toc
```
