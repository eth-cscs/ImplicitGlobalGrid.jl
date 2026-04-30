export activate_global_grid

"""
    activate_global_grid(new_gg)

Replaces the current active grid parameters with the ones provided by `new_gg`. Only one grid configuration can be active at a time. This function returns `nothing`; use [`active_global_grid()`](@ref) or [`get_global_grid()`](@ref) if you need to query the currently active grid.


# Argument
- `new_gg::GlobalGrid`: the global grid configuration to be set active. It must be a GlobalGrid returned by a call to `create_global_grid`.

# Usage example
    Given two local domains of different size and/or ghost cell properties: we have array `A1` and `B1` on grid `gg1`, and array `A2`, `B2` on grid `gg2`.
    
    activate_global_grid(A)       # Activate the first grid configuration
    update_halo!(A1, B1)  # Update the halo regions of arrays A1 and B1 on grid gg1
    activate_global_grid(gg2)       # Activate the second grid configuration
    update_halo!(smaller_array_B) # Update the halo regions of smaller_array_B according to configuration B

See also: [`init_global_grid`](@ref), [`create_global_grid`](@ref)
"""
function activate_global_grid(new_gg :: GlobalGrid) :: Nothing
    check_initialized()
    init_time = !grid_is_initialized()
    set_global_grid(new_gg)
    if init_time init_timing_functions() end
    return nothing
end