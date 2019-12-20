export finalize_global_grid

import MPI


"""
    finalize_global_grid()
    finalize_global_grid(;finalize_MPI=true)

Finalize the global grid (and also MPI by default).

# Arguments
!!! note "Advanced keyword arguments"
    - `finalize_MPI::Bool=true`: whether to finalize MPI (`true`) or not (`false`).

See also: [`init_global_grid`](@ref)
"""
function finalize_global_grid(;finalize_MPI::Bool=true)
    check_initialized();
    #TODO
    free_gather_buffer();
    free_update_halo_buffers();
    if (finalize_MPI)
        if (!MPI.Initialized()) error("MPI cannot be finalized as it has not been initialized. "); end  # This case should never occur as init_global_grid() must enforce that after a call to it, MPI is always initialized.
        if (MPI.Finalized()) error("MPI is already finalized. Set the argument 'finalize_MPI=false'."); end
        MPI.Finalize();
    end
    set_global_grid(GLOBAL_GRID_NULL);
    GC.gc();
    return nothing
end
