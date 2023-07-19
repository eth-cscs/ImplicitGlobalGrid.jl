export select_device

"""
    select_device()

Select the device (GPU) corresponding to the node-local MPI rank and return its ID.

!!! note "device indexing"
    - CUDA.jl device indexing is 0-based.
    - AMDGPU.jl device indexing is 1-based.
    - The returned ID is therefore 0-based for CUDA and 1-based for AMDGPU.

See also: [`init_global_grid`](@ref)
"""
function select_device()
    if cuda_enabled() || amdgpu_enabled()
        check_initialized();
        if cuda_enabled()
            @assert CUDA.functional(true)
            nb_devices = length(CUDA.devices())
        elseif amdgpu_enabled()
            @assert AMDGPU.functional()
            nb_devices = length(AMDGPU.devices())
        end
        comm_l = MPI.Comm_split_type(comm(), MPI.COMM_TYPE_SHARED, me())
        if (MPI.Comm_size(comm_l) > nb_devices) error("More processes have been launched per node than there are GPUs available."); end
        me_l      = MPI.Comm_rank(comm_l)
        device_id = amdgpu_enabled() ? me_l+1 : me_l
        if     cuda_enabled()   CUDA.device!(device_id)
        elseif amdgpu_enabled() AMDGPU.device!(device_id)
        end
        return device_id
    else
        error("Cannot select a device because neither CUDA nor AMDGPU is enabled (possibly detected non functional when the ImplicitGlobalGrid module was loaded).")
    end
end

_select_device() = select_device()
