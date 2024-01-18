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
    check_initialized()
    if (cuda_enabled() && amdgpu_enabled()) error("Cannot select a device because both CUDA and AMDGPU are enabled (meaning that both modules were imported before ImplicitGlobalGrid).") end
    if cuda_enabled() || amdgpu_enabled()
        if cuda_enabled()
            @assert cuda_functional()
            nb_devices = nb_cudevices()
        elseif amdgpu_enabled()
            @assert amdgpu_functional()
            nb_devices = nb_rocdevices()
        end
        comm_l = MPI.Comm_split_type(comm(), MPI.COMM_TYPE_SHARED, me())
        if (MPI.Comm_size(comm_l) > nb_devices) error("More processes have been launched per node than there are GPUs available."); end
        me_l      = MPI.Comm_rank(comm_l)
        device_id = amdgpu_enabled() ? me_l+1 : me_l
        if     cuda_enabled()   cudevice!(device_id)
        elseif amdgpu_enabled() rocdevice!(device_id)
        end
        return device_id
    else
        error("Cannot select a device because neither CUDA nor AMDGPU is enabled (meaning that the corresponding module was not imported before ImplicitGlobalGrid).")
    end
end

_select_device() = select_device()
