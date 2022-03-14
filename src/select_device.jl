export select_device

"""
    select_device()

Select the device (GPU) corresponding to the node-local MPI rank and return its ID.

!!! note "device indexing"
	- `CUDA.device!()` is 0-based.
    - `AMDGPU.device!()` is 1-based.
    - The returned ID is 0-based (node-local MPI rank).

See also: [`init_global_grid`](@ref)
"""
function select_device()
	if cuda_enabled() || amdgpu_enabled()
	    check_initialized();
	    if cuda_enabled()
			@assert CUDA.functional(true)
			nb_devices = length(CUDA.devices())
		elseif amdgpu_enabled()
			@assert AMDGPU.functional() # DEBUG: AMDGPU takes componant as args. See https://github.com/JuliaGPU/AMDGPU.jl/blob/3fe5af69269cdab2ccaf296340d1dc390ad03a6e/src/utils.jl#L108 
			nb_devices = length(AMDGPU.get_agents(:gpu))
		end
	    comm_l = MPI.Comm_split_type(comm(), MPI.MPI_COMM_TYPE_SHARED, me())
		if (MPI.Comm_size(comm_l) > nb_devices) error("More processes have been launched per node than there are GPUs available."); end
		me_l = MPI.Comm_rank(comm_l)
	    if     cuda_enabled()   CUDA.device!(me_l)
		elseif amdgpu_enabled()	AMDGPU.device!(me_l+1)
		end
	    return me_l
	else
		error("Cannot select a device because neither CUDA nor AMDGPU is enabled (possibly detected non functional when the ImplicitGlobalGrid module was loaded).")
	end
end

_select_device() = select_device()
