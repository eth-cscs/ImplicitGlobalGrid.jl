export select_gpu

import MPI
@static if ENABLE_CUDA
    using CUDA
end

"""
    select_gpu()

Select the GPU corresponding to the node-local MPI rank and return its ID. This function only needs to be called when using nodes with more than one GPU.
"""
function select_gpu()
    @static if ENABLE_CUDA
        @assert CUDA.functional(true)
        comm_l = MPI.Comm_split_type(comm(), MPI.MPI_COMM_TYPE_SHARED, me())
    	if (MPI.Comm_size(comm_l) > length(CUDA.devices())) error("More processes have been launched per node than there are GPUs available."); end
    	me_l = MPI.Comm_rank(comm_l)
        CUDA.device!(me_l)
        return me_l
    else
        error("Cannot select a GPU because ImplicitGlobalGrid was not precompiled for GPU usage (as CUDA was not functional).")
    end
end
