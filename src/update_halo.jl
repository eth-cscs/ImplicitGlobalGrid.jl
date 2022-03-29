export update_halo!

"""
    update_halo!(A)
    update_halo!(A...)

Update the halo of the given GPU/CPU-array(s).

# Typical use cases:
    update_halo!(A)        # Update the halo of the array A.
    update_halo!(A, B, C)  # Update the halos of the arrays A, B and C.

!!! note "Performance note"
    Group subsequent calls to `update_halo!` in a single call for better performance (enables additional pipelining).
    Consider activating CUDA-aware MPI for Nvidia GPUs or ROCm-aware MPI for AMD GPUs (see [`ImplicitGlobalGrid`](@ref)).
"""
function update_halo!(A::GGArray...)
    check_initialized();
    check_fields(A...);
    _update_halo!(A...);  # Asignment of A to fields in the internal function _update_halo!() as vararg A can consist of multiple fields; A will be used for a single field in the following (The args of update_halo! must however be "A..." for maximal simplicity and elegance for the user).
    return nothing
end

function _update_halo!(fields::GGArray...)
    if (any_cuarray(fields...) && !cuda_enabled())    error("CUDA is not enabled (possibly detected non functional when the ImplicitGlobalGrid module was loaded)."); end    #NOTE: in the following, it is only required to check for `cuda_enabled()` when the context does not imply `any_cuarray(fields...)` or `is_cuarray(A)`.
    if (any_rocarray(fields...) && !amdgpu_enabled()) error("AMDGPU is not enabled (possibly detected non functional when the ImplicitGlobalGrid module was loaded)."); end  #NOTE: in the following, it is only required to check for `amdgpu_enabled()` when the context does not imply `any_rocarray(fields...)` or `is_rocarray(A)`.
    allocate_bufs(fields...);
    if any_array(fields...) allocate_tasks(fields...); end
    if any_cuarray(fields...) allocate_custreams(fields...); end
    if any_rocarray(fields...) allocate_rocqueues(fields...); end

    for dim = 1:NDIMS_MPI  # NOTE: this works for 1D-3D (e.g. if nx>1, ny>1 and nz=1, then for d=3, there will be no neighbors, i.e. nothing will be done as desired...).
        for ns = 1:NNEIGHBORS_PER_DIM,  i = 1:length(fields)
            if has_neighbor(ns, dim) iwrite_sendbufs!(ns, dim, fields[i], i); end
        end
        # Send / receive if the neighbors are other processes (usual case).
        reqs = fill(MPI.REQUEST_NULL, length(fields), NNEIGHBORS_PER_DIM, 2);
        if all(neighbors(dim) .!= me())  # Note: handling of send/recv to itself requires special configurations for some MPI implementations (e.g. self BTL must be activated with OpenMPI); so we handle this case without MPI to avoid this complication.
            for nr = NNEIGHBORS_PER_DIM:-1:1,  i = 1:length(fields) # Note: if there were indeed more than 2 neighbors per dimension; then one would need to make sure which neigbour would communicate with which.
                if has_neighbor(nr, dim) reqs[i,nr,1] = irecv_halo!(nr, dim, fields[i], i); end
            end
            for ns = 1:NNEIGHBORS_PER_DIM,  i = 1:length(fields)
                if has_neighbor(ns, dim)
                    wait_iwrite(ns, fields[i], i);  # Right before starting to send, make sure that the data of neighbor ns and field i has finished writing to the sendbuffer.
                    reqs[i,ns,2] = isend_halo(ns, dim, fields[i], i);
                end
            end
        # Copy if I am my own neighbors (when periodic boundary and only one process in this dimension).
        elseif all(neighbors(dim) .== me())
            for ns = 1:NNEIGHBORS_PER_DIM,  i = 1:length(fields)
                wait_iwrite(ns, fields[i], i);  # Right before starting to send, make sure that the data of neighbor ns and field i has finished writing to the sendbuffer.
                sendrecv_halo_local(ns, dim, fields[i], i);
                nr = NNEIGHBORS_PER_DIM - ns + 1;
                iread_recvbufs!(nr, dim, fields[i], i);
            end
        else
            error("Incoherent neighbors in dimension $dim: either all neighbors must equal to me, or none.")
        end
        for nr = NNEIGHBORS_PER_DIM:-1:1,  i = 1:length(fields)  # Note: if there were indeed more than 2 neighbors per dimension; then one would need to make sure which neigbour would communicate with which.
            if (reqs[i,nr,1]!=MPI.REQUEST_NULL) MPI.Wait!(reqs[i,nr,1]); end
            if (has_neighbor(nr, dim) && neighbor(nr, dim)!=me()) iread_recvbufs!(nr, dim, fields[i], i); end  # Note: if neighbor(nr,dim) != me() is done directly in the sendrecv_halo_local loop above for better performance (thanks to pipelining)
        end
        for nr = NNEIGHBORS_PER_DIM:-1:1,  i = 1:length(fields) # Note: if there were indeed more than 2 neighbors per dimension; then one would need to make sure which neigbour would communicate with which.
            if has_neighbor(nr, dim) wait_iread(nr, fields[i], i); end
        end
        for ns = 1:NNEIGHBORS_PER_DIM
            if (any(reqs[:,ns,2].!=[MPI.REQUEST_NULL])) MPI.Waitall!(reqs[:,ns,2]); end
        end
    end
end


##---------------------------
## FUNCTIONS FOR SYNTAX SUGAR

halosize(dim::Integer, A::GGArray) = (ndims(A)>1) ? size(A)[1:ndims(A).!=dim] : (1,);


##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

let
    global free_update_halo_buffers, allocate_bufs, sendbuf, recvbuf, sendbuf_flat, recvbuf_flat, gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat, rocsendbuf, rocrecvbuf, rocsendbuf_flat, rocrecvbuf_flat
    sendbufs_raw = nothing
    recvbufs_raw = nothing
    cusendbufs_raw = nothing
    curecvbufs_raw = nothing
    cusendbufs_raw_h = nothing
    curecvbufs_raw_h = nothing
    rocsendbufs_raw = nothing
    rocrecvbufs_raw = nothing
    rocsendbufs_raw_h = nothing
    rocrecvbufs_raw_h = nothing

    function free_update_halo_buffers()
        if (cuda_enabled() && any(cudaaware_MPI())) free_gpubufs(cusendbufs_raw) end
        if (cuda_enabled() && any(cudaaware_MPI())) free_gpubufs(curecvbufs_raw) end
        if (cuda_enabled() && none(cudaaware_MPI())) unregister_gpubufs(cusendbufs_raw_h) end
        if (cuda_enabled() && none(cudaaware_MPI())) unregister_gpubufs(curecvbufs_raw_h) end
        if (amdgpu_enabled() && any(amdgpuaware_MPI())) free_gpubufs(rocsendbufs_raw) end
        if (amdgpu_enabled() && any(amdgpuaware_MPI())) free_gpubufs(rocrecvbufs_raw) end
        if (amdgpu_enabled() && none(amdgpuaware_MPI())) unregister_gpubufs(rocsendbufs_raw_h) end
        if (amdgpu_enabled() && none(amdgpuaware_MPI())) unregister_gpubufs(rocrecvbufs_raw_h) end
        sendbufs_raw = nothing
        recvbufs_raw = nothing
        cusendbufs_raw = nothing
        curecvbufs_raw = nothing
        cusendbufs_raw_h = nothing
        curecvbufs_raw_h = nothing
        rocsendbufs_raw = nothing
        rocrecvbufs_raw = nothing
        rocsendbufs_raw_h = nothing
        rocrecvbufs_raw_h = nothing
        GC.gc()
    end


    # (CUDA, AMDGPU functions)

    function free_gpubufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if is_cuarray(bufs[i][n])  CUDA.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end
                    # if is_rocarray(bufs[i][n]) AMDGPU.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end # DEBUG: One would need to have undafe_free to check if own==true and return without doing anything in that case https://github.com/JuliaGPU/AMDGPU.jl/blob/f836632a5370b9b791c825df3525457688b5e180/src/array.jl#L77-L81
                end
            end
        end
    end

    function unregister_gpubufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if (isa(bufs[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(bufs[i][n]); bufs[i][n] = []; end
                    if (isa(bufs[i][n],AMDGPU.Mem.Buffer))   AMDGPU.Mem.unlock(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    # Allocate for each field two send and recv buffers (one for the left and one for the right neighbour of a dimension). The required length of the buffer is given by the maximal number of halo elements in any of the dimensions. Note that buffers are not allocated separately for each dimension, as the updates are performed one dimension at a time (required for correctness).
    function allocate_bufs(fields::GGArray{T}...) where T <: GGNumber
        if (isnothing(sendbufs_raw) || isnothing(recvbufs_raw))
            free_update_halo_buffers();
            init_bufs_arrays();
            if cuda_enabled() init_cubufs_arrays(); end
            if amdgpu_enabled() init_rocbufs_arrays(); end
        end
        init_bufs(T, fields...);
        if cuda_enabled() init_cubufs(T, fields...); end
        if amdgpu_enabled() init_rocbufs(T, fields...); end
        for i = 1:length(fields)
            A = fields[i];
            for n = 1:NNEIGHBORS_PER_DIM # Ensure that the buffers are interpreted to contain elements of the same type as the array.
                reinterpret_bufs(T, i, n);
                if cuda_enabled() reinterpret_cubufs(T, i, n); end
                if amdgpu_enabled() reinterpret_rocbufs(T, i, n); end
            end
            max_halo_elems = (ndims(A) > 1) ? prod(sort([size(A)...])[2:end]) : 1;
            if (length(sendbufs_raw[i][1]) < max_halo_elems)
                for n = 1:NNEIGHBORS_PER_DIM
                    reallocate_bufs(T, i, n, max_halo_elems);
                    if (is_cuarray(A) && none(cudaaware_MPI())) reregister_cubufs(T, i, n); end  # Host memory is page-locked (and mapped to device memory) to ensure optimal access performance (from kernel or with 3-D memcopy).
                    if (is_rocarray(A) && none(amdgpuaware_MPI())) reregister_rocbufs(T, i, n); end  # ...
                end
                GC.gc(); # Too small buffers had been replaced with larger ones; free the now unused memory.
            end
            if (!isnothing(cusendbufs_raw) && length(cusendbufs_raw[i][1]) < max_halo_elems)
                for n = 1:NNEIGHBORS_PER_DIM
                    if (is_cuarray(A) &&  any(cudaaware_MPI())) reallocate_cubufs(T, i, n, max_halo_elems); GC.gc(); end # Too small buffers had been replaced with larger ones; free the unused memory immediately.
                end
            end
            if (!isnothing(rocsendbufs_raw) && length(rocsendbufs_raw[i][1]) < max_halo_elems)
                for n = 1:NNEIGHBORS_PER_DIM
                    if (is_rocarray(A) &&  any(amdgpuaware_MPI())) reallocate_rocbufs(T, i, n, max_halo_elems); GC.gc(); end # Too small buffers had been replaced with larger ones; free the unused memory immediately.
                end
            end
        end
    end


    # (CPU functions)

    function init_bufs_arrays()
        sendbufs_raw = Array{Array{Any,1},1}();
        recvbufs_raw = Array{Array{Any,1},1}();
    end

    function init_bufs(T::DataType, fields::GGArray...)
        while (length(sendbufs_raw) < length(fields)) push!(sendbufs_raw, [zeros(T,0), zeros(T,0)]); end
        while (length(recvbufs_raw) < length(fields)) push!(recvbufs_raw, [zeros(T,0), zeros(T,0)]); end
    end

    function reinterpret_bufs(T::DataType, i::Integer, n::Integer)
        if (eltype(sendbufs_raw[i][n]) != T) sendbufs_raw[i][n] = reinterpret(T, sendbufs_raw[i][n]); end
        if (eltype(recvbufs_raw[i][n]) != T) recvbufs_raw[i][n] = reinterpret(T, recvbufs_raw[i][n]); end
    end

    function reallocate_bufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        sendbufs_raw[i][n] = zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        recvbufs_raw[i][n] = zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end


    # (CUDA functions)

    function init_cubufs_arrays()
        cusendbufs_raw = Array{Array{Any,1},1}();
        curecvbufs_raw = Array{Array{Any,1},1}();
        cusendbufs_raw_h = Array{Array{Any,1},1}();
        curecvbufs_raw_h = Array{Array{Any,1},1}();
    end

    function init_cubufs(T::DataType, fields::GGArray...)
        while (length(cusendbufs_raw) < length(fields)) push!(cusendbufs_raw, [CuArray{T}(undef,0), CuArray{T}(undef,0)]); end
        while (length(curecvbufs_raw) < length(fields)) push!(curecvbufs_raw, [CuArray{T}(undef,0), CuArray{T}(undef,0)]); end
        while (length(cusendbufs_raw_h) < length(fields)) push!(cusendbufs_raw_h, [[], []]); end
        while (length(curecvbufs_raw_h) < length(fields)) push!(curecvbufs_raw_h, [[], []]); end
    end

    function reinterpret_cubufs(T::DataType, i::Integer, n::Integer)
        if (eltype(cusendbufs_raw[i][n]) != T) cusendbufs_raw[i][n] = reinterpret(T, cusendbufs_raw[i][n]); end
        if (eltype(curecvbufs_raw[i][n]) != T) curecvbufs_raw[i][n] = reinterpret(T, curecvbufs_raw[i][n]); end
    end

    function reallocate_cubufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        cusendbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        curecvbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_cubufs(T::DataType, i::Integer, n::Integer)
        if (isa(cusendbufs_raw_h[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(cusendbufs_raw_h[i][n]); cusendbufs_raw_h[i][n] = []; end # It is always initialized registered... if (cusendbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        if (isa(curecvbufs_raw_h[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(curecvbufs_raw_h[i][n]); curecvbufs_raw_h[i][n] = []; end # BUG? It is always initialized registered... if (curecvbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        cusendbufs_raw[i][n], cusendbufs_raw_h[i][n] = register(CuArray,sendbufs_raw[i][n]);
        curecvbufs_raw[i][n], curecvbufs_raw_h[i][n] = register(CuArray,recvbufs_raw[i][n]);
    end


    # (AMDGPU functions)

    function init_rocbufs_arrays()
        rocsendbufs_raw = Array{Array{Any,1},1}();
        rocrecvbufs_raw = Array{Array{Any,1},1}();
        rocsendbufs_raw_h = Array{Array{Any,1},1}();
        rocrecvbufs_raw_h = Array{Array{Any,1},1}();
    end

    function init_rocbufs(T::DataType, fields::GGArray...)
        while (length(rocsendbufs_raw) < length(fields)) push!(rocsendbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        while (length(rocrecvbufs_raw) < length(fields)) push!(rocrecvbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        while (length(rocsendbufs_raw_h) < length(fields)) push!(rocsendbufs_raw_h, [[], []]); end
        while (length(rocrecvbufs_raw_h) < length(fields)) push!(rocrecvbufs_raw_h, [[], []]); end
    end

    function reinterpret_rocbufs(T::DataType, i::Integer, n::Integer)
        if (eltype(rocsendbufs_raw[i][n]) != T) rocsendbufs_raw[i][n] = reinterpret(T, rocsendbufs_raw[i][n]); end
        if (eltype(rocrecvbufs_raw[i][n]) != T) rocrecvbufs_raw[i][n] = reinterpret(T, rocrecvbufs_raw[i][n]); end
    end

    function reallocate_rocbufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        rocsendbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        rocrecvbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_rocbufs(T::DataType, i::Integer, n::Integer)
        if (isa(rocsendbufs_raw_h[i][n],AMDGPU.Mem.Buffer)) AMDGPU.Mem.unlock(rocsendbufs_raw_h[i][n]); rocsendbufs_raw_h[i][n] = []; end
        if (isa(rocrecvbufs_raw_h[i][n],AMDGPU.Mem.Buffer)) AMDGPU.Mem.unlock(rocrecvbufs_raw_h[i][n]); rocrecvbufs_raw_h[i][n] = []; end
        rocsendbufs_raw[i][n], rocsendbufs_raw_h[i][n] = register(ROCArray,sendbufs_raw[i][n]); # DEBUG: register no yet implemented
        rocrecvbufs_raw[i][n], rocrecvbufs_raw_h[i][n] = register(ROCArray,recvbufs_raw[i][n]); # DEBUG: register no yet implemented
    end


    # (CPU functions)

    function sendbuf_flat(n::Integer, dim::Integer, i::Integer, A::GGArray{T}) where T <: GGNumber
        return view(sendbufs_raw[i][n]::AbstractVector{T},1:prod(halosize(dim,A)));
    end

    function recvbuf_flat(n::Integer, dim::Integer, i::Integer, A::GGArray{T}) where T <: GGNumber
        return view(recvbufs_raw[i][n]::AbstractVector{T},1:prod(halosize(dim,A)));
    end

    function sendbuf(n::Integer, dim::Integer, i::Integer, A::GGArray)
        return reshape(sendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function recvbuf(n::Integer, dim::Integer, i::Integer, A::GGArray)
        return reshape(recvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # (CUDA functions)

    function gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
        return view(cusendbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
    end

    function gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
        return view(curecvbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
    end


    # (AMDGPU functions)

    function gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCArray{T}) where T <: GGNumber
        return view(rocsendbufs_raw[i][n]::ROCVector{T},1:prod(halosize(dim,A)));
    end

    function gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCArray{T}) where T <: GGNumber
        return view(rocrecvbufs_raw[i][n]::ROCVector{T},1:prod(halosize(dim,A)));
    end


    # (GPU functions)

    #TODO: see if remove T here and in other cases for CuArray, ROCArray or Array (but then it does not verify that CuArray/ROCArray is of type GGNumber) or if I should instead change GGArray to GGArrayUnion and create: GGArray = Array{T} where T <: GGNumber  and  GGCuArray = CuArray{T} where T <: GGNumber; This is however more difficult to read and understand for others.
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::Union{CuArray{T}, ROCArray{T}}) where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::Union{CuArray{T}, ROCArray{T}}) where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_sendbufs_raw, get_recvbufs_raw, get_cusendbufs_raw, get_curecvbufs_raw, get_rocsendbufs_raw, get_rocrecvbufs_raw
    get_sendbufs_raw()    = deepcopy(sendbufs_raw)
    get_recvbufs_raw()    = deepcopy(recvbufs_raw)
    get_cusendbufs_raw()  = deepcopy(cusendbufs_raw)
    get_curecvbufs_raw()  = deepcopy(curecvbufs_raw)
    get_rocsendbufs_raw() = deepcopy(rocsendbufs_raw)
    get_rocrecvbufs_raw() = deepcopy(rocrecvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

# NOTE: the tasks, custreams and rocqueues are stored here in a let clause to have them survive the end of a call to update_boundaries. This avoids the creation of new tasks and cuda streams every time. Besides, that this could be relevant for performance, it is important for debugging the overlapping the communication with computation (if at every call new stream/task objects are created this becomes very messy and hard to analyse).


# (CPU functions)

function allocate_tasks(fields::GGArray...)
    allocate_tasks_iwrite(fields...);
    allocate_tasks_iread(fields...);
end

let
    global iwrite_sendbufs!, allocate_tasks_iwrite, wait_iwrite

    tasks = Array{Task}(undef, NNEIGHBORS_PER_DIM, 0);

    wait_iwrite(n::Integer, A::Array{T}, i::Integer) where T <: GGNumber = (schedule(tasks[n,i]); wait(tasks[n,i]);) # The argument A is used for multiple dispatch. #NOTE: The current implementation only starts a task when it is waited for, in order to make sure that only one task is run at a time and that they are run in the desired order (best for performance as the tasks are mapped only to one thread via context switching).

    function allocate_tasks_iwrite(fields::GGArray...)
        if length(fields) > size(tasks,2)  # Note: for simplicity, we create a tasks for every field even if it is not an Array
            tasks = [tasks Array{Task}(undef, NNEIGHBORS_PER_DIM, length(fields)-size(tasks,2))];  # Create (additional) emtpy tasks.
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, A::Array{T}, i::Integer) where T <: GGNumber  # Function to be called if A is a CPU array.
        tasks[n,i] = @task begin
            if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
                write_h2h!(sendbuf(n,dim,i,A), A, sendranges(n,dim,A), dim);
            end
        end
    end

    # Make tasks accessible for unit testing.
    global get_tasks_iwrite
    get_tasks_iwrite() = deepcopy(tasks)
end

let
    global iread_recvbufs!, allocate_tasks_iread, wait_iread

    tasks = Array{Task}(undef, NNEIGHBORS_PER_DIM, 0);

    wait_iread(n::Integer, A::Array{T}, i::Integer) where T <: GGNumber = (schedule(tasks[n,i]); wait(tasks[n,i]);) #NOTE: The current implementation only starts a task when it is waited for, in order to make sure that only one task is run at a time and that they are run in the desired order (best for performance currently as the tasks are mapped only to one thread via context switching).

    function allocate_tasks_iread(fields::GGArray...)
        if length(fields) > size(tasks,2)  # Note: for simplicity, we create a tasks for every field even if it is not an Array
            tasks = [tasks Array{Task}(undef, NNEIGHBORS_PER_DIM, length(fields)-size(tasks,2))];  # Create (additional) emtpy tasks.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, A::Array{T}, i::Integer) where T <: GGNumber
        tasks[n,i] = @task begin
            if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
                read_h2h!(recvbuf(n,dim,i,A), A, recvranges(n,dim,A), dim);
            end
        end
    end

    # Make tasks accessible for unit testing.
    global get_tasks_iread
    get_tasks_iread() = deepcopy(tasks)
end


# (CUDA functions)

function allocate_custreams(fields::GGArray...)
    allocate_custreams_iwrite(fields...);
    allocate_custreams_iread(fields...);
end

let
    global iwrite_sendbufs!, allocate_custreams_iwrite, wait_iwrite

    custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iwrite(n::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber = synchronize(custreams[n,i]);

    function allocate_custreams_iwrite(fields::GGArray...)
        if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuArray
            custreams = [custreams [CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber
        if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
            if dim == 1 || cudaaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = sendranges(n, dim, A);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] write_d2x!(gpusendbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim);
            else
                write_d2h_async!(sendbuf_flat(n,dim,i,A), A, sendranges(n,dim,A), dim, custreams[n,i]);
            end
        end
    end
end

let
    global iread_recvbufs!, allocate_custreams_iread, wait_iread

    custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iread(n::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber = synchronize(custreams[n,i]);

    function allocate_custreams_iread(fields::GGArray...)
        if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuArray
            custreams = [custreams [CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber
        if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
            if dim == 1 || cudaaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = recvranges(n, dim, A);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] read_x2d!(gpurecvbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim);
            else
                read_h2d_async!(recvbuf_flat(n,dim,i,A), A, recvranges(n,dim,A), dim, custreams[n,i]);
            end
        end
    end
end


# (AMDGPU functions)

function allocate_rocqueues(fields::GGArray...)
    allocate_rocqueues_iwrite(fields...);
    allocate_rocqueues_iread(fields...);
end

let
    global iwrite_sendbufs!, allocate_rocqueues_iwrite, wait_iwrite

    rocqueues  = Array{AMDGPU.HSAQueue}(undef, NNEIGHBORS_PER_DIM, 0)
    rocsignals = Array{Union{AMDGPU.HSASignal,AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal},Missing}}(undef, NNEIGHBORS_PER_DIM, 0)

    function wait_iwrite(n::Integer, A::ROCArray{T}, i::Integer) where T <: GGNumber
        if !ismissing(rocsignals[n,i]) # DEBUG: tmp solution to avoid rocsignals array access filing when accessing an unset signal
            wait(rocsignals[n,i]);
            rocsignals[n,i] = missing;
        end
    end

    function allocate_rocqueues_iwrite(fields::GGArray...)
        if length(fields) > size(rocqueues,2)  # Note: for simplicity, we create a queue for every field even if it is not a ROCArray
            nqueues = length(fields)-size(rocqueues,2);
            new_rocqueues  = Array{AMDGPU.HSAQueue}(undef, NNEIGHBORS_PER_DIM, nqueues);
            new_rocsignals = Array{Union{AMDGPU.HSASignal,AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal},Missing}}(missing, NNEIGHBORS_PER_DIM, nqueues); # DEBUG: tmp solution to avoid rocsignals array access filing when accessing an unset signal
            for i = 1:nqueues
                for n=1:NNEIGHBORS_PER_DIM
                    q = AMDGPU.HSAQueue(get_default_agent())
                    AMDGPU.HSA.amd_queue_set_priority(q.queue, AMDGPU.HSA.AMD_QUEUE_PRIORITY_HIGH)
                    new_rocqueues[n,i] = q
                end
            end
            rocqueues  = [rocqueues  new_rocqueues]
            rocsignals = [rocsignals new_rocsignals]
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, A::ROCArray{T}, i::Integer) where T <: GGNumber
        if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
            # DEBUG: write_d2h_async commented for now as AMDGPU support for it in dev
            if dim == 1 || amdgpuaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges   = sendranges(n, dim, A);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = Tuple([r[end] - r[1] + 1 for r in ranges]);
                rocsignals[n,i] = @roc gridsize=halosize groupsize=nthreads queue=rocqueues[n,i] write_d2x!(gpusendbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim); # DEBUG: usually @roc is wrapped by wait(), but since here we don0t want sync one should check what to do.
                # tmp DEBUG:
                # wait(@roc gridsize=halosize groupsize=nthreads queue=rocqueues[n,i] write_d2x!(gpusendbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim)); # DEBUG: usually @roc is wrapped by wait(), but since here we don0t want sync one should check what to do.
                # !amdgpuaware_MPI(dim) && sendbuf_flat(n,dim,i,A) .= Array(gpusendbuf_flat(n,dim,i,A)); # DEBUG: low-tech hack needed until further AMDGPU.jl functionalities are ready
            else
                rocsignals[n,i] = HSASignal(1)
                write_d2h_async!(sendbuf_flat(n,dim,i,A),A,sendranges(n,dim,A),dim,rocsignals[n,i]);
            end
        end
    end
end

let
    global iread_recvbufs!, allocate_rocqueues_iread, wait_iread

    rocqueues  = Array{AMDGPU.HSAQueue}(undef, NNEIGHBORS_PER_DIM, 0)
    rocsignals = Array{Union{AMDGPU.HSASignal,AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal},Missing}}(undef, NNEIGHBORS_PER_DIM, 0)

    function wait_iread(n::Integer, A::ROCArray{T}, i::Integer) where T <: GGNumber
        if !ismissing(rocsignals[n,i]) # DEBUG: tmp solution to avoid rocsignals array access filing when accessing an unset signal
            wait(rocsignals[n,i]);
            rocsignals[n,i] = missing;
        end
        return
    end

    function allocate_rocqueues_iread(fields::GGArray...)
        if length(fields) > size(rocqueues,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuArray
            nqueues = length(fields)-size(rocqueues,2);
            new_rocqueues  = Array{AMDGPU.HSAQueue}(undef, NNEIGHBORS_PER_DIM, nqueues);
            new_rocsignals = Array{Union{AMDGPU.HSASignal,AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal},Missing}}(missing, NNEIGHBORS_PER_DIM, nqueues); # DEBUG: tmp solution to avoid rocsignals array access filing when accessing an unset signal
            for i = 1:nqueues
                for n=1:NNEIGHBORS_PER_DIM
                    q = AMDGPU.HSAQueue(get_default_agent())
                    AMDGPU.HSA.amd_queue_set_priority(q.queue, AMDGPU.HSA.AMD_QUEUE_PRIORITY_HIGH)
                    new_rocqueues[n,i] = q
                end
            end
            rocqueues  = [rocqueues  new_rocqueues]
            rocsignals = [rocsignals new_rocsignals]
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, A::ROCArray{T}, i::Integer) where T <: GGNumber
        if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
            # DEBUG: write_d2h_async commented for now as AMDGPU support for it in dev
            if dim == 1 || amdgpuaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges   = recvranges(n, dim, A);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = Tuple([r[end] - r[1] + 1 for r in ranges]);
                rocsignals[n,i] = @roc gridsize=halosize groupsize=nthreads queue=rocqueues[n,i] read_x2d!(gpurecvbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim); # DEBUG: usually @roc is wrapped by wait(), but since here we don't want sync one should check what to do.
                # tmp DEBUG:
                # !amdgpuaware_MPI(dim) && gpurecvbuf_flat(n,dim,i,A) .= ROCArray(recvbuf_flat(n,dim,i,A)); # DEBUG: low-tech hack needed until further AMDGPU.jl functionalities are ready
                # wait(@roc gridsize=halosize groupsize=nthreads queue=rocqueues[n,i] read_x2d!(gpurecvbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim)); # DEBUG: usually @roc is wrapped by wait(), but since here we don't want sync one should check what to do.
            else
                rocsignals[n,i] = HSASignal(1)
                read_h2d_async!(recvbuf_flat(n,dim,i,A), A, recvranges(n,dim,A), dim, rocsignals[n,i]);
            end
        end
    end

end


# (CPU/GPU functions)

# Return the ranges from A to be sent. It will always return ranges for the dimensions x,y and z even if the A is 1D or 2D (for 2D, the 3rd range is 1:1; for 1D, the 2nd and 3rd range are 1:1).
function sendranges(n::Integer, dim::Integer, A::GGArray)
    if (ol(dim, A) < 2) error("Incoherent arguments: ol(A,dim)<2."); end
    if     (n==2) ixyz_dim = size(A, dim) - (ol(dim, A) - 1);
    elseif (n==1) ixyz_dim = 1            + (ol(dim, A) - 1);
    end
    sendranges      = [1:size(A,1), 1:size(A,2), 1:size(A,3)];  # Initialize with the ranges of A.
    sendranges[dim] = ixyz_dim:ixyz_dim;
    return sendranges
end

# Return the ranges from A to be received. It will always return ranges for the dimensions x,y and z even if the A is 1D or 2D (for 2D, the 3rd range is 1:1; for 1D, the 2nd and 3rd range are 1:1).
function recvranges(n::Integer, dim::Integer, A::GGArray)
    if (ol(dim, A) < 2) error("Incoherent arguments: ol(A,dim)<2."); end
    if     (n==2) ixyz_dim = size(A, dim);
    elseif (n==1) ixyz_dim = 1;
    end
    recvranges      = [1:size(A,1), 1:size(A,2), 1:size(A,3)];  # Initialize with the ranges of A.
    recvranges[dim] = ixyz_dim:ixyz_dim;
    return recvranges
end


# (CPU functions)

# Write to the send buffer on the host from the array on the host (h2h). Note: it works for 1D-3D, as sendranges contains always 3 ranges independently of the number of dimensions of A (see function sendranges).
function write_h2h!(sendbuf::AbstractArray{T}, A::Array{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    ix = (length(sendranges[1])==1) ? sendranges[1][1] : sendranges[1];
    iy = (length(sendranges[2])==1) ? sendranges[2][1] : sendranges[2];
    iz = (length(sendranges[3])==1) ? sendranges[3][1] : sendranges[3];
    if     (dim == 1 && length(ix)==1     && iy == 1:size(A,2) && iz == 1:size(A,3)) memcopy!(sendbuf, view(A,ix, :, :), loopvectorization(dim));
    elseif (dim == 1 && length(ix)==1     && iy == 1:size(A,2) && length(iz)==1    ) memcopy!(sendbuf, view(A,ix, :,iz), loopvectorization(dim));
    elseif (dim == 1 && length(ix)==1     && length(iy)==1     && length(iz)==1    ) memcopy!(sendbuf, view(A,ix,iy,iz), loopvectorization(dim));
    elseif (dim == 2 && ix == 1:size(A,1) && length(iy)==1     && iz == 1:size(A,3)) memcopy!(sendbuf, view(A, :,iy, :), loopvectorization(dim));
    elseif (dim == 2 && ix == 1:size(A,1) && length(iy)==1     && length(iz)==1    ) memcopy!(sendbuf, view(A, :,iy,iz), loopvectorization(dim));
    elseif (dim == 3 && ix == 1:size(A,1) && iy == 1:size(A,2)                     ) memcopy!(sendbuf, view(A, :, :,iz), loopvectorization(dim));
    elseif (dim == 1 || dim == 2 || dim == 3)                                        memcopy!(sendbuf, view(A,sendranges...), loopvectorization(dim)); # This general case is slower than the three optimised cases above (the result would be the same, of course).
    end
end

# Read from the receive buffer on the host and store on the array on the host (h2h). Note: it works for 1D-3D, as recvranges contains always 3 ranges independently of the number of dimensions of A (see function recvranges).
function read_h2h!(recvbuf::AbstractArray{T}, A::Array{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    ix = (length(recvranges[1])==1) ? recvranges[1][1] : recvranges[1];
    iy = (length(recvranges[2])==1) ? recvranges[2][1] : recvranges[2];
    iz = (length(recvranges[3])==1) ? recvranges[3][1] : recvranges[3];
    if     (dim == 1 && length(ix)==1     && iy == 1:size(A,2) && iz == 1:size(A,3)) memcopy!(view(A,ix, :, :), recvbuf, loopvectorization(dim));
    elseif (dim == 1 && length(ix)==1     && iy == 1:size(A,2) && length(iz)==1    ) memcopy!(view(A,ix, :,iz), recvbuf, loopvectorization(dim));
    elseif (dim == 1 && length(ix)==1     && length(iy)==1     && length(iz)==1    ) memcopy!(view(A,ix,iy,iz), recvbuf, loopvectorization(dim));
    elseif (dim == 2 && ix == 1:size(A,1) && length(iy)==1     && iz == 1:size(A,3)) memcopy!(view(A, :,iy, :), recvbuf, loopvectorization(dim));
    elseif (dim == 2 && ix == 1:size(A,1) && length(iy)==1     && length(iz)==1    ) memcopy!(view(A, :,iy,iz), recvbuf, loopvectorization(dim));
    elseif (dim == 3 && ix == 1:size(A,1) && iy == 1:size(A,2)                     ) memcopy!(view(A, :, :,iz), recvbuf, loopvectorization(dim));
    elseif (dim == 1 || dim == 2 || dim == 3)                                        memcopy!(view(A,recvranges...), recvbuf, loopvectorization(dim)); # This general case is slower than the three optimised cases above (the result would be the same, of course).
    end
end


# (CUDA functions)

# Write to the send buffer on the host or device from the array on the device (d2x).
function write_d2x!(gpusendbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x + sendrangex[1] - 1
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y + sendrangey[1] - 1
    iz = (CUDA.blockIdx().z-1) * CUDA.blockDim().z + CUDA.threadIdx().z + sendrangez[1] - 1
    if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
    if     (dim == 1) gpusendbuf[iy,iz] = A[ix,iy,iz];
    elseif (dim == 2) gpusendbuf[ix,iz] = A[ix,iy,iz];
    elseif (dim == 3) gpusendbuf[ix,iy] = A[ix,iy,iz];
    end
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function read_x2d!(gpurecvbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x + recvrangex[1] - 1
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y + recvrangey[1] - 1
    iz = (CUDA.blockIdx().z-1) * CUDA.blockDim().z + CUDA.threadIdx().z + recvrangez[1] - 1
    if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
    if     (dim == 1) A[ix,iy,iz] = gpurecvbuf[iy,iz];
    elseif (dim == 2) A[ix,iy,iz] = gpurecvbuf[ix,iz];
    elseif (dim == 3) A[ix,iy,iz] = gpurecvbuf[ix,iy];
    end
    return nothing
end

# Write to the send buffer on the host from the array on the device (d2h).
function write_d2h_async!(sendbuf::AbstractArray{T}, A::CuArray{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer, custream::CuStream) where T <: GGNumber where T2 <: Integer
    CUDA.Mem.unsafe_copy3d!(
        pointer(sendbuf), CUDA.Mem.Host, pointer(A), CUDA.Mem.Device,
        length(sendranges[1]), length(sendranges[2]), length(sendranges[3]);
        srcPos=(sendranges[1][1], sendranges[2][1], sendranges[3][1]),
        srcPitch=sizeof(T)*size(A,1), srcHeight=size(A,2),
        dstPitch=sizeof(T)*length(sendranges[1]), dstHeight=length(sendranges[2]),
        async=true, stream=custream
    )
end

# Read from the receive buffer on the host and store on the array on the device (h2d).
function read_h2d_async!(recvbuf::AbstractArray{T}, A::CuArray{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer, custream::CuStream) where T <: GGNumber where T2 <: Integer
    CUDA.Mem.unsafe_copy3d!(
        pointer(A), CUDA.Mem.Device, pointer(recvbuf), CUDA.Mem.Host,
        length(recvranges[1]), length(recvranges[2]), length(recvranges[3]);
        dstPos=(recvranges[1][1], recvranges[2][1], recvranges[3][1]),
        srcPitch=sizeof(T)*length(recvranges[1]), srcHeight=length(recvranges[2]),
        dstPitch=sizeof(T)*size(A,1), dstHeight=size(A,2),
        async=true, stream=custream
    )
end


# (AMDGPU functions)

# Write to the send buffer on the host or device from the array on the device (d2x).
function write_d2x!(gpusendbuf::ROCDeviceArray{T}, A::ROCDeviceArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
    ix = (AMDGPU.workgroupIdx().x-1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x + sendrangex[1] - 1
    iy = (AMDGPU.workgroupIdx().y-1) * AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y + sendrangey[1] - 1
    iz = (AMDGPU.workgroupIdx().z-1) * AMDGPU.workgroupDim().z + AMDGPU.workitemIdx().z + sendrangez[1] - 1
    if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
    if     (dim == 1) gpusendbuf[iy,iz] = A[ix,iy,iz];
    elseif (dim == 2) gpusendbuf[ix,iz] = A[ix,iy,iz];
    elseif (dim == 3) gpusendbuf[ix,iy] = A[ix,iy,iz];
    end
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function read_x2d!(gpurecvbuf::ROCDeviceArray{T}, A::ROCDeviceArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
    ix = (AMDGPU.workgroupIdx().x-1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x + recvrangex[1] - 1
    iy = (AMDGPU.workgroupIdx().y-1) * AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y + recvrangey[1] - 1
    iz = (AMDGPU.workgroupIdx().z-1) * AMDGPU.workgroupDim().z + AMDGPU.workitemIdx().z + recvrangez[1] - 1
    if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
    if     (dim == 1) A[ix,iy,iz] = gpurecvbuf[iy,iz];
    elseif (dim == 2) A[ix,iy,iz] = gpurecvbuf[ix,iz];
    elseif (dim == 3) A[ix,iy,iz] = gpurecvbuf[ix,iy];
    end
    return nothing
end

# # Write to the send buffer on the host from the array on the device (d2h).
# function write_d2h_async!(sendbuf::AbstractArray{T}, A::ROCArray{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer, rocqueue::HSAQueue) where T <: GGNumber where T2 <: Integer
#     error("AMDGPU is not yet supported")
# end

# # Read from the receive buffer on the host and store on the array on the device (h2d).
# function read_h2d_async!(recvbuf::AbstractArray{T}, A::ROCArray{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer, rocqueue::HSAQueue) where T <: GGNumber where T2 <: Integer
#     error("AMDGPU is not yet supported")
# end

# Write to the send buffer on the host from the array on the device (d2h).
function write_d2h_async!(sendbuf::AbstractArray{T}, A::ROCArray{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer, signal::HSASignal) where T <: GGNumber where T2 <: Integer
    locked_ptr = AMDGPU.Mem.lock(pointer(sendbuf), sizeof(sendbuf), get_default_agent())

    src_bytes   = (sizeof(T)*(sendranges[1][1]-1),sendranges[2][1]-1,sendranges[3][1]-1)
    range_bytes = (sizeof(T)*(length(sendranges[1])),length(sendranges[2]),length(sendranges[3]))

    dst_ref     = Ref(AMDGPU.HSA.PitchedPtr(locked_ptr,sizeof(T)*length(sendranges[1]),sizeof(T)*length(sendranges[1])*length(sendranges[2])))
    src_ref     = Ref(AMDGPU.HSA.PitchedPtr(pointer(A),sizeof(T)*size(A,1)            ,sizeof(T)*size(A,1)*size(A,2)))
    dst_off     = Ref(AMDGPU.HSA.Dim3(0,0,0))
    src_off     = Ref(AMDGPU.HSA.Dim3(src_bytes...))
    range       = Ref(AMDGPU.HSA.Dim3(range_bytes...))

    dst_ptr     = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr},dst_ref)
    src_ptr     = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr},src_ref)
    dst_off_ptr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},dst_off)
    src_off_ptr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},src_off)
    range_ptr   = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},range)

    @assert dst_ptr     != C_NULL "dst_ptr is null"
    @assert src_ptr     != C_NULL "src_ptr is null"
    @assert dst_off_ptr != C_NULL "dst_off_ptr is null"
    @assert src_off_ptr != C_NULL "src_off_ptr is null"
    @assert range_ptr   != C_NULL "range_ptr is null"

    @assert convert(Int,dst_ref[].base) % 4 == 0 "dst base % 4 != 0"
    @assert convert(Int,src_ref[].base) % 4 == 0 "src base % 4 != 0"

    @assert dst_ref[].pitch % 4 == 0 "dst pitch % 4 != 0"
    @assert src_ref[].pitch % 4 == 0 "src pitch % 4 != 0"

    @assert dst_ref[].slice % 4 == 0 "dst slice % 4 != 0"
    @assert src_ref[].slice % 4 == 0 "src slice % 4 != 0"

    @assert src_off[].x + range[].x <= src_ref[].pitch "Src rect width out of range"
    @assert dst_off[].x + range[].x <= dst_ref[].pitch "Dst rect width out of range"
    
    @assert src_ref[].slice == 0 || (src_off[].y + range[].y) <= src_ref[].slice ÷ src_ref[].pitch "Src rect height out of range"
    @assert dst_ref[].slice == 0 || (dst_off[].y + range[].y) <= dst_ref[].slice ÷ dst_ref[].pitch "Dst rect height out of range"

    @assert range[].z <= 1 || (src_ref[].slice != 0 && dst_ref[].slice != 0) "Copy rect slice needed."
    
    AMDGPU.HSA.amd_memory_async_copy_rect(dst_ptr,dst_off_ptr,src_ptr,src_off_ptr,range_ptr,
                                          get_default_agent().agent,AMDGPU.HSA.LibHSARuntime.hsaDeviceToHost,UInt32(0),C_NULL,signal.signal[]) |> AMDGPU.check
    
    AMDGPU.Mem.unlock(pointer(sendbuf))
    return nothing
end

# Read from the receive buffer on the host and store on the array on the device (h2d).
function read_h2d_async!(recvbuf::AbstractArray{T}, A::ROCArray{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer, signal::HSASignal) where T <: GGNumber where T2 <: Integer
    locked_ptr = AMDGPU.Mem.lock(pointer(recvbuf), sizeof(recvbuf), get_default_agent())

    dst_bytes   = (sizeof(T)*(recvranges[1][1]-1),recvranges[2][1]-1,recvranges[3][1]-1)
    range_bytes = (sizeof(T)*(length(recvranges[1])),length(recvranges[2]),length(recvranges[3]))

    src_ref     = Ref(AMDGPU.HSA.PitchedPtr(locked_ptr,sizeof(T)*length(recvranges[1]),sizeof(T)*length(recvranges[1])*length(recvranges[2])))
    dst_ref     = Ref(AMDGPU.HSA.PitchedPtr(pointer(A),sizeof(T)*size(A,1)            ,sizeof(T)*size(A,1)*size(A,2)))
    src_off     = Ref(AMDGPU.HSA.Dim3(0,0,0))
    dst_off     = Ref(AMDGPU.HSA.Dim3(dst_bytes...))
    range       = Ref(AMDGPU.HSA.Dim3(range_bytes...))

    src_ptr     = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr},src_ref)
    dst_ptr     = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr},dst_ref)
    src_off_ptr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},src_off)
    dst_off_ptr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},dst_off)
    range_ptr   = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},range)

    @assert dst_ptr     != C_NULL "dst_ptr is null"
    @assert src_ptr     != C_NULL "src_ptr is null"
    @assert dst_off_ptr != C_NULL "dst_off_ptr is null"
    @assert src_off_ptr != C_NULL "src_off_ptr is null"
    @assert range_ptr   != C_NULL "range_ptr is null"

    @assert convert(Int,dst_ref[].base) % 4 == 0 "dst base % 4 != 0"
    @assert convert(Int,src_ref[].base) % 4 == 0 "src base % 4 != 0"

    @assert dst_ref[].pitch % 4 == 0 "dst pitch % 4 != 0"
    @assert src_ref[].pitch % 4 == 0 "src pitch % 4 != 0"

    @assert dst_ref[].slice % 4 == 0 "dst slice % 4 != 0"
    @assert src_ref[].slice % 4 == 0 "src slice % 4 != 0"

    @assert src_off[].x + range[].x <= src_ref[].pitch "Src rect width out of range"
    @assert dst_off[].x + range[].x <= dst_ref[].pitch "Dst rect width out of range"
    
    @assert src_ref[].slice == 0 || (src_off[].y + range[].y) <= src_ref[].slice ÷ src_ref[].pitch "Src rect height out of range"
    @assert dst_ref[].slice == 0 || (dst_off[].y + range[].y) <= dst_ref[].slice ÷ dst_ref[].pitch "Dst rect height out of range"

    @assert range[].z <= 1 || (src_ref[].slice != 0 && dst_ref[].slice != 0) "Copy rect slice needed."
    
    AMDGPU.HSA.amd_memory_async_copy_rect(dst_ptr,dst_off_ptr,src_ptr,src_off_ptr,range_ptr,
                                          get_default_agent().agent,AMDGPU.HSA.LibHSARuntime.hsaHostToDevice,UInt32(0),C_NULL,signal.signal[]) |> AMDGPU.check
    
    AMDGPU.Mem.unlock(pointer(recvbuf))
    return nothing
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function irecv_halo!(n::Integer, dim::Integer, A::GGArray, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            req = MPI.Irecv!(gpurecvbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        else
            req = MPI.Irecv!(recvbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function isend_halo(n::Integer, dim::Integer, A::GGArray, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            req = MPI.Isend(gpusendbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        else
            req = MPI.Isend(sendbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function sendrecv_halo_local(n::Integer, dim::Integer, A::GGArray, i::Integer)
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            if n == 1
                gpumemcopy!(gpurecvbuf_flat(2,dim,i,A), gpusendbuf_flat(1,dim,i,A));
            elseif n == 2
                gpumemcopy!(gpurecvbuf_flat(1,dim,i,A), gpusendbuf_flat(2,dim,i,A));
            end
        else
            if n == 1
                memcopy!(recvbuf_flat(2,dim,i,A), sendbuf_flat(1,dim,i,A), loopvectorization(dim));
            elseif n == 2
                memcopy!(recvbuf_flat(1,dim,i,A), sendbuf_flat(2,dim,i,A), loopvectorization(dim));
            end
        end
    end
end

function memcopy!(dst::AbstractArray{T}, src::AbstractArray{T}, loopvectorization::Bool) where T <: GGNumber
    if loopvectorization && !(T <: Complex)  # NOTE: LoopVectorization does not yet support Complex numbers and copy reinterpreted arrays leads to bad performance.
        memcopy_loopvect!(dst, src)
    else
        dst_flat = view(dst,:)
        src_flat = view(src,:)
        memcopy_threads!(dst_flat, src_flat)
    end
end

# (CPU functions)
function memcopy_threads!(dst::AbstractArray{T}, src::AbstractArray{T}) where T <: GGNumber
    if nthreads() > 1 && sizeof(src) >= GG_THREADCOPY_THRESHOLD
        @threads for i = 1:length(dst)  # NOTE: Set the number of threads e.g. as: export JULIA_NUM_THREADS=12
            @inbounds dst[i] = src[i]   # NOTE: We fix here exceptionally the use of @inbounds as this copy between two flat vectors (which must have the right length) is considered safe.
        end
    else
        @inbounds copyto!(dst, src)
    end
end

function memcopy_loopvect!(dst::AbstractArray{T}, src::AbstractArray{T}) where T <: GGNumber
    if nthreads() > 1 && length(src) > 1
        @tturbo for i ∈ eachindex(dst, src)  # NOTE: tturbo will use maximally Threads.nthreads() threads. Set the number of threads e.g. as: export JULIA_NUM_THREADS=12. NOTE: tturbo fails if src_flat and dst_flat are used due to an issue in ArrayInterface : https://github.com/JuliaArrays/ArrayInterface.jl/issues/228
            @inbounds dst[i] = src[i]        # NOTE: We fix here exceptionally the use of @inbounds (currently anyways done by LoopVectorization) as this copy between two flat vectors (which must have the right length) is considered safe.
        end
    else
        @inbounds copyto!(dst, src)
    end
end


# (CUDA functions)

function gpumemcopy!(dst::CuArray{T}, src::CuArray{T}) where T <: GGNumber
    @inbounds CUDA.copyto!(dst, src)
end


# (AMDGPU functions)

function gpumemcopy!(dst::ROCArray{T}, src::ROCArray{T}) where T <: GGNumber
    @inbounds AMDGPU.copyto!(dst, src)
end


##-------------------------------------------
## FUNCTIONS FOR CHECKING THE INPUT ARGUMENTS

function check_fields(fields::GGArray...)
    # Raise an error if any of the given fields does not have a halo.
    no_halo = Int[];
    for i = 1:length(fields)
        A = fields[i];
        if all([ol(dim, A) < 2 for dim = 1:ndims(A)]) # There is no halo if the overlap is less than 2...
            push!(no_halo, i);
        end
    end
    if length(no_halo) > 1
        error("The fields at positions $(join(no_halo,", "," and ")) have no halo; remove them from the call.")
    elseif length(no_halo) > 0
        error("The field at position $(no_halo[1]) has no halo; remove it from the call.")
    end

    # Raise an error if any of the given fields contains any duplicates.
    duplicates = [[i,j] for i=1:length(fields) for j=i+1:length(fields) if fields[i]===fields[j]];
    if length(duplicates) > 2
        error("The pairs of fields with the positions $(join(duplicates,", "," and ")) are the same; remove any duplicates from the call.")
    elseif length(duplicates) > 0
        error("The field at position $(duplicates[1][2]) is a duplicate of the one at the position $(duplicates[1][1]); remove the duplicate from the call.")
    end

    # Raise an error if not all fields are of the same datatype (restriction comes from buffer handling).
    different_types = [i for i=2:length(fields) if typeof(fields[i])!=typeof(fields[1])];
    if length(different_types) > 1
        error("The fields at positions $(join(different_types,", "," and ")) are of different type than the first field; make sure that in a same call all fields are of the same type.")
    elseif length(different_types) == 1
        error("The field at position $(different_types[1]) is of different type than the first field; make sure that in a same call all fields are of the same type.")
    end
end
