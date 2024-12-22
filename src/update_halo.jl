export update_halo!

"""
    update_halo!(A)
    update_halo!(A...)

!!! note "Advanced"
        update_halo!(A, B, (A=C, halowidths=..., (A=D, halowidths=...), ...)

Update the halo of the given GPU/CPU-array(s).

# Typical use cases:
    update_halo!(A)                                # Update the halo of the array A.
    update_halo!(A, B, C)                          # Update the halos of the arrays A, B and C.
    update_halo!(A, B, (A=C, halowidths=(2,2,2)))  # Update the halos of the arrays A, B, C, defining non default halowidth for C.

!!! note "Performance note"
    Group subsequent calls to `update_halo!` in a single call for better performance (enables additional pipelining).

!!! note "Performance note"
    If the system supports CUDA-aware MPI (for Nvidia GPUs) or ROCm-aware MPI (for AMD GPUs), it may be activated for ImplicitGlobalGrid by setting one of the following environment variables (at latest before the call to `init_global_grid`):
    ```shell
    shell> export IGG_CUDAAWARE_MPI=1
    ```
    ```shell
    shell> export IGG_ROCMAWARE_MPI=1
    ```
"""
function update_halo!(A::Union{GGArray, GGFieldConvertible, GGCellArray, GGCellFieldConvertible, GGField}...; dims=(NDIMS_MPI,(1:NDIMS_MPI-1)...))
    check_initialized();
    As = ((extract.(A)...)...,);
    fields = wrap_field.(As);
    check_fields(fields...);
    _update_halo!(fields...; dims=dims);  # Assignment of A to fields in the internal function _update_halo!() as vararg A can consist of multiple fields; A will be used for a single field in the following (The args of update_halo! must however be "A..." for maximal simplicity and elegance for the user).
    return nothing
end

function _update_halo!(fields::GGField...; dims=dims)
    if (!cuda_enabled() && !amdgpu_enabled() && !all_arrays(fields...)) error("not all arrays are CPU arrays, but no GPU extension is loaded.") end #NOTE: in the following, it is only required to check for `cuda_enabled()`/`amdgpu_enabled()` when the context does not imply `any_cuarray(fields...)` or `is_cuarray(A)` or the corresponding for AMDGPU. # NOTE: the case where only one of the two extensions are loaded, but an array dad would be for the other extension is passed is very unlikely and therefore not explicitly checked here (but could be added later).
    allocate_bufs(fields...);
    if any_array(fields...) allocate_tasks(fields...); end
    if any_cuarray(fields...) allocate_custreams(fields...); end
    if any_rocarray(fields...) allocate_rocstreams(fields...); end

    for dim in dims  # NOTE: this works for 1D-3D (e.g. if nx>1, ny>1 and nz=1, then for d=3, there will be no neighbors, i.e. nothing will be done as desired...).
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

halosize(dim::Integer, A::GGField) = (dim==1) ? (A.halowidths[1], size(A,2), size(A,3)) : ((dim==2) ? (size(A,1), A.halowidths[2], size(A,3)) : (size(A,1), size(A,2), A.halowidths[3]))


##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

let
    #TODO: this was: global free_update_halo_buffers, allocate_bufs, sendbuf, recvbuf, sendbuf_flat, recvbuf_flat, gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat, rocsendbuf, rocrecvbuf, rocsendbuf_flat, rocrecvbuf_flat
    global free_update_halo_buffers, allocate_bufs, sendbuf, recvbuf, sendbuf_flat, recvbuf_flat
    sendbufs_raw = nothing
    recvbufs_raw = nothing

    function free_update_halo_buffers()
        free_update_halo_cpubuffers()
        if (cuda_enabled() && none(cudaaware_MPI()))     free_update_halo_cubuffers() end
        if (amdgpu_enabled() && none(amdgpuaware_MPI())) free_update_halo_rocbuffers() end
        GC.gc() #TODO: see how to modify this!
    end

    function free_update_halo_cpubuffers()
        reset_cpu_buffers();
    end

    function reset_cpu_buffers()
        sendbufs_raw = nothing
        recvbufs_raw = nothing
    end

    # Allocate for each field two send and recv buffers (one for the left and one for the right neighbour of a dimension). The required length of the buffer is given by the maximal number of halo elements in any of the dimensions. Note that buffers are not allocated separately for each dimension, as the updates are performed one dimension at a time (required for correctness).
    function allocate_bufs(fields::GGField{T}...) where T <: GGNumber
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
            A, halowidths = fields[i];
            for n = 1:NNEIGHBORS_PER_DIM # Ensure that the buffers are interpreted to contain elements of the same type as the array.
                reinterpret_bufs(T, i, n);
                if cuda_enabled() reinterpret_cubufs(T, i, n); end
                if amdgpu_enabled() reinterpret_rocbufs(T, i, n); end
            end
            max_halo_elems = maximum((size(A,1)*size(A,2)*halowidths[3], size(A,1)*size(A,3)*halowidths[2], size(A,2)*size(A,3)*halowidths[1]));
            reallocate_undersized_hostbufs(T, i, max_halo_elems, A);
            if (is_cuarray(A) && any(cudaaware_MPI())) reallocate_undersized_cubufs(T, i, max_halo_elems) end
            if (is_rocarray(A) && any(amdgpuaware_MPI())) reallocate_undersized_rocbufs(T, i, max_halo_elems) end
        end
    end


    # (CPU functions)

    function init_bufs_arrays()
        sendbufs_raw = Array{Array{Any,1},1}();
        recvbufs_raw = Array{Array{Any,1},1}();
    end

    function init_bufs(T::DataType, fields::GGField...)
        while (length(sendbufs_raw) < length(fields)) push!(sendbufs_raw, [zeros(T,0), zeros(T,0)]); end
        while (length(recvbufs_raw) < length(fields)) push!(recvbufs_raw, [zeros(T,0), zeros(T,0)]); end
    end

    function reinterpret_bufs(T::DataType, i::Integer, n::Integer)
        if (eltype(sendbufs_raw[i][n]) != T) sendbufs_raw[i][n] = reinterpret(T, sendbufs_raw[i][n]); end
        if (eltype(recvbufs_raw[i][n]) != T) recvbufs_raw[i][n] = reinterpret(T, recvbufs_raw[i][n]); end
    end

    function reallocate_undersized_hostbufs(T::DataType, i::Integer, max_halo_elems::Integer, A::GGArray)
        if (length(sendbufs_raw[i][1]) < max_halo_elems)
            for n = 1:NNEIGHBORS_PER_DIM
                reallocate_bufs(T, i, n, max_halo_elems);
                if (is_cuarray(A) && none(cudaaware_MPI())) reregister_cubufs(T, i, n, sendbufs_raw, recvbufs_raw); end  # Host memory is page-locked (and mapped to device memory) to ensure optimal access performance (from kernel or with 3-D memcopy).
                if (is_rocarray(A) && none(amdgpuaware_MPI())) reregister_rocbufs(T, i, n, sendbufs_raw, recvbufs_raw); end  # ...
            end
            GC.gc(); # Too small buffers had been replaced with larger ones; free the now unused memory.
        end
    end

    function reallocate_bufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        sendbufs_raw[i][n] = zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        recvbufs_raw[i][n] = zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end


    # (CPU functions)

    function sendbuf_flat(n::Integer, dim::Integer, i::Integer, A::GGField{T}) where T <: GGNumber
        return view(sendbufs_raw[i][n]::AbstractVector{T},1:prod(halosize(dim,A)));
    end

    function recvbuf_flat(n::Integer, dim::Integer, i::Integer, A::GGField{T}) where T <: GGNumber
        return view(recvbufs_raw[i][n]::AbstractVector{T},1:prod(halosize(dim,A)));
    end

    function sendbuf(n::Integer, dim::Integer, i::Integer, A::GGField)
        return reshape(sendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function recvbuf(n::Integer, dim::Integer, i::Integer, A::GGField)
        return reshape(recvbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_sendbufs_raw, get_recvbufs_raw
    get_sendbufs_raw()    = deepcopy(sendbufs_raw)
    get_recvbufs_raw()    = deepcopy(recvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

# NOTE: the tasks, custreams and rocqueues are stored here in a let clause to have them survive the end of a call to update_boundaries. This avoids the creation of new tasks and cuda streams every time. Besides, that this could be relevant for performance, it is important for debugging the overlapping the communication with computation (if at every call new stream/task objects are created this becomes very messy and hard to analyse).


# (CPU functions)

function allocate_tasks(fields::GGField...)
    allocate_tasks_iwrite(fields...);
    allocate_tasks_iread(fields...);
end

let
    global iwrite_sendbufs!, allocate_tasks_iwrite, wait_iwrite

    tasks = Array{Task}(undef, NNEIGHBORS_PER_DIM, 0);

    wait_iwrite(n::Integer, A::CPUField{T}, i::Integer) where T <: GGNumber = (schedule(tasks[n,i]); wait(tasks[n,i]);) # The argument A is used for multiple dispatch. #NOTE: The current implementation only starts a task when it is waited for, in order to make sure that only one task is run at a time and that they are run in the desired order (best for performance as the tasks are mapped only to one thread via context switching).

    function allocate_tasks_iwrite(fields::GGField...)
        if length(fields) > size(tasks,2)  # Note: for simplicity, we create a tasks for every field even if it is not an CPUField
            tasks = [tasks Array{Task}(undef, NNEIGHBORS_PER_DIM, length(fields)-size(tasks,2))];  # Create (additional) emtpy tasks.
        end 
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, F::CPUField{T}, i::Integer) where T <: GGNumber  # Function to be called if A is a CPUField.
        A, halowidths = F;
        tasks[n,i] = @task begin
            if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
                write_h2h!(sendbuf(n,dim,i,F), A, sendranges(n,dim,F), dim);
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

    wait_iread(n::Integer, A::CPUField{T}, i::Integer) where T <: GGNumber = (schedule(tasks[n,i]); wait(tasks[n,i]);) #NOTE: The current implementation only starts a task when it is waited for, in order to make sure that only one task is run at a time and that they are run in the desired order (best for performance currently as the tasks are mapped only to one thread via context switching).

    function allocate_tasks_iread(fields::GGField...)
        if length(fields) > size(tasks,2)  # Note: for simplicity, we create a tasks for every field even if it is not an Array
            tasks = [tasks Array{Task}(undef, NNEIGHBORS_PER_DIM, length(fields)-size(tasks,2))];  # Create (additional) emtpy tasks.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, F::CPUField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        tasks[n,i] = @task begin
            if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
                read_h2h!(recvbuf(n,dim,i,F), A, recvranges(n,dim,F), dim);
            end
        end
    end

    # Make tasks accessible for unit testing.
    global get_tasks_iread
    get_tasks_iread() = deepcopy(tasks)
end


# (CPU/GPU functions)

# Return the ranges from A to be sent. It will always return ranges for the dimensions x,y and z even if the A is 1D or 2D (for 2D, the 3rd range is 1:1; for 1D, the 2nd and 3rd range are 1:1).
function sendranges(n::Integer, dim::Integer, F::GGField)
    A, halowidths = F;
    if (ol(dim, A) < 2*halowidths[dim]) error("Incoherent arguments: ol(A,dim)<2*halowidths[dim]."); end
    if     (n==2) ixyz_dim = size(A, dim) - (ol(dim, A) - 1);
    elseif (n==1) ixyz_dim = 1            + (ol(dim, A) - halowidths[dim]);
    end
    sendranges      = [1:size(A,1), 1:size(A,2), 1:size(A,3)];  # Initialize with the ranges of A.
    sendranges[dim] = ixyz_dim:ixyz_dim+halowidths[dim]-1;
    return sendranges
end

# Return the ranges from A to be received. It will always return ranges for the dimensions x,y and z even if the A is 1D or 2D (for 2D, the 3rd range is 1:1; for 1D, the 2nd and 3rd range are 1:1).
function recvranges(n::Integer, dim::Integer, F::GGField)
    A, halowidths = F;
    if (ol(dim, A) < 2*halowidths[dim]) error("Incoherent arguments: ol(A,dim)<2*halowidths[dim]."); end
    if     (n==2) ixyz_dim = size(A, dim) - (halowidths[dim] - 1);
    elseif (n==1) ixyz_dim = 1;
    end
    recvranges      = [1:size(A,1), 1:size(A,2), 1:size(A,3)];  # Initialize with the ranges of A.
    recvranges[dim] = ixyz_dim:ixyz_dim+halowidths[dim]-1;
    return recvranges
end


# (CPU functions)

# Write to the send buffer on the host from the array on the host (h2h). Note: it works for 1D-3D, as sendranges contains always 3 ranges independently of the number of dimensions of A (see function sendranges).
function write_h2h!(sendbuf::AbstractArray{T}, A::AbstractArray{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    ix = (length(sendranges[1])==1) ? sendranges[1][1] : sendranges[1];
    iy = (length(sendranges[2])==1) ? sendranges[2][1] : sendranges[2];
    iz = (length(sendranges[3])==1) ? sendranges[3][1] : sendranges[3];
    if     (length(ix)==1     && iy == 1:size(A,2) && iz == 1:size(A,3) && !use_polyester(dim)) memcopy!(view(sendbuf, 1, :, :), view(A,ix, :, :), use_polyester(dim));
    elseif (length(ix)==1     && length(iy)==1     && iz == 1:size(A,3) && !use_polyester(dim)) memcopy!(view(sendbuf, 1, 1, :), view(A,ix,iy, :), use_polyester(dim));
    elseif (length(ix)==1     && iy == 1:size(A,2) && length(iz)==1     && !use_polyester(dim)) memcopy!(view(sendbuf, 1, :, 1), view(A,ix, :,iz), use_polyester(dim));
    elseif (length(ix)==1     && length(iy)==1     && length(iz)==1     && !use_polyester(dim)) memcopy!(view(sendbuf, 1, 1, 1), view(A,ix,iy,iz), use_polyester(dim));
    elseif (ix == 1:size(A,1) && length(iy)==1     && iz == 1:size(A,3)                       ) memcopy!(view(sendbuf, :, 1, :), view(A, :,iy, :), use_polyester(dim));
    elseif (ix == 1:size(A,1) && length(iy)==1     && length(iz)==1                           ) memcopy!(view(sendbuf, :, 1, 1), view(A, :,iy,iz), use_polyester(dim));
    elseif (ix == 1:size(A,1) && iy == 1:size(A,2) && length(iz)==1                           ) memcopy!(view(sendbuf, :, :, 1), view(A, :, :,iz), use_polyester(dim));
    else                                                                                        memcopy!(sendbuf, view(A,sendranges...), use_polyester(dim)); # This general case is slower than the optimised cases above (the result would be the same, of course).
    end
end

# Read from the receive buffer on the host and store on the array on the host (h2h). Note: it works for 1D-3D, as recvranges contains always 3 ranges independently of the number of dimensions of A (see function recvranges).
function read_h2h!(recvbuf::AbstractArray{T}, A::AbstractArray{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    ix = (length(recvranges[1])==1) ? recvranges[1][1] : recvranges[1];
    iy = (length(recvranges[2])==1) ? recvranges[2][1] : recvranges[2];
    iz = (length(recvranges[3])==1) ? recvranges[3][1] : recvranges[3];
    if     (length(ix)==1     && iy == 1:size(A,2) && iz == 1:size(A,3) && !use_polyester(dim)) memcopy!(view(A,ix, :, :), view(recvbuf, 1, :, :), use_polyester(dim));
    elseif (length(ix)==1     && length(iy)==1     && iz == 1:size(A,3) && !use_polyester(dim)) memcopy!(view(A,ix,iy, :), view(recvbuf, 1, 1, :), use_polyester(dim));
    elseif (length(ix)==1     && iy == 1:size(A,2) && length(iz)==1     && !use_polyester(dim)) memcopy!(view(A,ix, :,iz), view(recvbuf, 1, :, 1), use_polyester(dim));
    elseif (length(ix)==1     && length(iy)==1     && length(iz)==1     && !use_polyester(dim)) memcopy!(view(A,ix,iy,iz), view(recvbuf, 1, 1, 1), use_polyester(dim));
    elseif (ix == 1:size(A,1) && length(iy)==1     && iz == 1:size(A,3)                       ) memcopy!(view(A, :,iy, :), view(recvbuf, :, 1, :), use_polyester(dim));
    elseif (ix == 1:size(A,1) && length(iy)==1     && length(iz)==1                           ) memcopy!(view(A, :,iy,iz), view(recvbuf, :, 1, 1), use_polyester(dim));
    elseif (ix == 1:size(A,1) && iy == 1:size(A,2) && length(iz)==1                           ) memcopy!(view(A, :, :,iz), view(recvbuf, :, :, 1), use_polyester(dim));
    else                                                                                        memcopy!(view(A,recvranges...), recvbuf, use_polyester(dim)); # This general case is slower than the optimised cases above (the result would be the same, of course).
    end
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function irecv_halo!(n::Integer, dim::Integer, F::GGField, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    A, halowidths = F;
    if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            req = MPI.Irecv!(gpurecvbuf_flat(n,dim,i,F), neighbor(n,dim), tag, comm());
        else
            req = MPI.Irecv!(recvbuf_flat(n,dim,i,F), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function isend_halo(n::Integer, dim::Integer, F::GGField, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    A, halowidths = F;
    if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            req = MPI.Isend(gpusendbuf_flat(n,dim,i,F), neighbor(n,dim), tag, comm());
        else
            req = MPI.Isend(sendbuf_flat(n,dim,i,F), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function sendrecv_halo_local(n::Integer, dim::Integer, F::GGField, i::Integer)
    A, halowidths = F;
    if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
        if (cudaaware_MPI(dim) && is_cuarray(A)) || (amdgpuaware_MPI(dim) && is_rocarray(A))
            if n == 1
                gpumemcopy!(gpurecvbuf_flat(2,dim,i,F), gpusendbuf_flat(1,dim,i,F));
            elseif n == 2
                gpumemcopy!(gpurecvbuf_flat(1,dim,i,F), gpusendbuf_flat(2,dim,i,F));
            end
        else
            if n == 1
                memcopy!(recvbuf_flat(2,dim,i,F), sendbuf_flat(1,dim,i,F), use_polyester(dim));
            elseif n == 2
                memcopy!(recvbuf_flat(1,dim,i,F), sendbuf_flat(2,dim,i,F), use_polyester(dim));
            end
        end
    end
end

function memcopy!(dst::AbstractArray{T}, src::AbstractArray{T}, use_polyester::Bool) where T <: GGNumber
    if use_polyester && nthreads() > 1 && length(src) > 1 && !(T <: Complex)  # NOTE: Polyester does not yet support Complex numbers and copy reinterpreted arrays leads to bad performance.
        memcopy_polyester!(dst, src)
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


##-------------------------------------------
## FUNCTIONS FOR CHECKING THE INPUT ARGUMENTS

# NOTE: no comparison must be done between the field-local halowidths and field-local overlaps because any combination is valid: the rational is that a field has simply no halo but only computation overlap in a given dimension if the corresponding local overlap is less than 2 times the local halowidth. This allows to determine whether a halo update needs to be done in a certain dimension or not.
function check_fields(fields::GGField...)
    # Raise an error if any of the given fields has a halowidth less than 1.
    invalid_halowidths = [i for i=1:length(fields) if any([fields[i].halowidths[dim]<1 for dim=1:ndims(fields[i])])];
    if length(invalid_halowidths) > 1
        error("The fields at positions $(join(invalid_halowidths,", "," and ")) have a halowidth less than 1.")
    elseif length(invalid_halowidths) > 0
        error("The field at position $(invalid_halowidths[1]) has a halowidth less than 1.")
    end
    
    # Raise an error if any of the given fields has no halo at all (in any dimension) - in this case there is no halo update to do and including the field in the call is inconsistent.
    no_halo = Int[];
    for i = 1:length(fields)
        A, halowidths = fields[i]
        if all([(ol(dim, A) < 2*halowidths[dim]) for dim = 1:ndims(A)]) # There is no halo if the overlap is less than 2 times the halowidth (only computation overlap in this case)...
            push!(no_halo, i);
        end
    end
    if length(no_halo) > 1
        error("The fields at positions $(join(no_halo,", "," and ")) have no halo; remove them from the call.")
    elseif length(no_halo) > 0
        error("The field at position $(no_halo[1]) has no halo; remove it from the call.")
    end

    # Raise an error if any of the given fields contains any duplicates.
    duplicates = [[i,j] for i=1:length(fields) for j=i+1:length(fields) if fields[i].A===fields[j].A];
    if length(duplicates) > 2
        error("The pairs of fields with the positions $(join(duplicates,", "," and ")) are the same; remove any duplicates from the call.")
    elseif length(duplicates) > 0
        error("The field at position $(duplicates[1][2]) is a duplicate of the one at the position $(duplicates[1][1]); remove the duplicate from the call.")
    end

    # Raise an error if the elements of any field are not of bits type or "is-bits" Union type.
    invalid_types = [i for i=1:length(fields) if !(isbitstype(eltype(fields[i].A)) || Base.isbitsunion(eltype(fields[i].A)))];
    if length(invalid_types) > 1
        error("The fields at positions $(join(invalid_types,", "," and ")) are not of bits type or 'is-bits' Union type.")
    elseif length(invalid_types) > 0
        error("The field at position $(invalid_types[1]) is not of bits type or 'is-bits' Union type.")
    end

    # Raise an error if any of the given fields is non-contiguous (in indexing).
    non_contiguous = [i for i=1:length(fields) if !Base.iscontiguous(fields[i].A)];
    if length(non_contiguous) > 1
        error("The fields at positions $(join(non_contiguous,", "," and ")) are non-contiguous (in indexing).")
    elseif length(non_contiguous) > 0
        error("The field at position $(non_contiguous[1]) is non-contiguous (in indexing)..")
    end

    # Raise an error if any of the given fields does not have a supported array type.
    unsupported_types = [i for i=1:length(fields) if !(is_array(fields[i].A) || is_cuarray(fields[i].A) || is_rocarray(fields[i].A))];
    if length(unsupported_types) > 1
        error("The fields at positions $(join(unsupported_types,", "," and ")) do not have a supported array type.")
    elseif length(unsupported_types) > 0
        error("The field at position $(unsupported_types[1]) does not have a supported array type.")
    end

    # Raise an error if not all fields are of the same datatype (restriction comes from buffer handling).
    different_types = [i for i=2:length(fields) if typeof(fields[i].A)!=typeof(fields[1].A)];
    if length(different_types) > 1
        error("The fields at positions $(join(different_types,", "," and ")) are of different type than the first field; make sure that in a same call all fields are of the same type.")
    elseif length(different_types) == 1
        error("The field at position $(different_types[1]) is of different type than the first field; make sure that in a same call all fields are of the same type.")
    end
end
