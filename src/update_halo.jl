export update_halo!

import MPI
@static if ENABLE_CUDA
    using CUDAdrv, CUDAnative, CuArrays
end
using Base.Threads


"""
    update_halo!(A)
    update_halo!(A...)

Update the halo of the given GPU/CPU-array(s).

# Typical use cases:
    update_halo!(A)        # Update the halo of the array A.
    update_halo!(A, B, C)  # Update the halos of the arrays A, B and C.

!!! note "Performance note"
    Group subsequent calls to `update_halo!` in a single call for better performance (enables additional pipelining).
    Consider activating CUDA-aware MPI (see [`ImplicitGlobalGrid`](@ref)).
"""
function update_halo!(A::GGArray...)
    check_initialized();
    check_fields(A...);
    _update_halo!(A...);  # Asignment of A to fields in the internal function _update_halo!() as vararg A can consist of multiple fields; A will be used for a single field in the following (The args of update_halo! must however be "A..." for maximal simplicity and elegance for the user).
    return nothing
end

function _update_halo!(fields::GGArray...)
    allocate_bufs(fields...);
    if any_array(fields...) allocate_tasks(fields...); end
    @enable_if_cuda if any_cuarray(fields...) allocate_custreams(fields...); end

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

let
    global free_update_halo_buffers, allocate_bufs, sendbuf, recvbuf, sendbuf_flat, recvbuf_flat, cusendbuf, curecvbuf, cusendbuf_flat, curecvbuf_flat
    sendbufs_raw = nothing
    recvbufs_raw = nothing
    cusendbufs_raw = nothing
    curecvbufs_raw = nothing
    cusendbufs_raw_h = nothing
    curecvbufs_raw_h = nothing

    function free_update_halo_buffers()
        @enable_if_cuda if any(cudaaware_MPI()) free_cubufs(cusendbufs_raw) end
        @enable_if_cuda if any(cudaaware_MPI()) free_cubufs(curecvbufs_raw) end
        @enable_if_cuda if none(cudaaware_MPI()) unregister_cubufs(cusendbufs_raw_h) end
        @enable_if_cuda if none(cudaaware_MPI()) unregister_cubufs(curecvbufs_raw_h) end
        sendbufs_raw = nothing;
        recvbufs_raw = nothing;
        cusendbufs_raw = nothing
        curecvbufs_raw = nothing
        cusendbufs_raw_h = nothing
        curecvbufs_raw_h = nothing
        GC.gc();
    end

    @static if ENABLE_CUDA
        function free_cubufs(bufs)
            if (bufs !== nothing)
                for i = 1:length(bufs)
                    for n = 1:length(bufs[i])
                        if is_cuarray(bufs[i][n]) CuArrays.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end
                    end
                end
            end
        end
        function unregister_cubufs(bufs)
            if (bufs !== nothing)
                for i = 1:length(bufs)
                    for n = 1:length(bufs[i])
                        if (isa(bufs[i][n],CUDAdrv.Mem.HostBuffer)) CUDAdrv.Mem.unregister(bufs[i][n]); bufs[i][n] = []; end
                    end
                end
            end
        end
    end

    # Allocate for each field two send and recv buffers (one for the left and one for the right neighbour of a dimension). The required length of the buffer is given by the maximal number of halo elements in any of the dimensions. Note that buffers are not allocated separately for each dimension, as the updates are performed one dimension at a time (required for correctness).
    function allocate_bufs(fields::GGArray{T}...) where T <: GGNumber
        if (isnothing(sendbufs_raw) || isnothing(recvbufs_raw))
            free_update_halo_buffers();
            init_bufs_arrays();
            @enable_if_cuda init_cubufs_arrays();
        end
        init_bufs(T, fields...);
        @enable_if_cuda init_cubufs(T, fields...);
        for i = 1:length(fields)
            A = fields[i];
            for n = 1:NNEIGHBORS_PER_DIM # Ensure that the buffers are interpreted to contain elements of the same type as the array.
                reinterpret_bufs(T, i, n);
                @enable_if_cuda reinterpret_cubufs(T, i, n);
            end
            max_halo_elems = (ndims(A) > 1) ? prod(sort([size(A)...])[2:end]) : 1;
            if (length(sendbufs_raw[i][1]) < max_halo_elems)
                for n = 1:NNEIGHBORS_PER_DIM
                    reallocate_bufs(T, i, n, max_halo_elems);
                    @enable_if_cuda if (is_cuarray(A) && none(cudaaware_MPI())) reregister_cubufs(T, i, n); end
                end
                GC.gc(); # Too small buffers had been replaced with larger ones; free the now unused memory.
            end
            if (length(cusendbufs_raw[i][1]) < max_halo_elems)
                for n = 1:NNEIGHBORS_PER_DIM
                    @enable_if_cuda if (is_cuarray(A) &&  any(cudaaware_MPI())) reallocate_cubufs(T, i, n, max_halo_elems); GC.gc(); end # Too small buffers had been replaced with larger ones; free the unused memory immediately.
                end
            end
        end
    end

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

    @static if ENABLE_CUDA
        function init_cubufs_arrays()
            cusendbufs_raw = Array{Array{Any,1},1}();
            curecvbufs_raw = Array{Array{Any,1},1}();
            cusendbufs_raw_h = Array{Array{Any,1},1}();
            curecvbufs_raw_h = Array{Array{Any,1},1}();
        end

        function init_cubufs(T::DataType, fields::GGArray...)
            while (length(cusendbufs_raw) < length(fields)) push!(cusendbufs_raw, [cuzeros(T,0), cuzeros(T,0)]); end
            while (length(curecvbufs_raw) < length(fields)) push!(curecvbufs_raw, [cuzeros(T,0), cuzeros(T,0)]); end
            while (length(cusendbufs_raw_h) < length(fields)) push!(cusendbufs_raw_h, [[], []]); end
            while (length(curecvbufs_raw_h) < length(fields)) push!(curecvbufs_raw_h, [[], []]); end
        end

        function reinterpret_cubufs(T::DataType, i::Integer, n::Integer)
            if (eltype(cusendbufs_raw[i][n]) != T) cusendbufs_raw[i][n] = reinterpret(T, cusendbufs_raw[i][n]); end
            if (eltype(curecvbufs_raw[i][n]) != T) curecvbufs_raw[i][n] = reinterpret(T, curecvbufs_raw[i][n]); end
        end

        function reallocate_cubufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
            cusendbufs_raw[i][n] = cuzeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
            curecvbufs_raw[i][n] = cuzeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
        end

        function reregister_cubufs(T::DataType, i::Integer, n::Integer)
            if (isa(cusendbufs_raw_h[i][n],CUDAdrv.Mem.HostBuffer)) CUDAdrv.Mem.unregister(cusendbufs_raw_h[i][n]); cusendbufs_raw_h[i][n] = []; end # It is always initialized registered... if (cusendbufs_raw_h[i][n].bytesize > 32*sizeof(T))
            if (isa(curecvbufs_raw_h[i][n],CUDAdrv.Mem.HostBuffer)) CUDAdrv.Mem.unregister(curecvbufs_raw_h[i][n]); cusendbufs_raw_h[i][n] = []; end # It is always initialized registered... if (curecvbufs_raw_h[i][n].bytesize > 32*sizeof(T))
            cusendbufs_raw[i][n], cusendbufs_raw_h[i][n] = register(sendbufs_raw[i][n]);
            curecvbufs_raw[i][n], curecvbufs_raw_h[i][n] = register(recvbufs_raw[i][n]);
        end
    end

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

    @static if ENABLE_CUDA
        function cusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
            return view(cusendbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
            #return view(sendbufs_raw[i][n],1:prod(halosize));
        end

        function curecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
            return view(curecvbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
            #return view(recvbufs_raw[i][n],1:prod(halosize));
        end

        #TODO: see if I should remove T here and in other cases for CuArray or Array (but then it does not verify that CuArray is of type GGNumber) or if I should instead change GGArray to GGArrayUnion and create: GGArray = Array{T} where T <: GGNumber  and  GGCuArray = CuArray{T} where T <: GGNumber; This is however more difficult to read and understand for others.
        function cusendbuf(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
            return reshape(cusendbuf_flat(n,dim,i,A), halosize(dim,A));
            #return view(sendbufs_raw[i][n],1:prod(halosize));
        end

        function curecvbuf(n::Integer, dim::Integer, i::Integer, A::CuArray{T}) where T <: GGNumber
            return reshape(curecvbuf_flat(n,dim,i,A), halosize(dim,A));
            #return view(recvbufs_raw[i][n],1:prod(halosize));
        end
    end

    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_sendbufs_raw, get_recvbufs_raw, get_cusendbufs_raw, get_curecvbufs_raw
    get_sendbufs_raw()   = deepcopy(sendbufs_raw)
    get_recvbufs_raw()   = deepcopy(recvbufs_raw)
    get_cusendbufs_raw() = deepcopy(cusendbufs_raw)
    get_curecvbufs_raw() = deepcopy(curecvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

# NOTE: the tasks and custreams are stored here in a let clause to have them survive the end of a call to update_boundaries. This avoids the creation of new tasks and cuda streams every time. Besides, that this could be relevant for performance, it is important for debugging the overlapping the communication with computation (if at every call new stream/task objects are created this becomes very messy and hard to analyse).

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

    wait_iread(n::Integer, A::Array{T}, i::Integer) where T <: GGNumber = (schedule(tasks[n,i]); wait(tasks[n,i]);) # The argument A is used for multiple dispatch. #NOTE: The current implementation only starts a task when it is waited for, in order to make sure that only one task is run at a time and that they are run in the desired order (best for performance currently as the tasks are mapped only to one thread via context switching).

    function allocate_tasks_iread(fields::GGArray...)
        if length(fields) > size(tasks,2)  # Note: for simplicity, we create a tasks for every field even if it is not an Array
            tasks = [tasks Array{Task}(undef, NNEIGHBORS_PER_DIM, length(fields)-size(tasks,2))];  # Create (additional) emtpy tasks.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, A::Array{T}, i::Integer) where T <: GGNumber  # Function to be called if A is a CPU array.
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


@static if ENABLE_CUDA
    function allocate_custreams(fields::GGArray...)
        allocate_custreams_iwrite(fields...);
        allocate_custreams_iread(fields...);
    end

    let
        global iwrite_sendbufs!, allocate_custreams_iwrite, wait_iwrite

        custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

        wait_iwrite(n::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber = synchronize(custreams[n,i]); # The argument A is used for multiple dispatch.

        function allocate_custreams_iwrite(fields::GGArray...)
    	    if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuArray
                custreams = [custreams [CuStream(streamPriorityRange()[2], CUDAdrv.STREAM_NON_BLOCKING) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
            end
        end

        function iwrite_sendbufs!(n::Integer, dim::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber  # Function to be called if A is a GPU array.
            if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
                if dim == 1 || cudaaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                    ranges = sendranges(n, dim, A);
                    nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                    halosize = [r[end] - r[1] + 1 for r in ranges];
                    nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                    @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] write_d2x!(cusendbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim);
                else
                    write_d2h_async!(sendbuf_flat(n,dim,i,A), A, sendranges(n,dim,A), dim, custreams[n,i]);
                end
            end
        end
    end

    let
        global iread_recvbufs!, allocate_custreams_iread, wait_iread

        custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

        wait_iread(n::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber = synchronize(custreams[n,i]); # The argument A is used for multiple dispatch.

        function allocate_custreams_iread(fields::GGArray...)
    	    if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuArray
                custreams = [custreams [CuStream(streamPriorityRange()[2], CUDAdrv.STREAM_NON_BLOCKING) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
            end
        end

        function iread_recvbufs!(n::Integer, dim::Integer, A::CuArray{T}, i::Integer) where T <: GGNumber # Function to be called if A is a GPU array.
            if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
                if dim == 1 || cudaaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                    ranges = recvranges(n, dim, A);
                    nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                    halosize = [r[end] - r[1] + 1 for r in ranges];
                    nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                    @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] read_x2d!(curecvbuf(n,dim,i,A), A, ranges[1], ranges[2], ranges[3], dim);
                else
                    read_h2d_async!(recvbuf_flat(n,dim,i,A), A, recvranges(n,dim,A), dim, custreams[n,i]);
                end
            end
        end
    end
end

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

# Write to the send buffer on the host from the array on the host (h2h). Note: it works for 1D-3D, as sendranges contains always 3 ranges independently of the number of dimensions of A (see function sendranges).
function write_h2h!(sendbuf::AbstractArray{T}, A::Array{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    if nthreads() > 1
        if dim == 1
            @threads for iz in sendranges[3]
                @threads for iy in sendranges[2]
                    for ix in sendranges[1]
                        sendbuf[iy,iz] = A[ix,iy,iz];
                    end
                end
            end
        elseif dim == 2
            @threads for iz in sendranges[3]
                for iy in sendranges[2]
                    @threads for ix in sendranges[1]
                        sendbuf[ix,iz] = A[ix,iy,iz];
                    end
                end
            end
        elseif dim == 3
            for iz in sendranges[3]
                @threads for iy in sendranges[2]
                    @threads for ix in sendranges[1]
                        sendbuf[ix,iy] = A[ix,iy,iz];
                    end
                end
            end
        end
    else
        ix = sendranges[1];
        iy = sendranges[2];
        iz = sendranges[3];
        if     (dim == 1) sendbuf[iy,iz] .= reshape(A[ix,iy,iz], length.(sendranges[2:3])...);
        elseif (dim == 2) sendbuf[ix,iz] .= reshape(A[ix,iy,iz], length.(sendranges[[1,3]])...);
        elseif (dim == 3) sendbuf[ix,iy] .= reshape(A[ix,iy,iz], length.(sendranges[1:2])...);
        end
    end
end

# Read from the receive buffer on the host and store on the array on the host (h2h). This function could be written simpler using vectorized notation. It is written as is to achieve a maximal similarity to the function read_d2h!. Note: it works for 1D-3D, as recvranges contains always 3 ranges independently of the number of dimensions of A (see function recvranges).
function read_h2h!(recvbuf::AbstractArray{T}, A::Array{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer) where T <: GGNumber where T2 <: Integer
    if nthreads() > 1
        if dim == 1
            @threads for iz in recvranges[3]
                @threads for iy in recvranges[2]
                    for ix in recvranges[1]
                        A[ix,iy,iz] = recvbuf[iy,iz];
                    end
                end
            end
        elseif dim == 2
            @threads for iz in recvranges[3]
                for iy in recvranges[2]
                    @threads for ix in recvranges[1]
                        A[ix,iy,iz] = recvbuf[ix,iz];
                    end
                end
            end
        elseif dim == 3
            for iz in recvranges[3]
                @threads for iy in recvranges[2]
                    @threads for ix in recvranges[1]
                        A[ix,iy,iz] = recvbuf[ix,iy];
                    end
                end
            end
        end
    else
        ix = recvranges[1];
        iy = recvranges[2];
        iz = recvranges[3];
        if     (dim == 1) A[ix,iy,iz] .= reshape(recvbuf[iy,iz], length.(recvranges)...);
        elseif (dim == 2) A[ix,iy,iz] .= reshape(recvbuf[ix,iz], length.(recvranges)...);
        elseif (dim == 3) A[ix,iy,iz] .= reshape(recvbuf[ix,iy], length.(recvranges)...);
        end
    end
end

@static if ENABLE_CUDA
    # Write to the send buffer on the host or device from the array on the device (d2x).
    function write_d2x!(cusendbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
        ix = (blockIdx().x-1) * blockDim().x + threadIdx().x + sendrangex[1] - 1
        iy = (blockIdx().y-1) * blockDim().y + threadIdx().y + sendrangey[1] - 1
        iz = (blockIdx().z-1) * blockDim().z + threadIdx().z + sendrangez[1] - 1
        if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
        if     (dim == 1) cusendbuf[iy,iz] = A[ix,iy,iz];
        elseif (dim == 2) cusendbuf[ix,iz] = A[ix,iy,iz];
        elseif (dim == 3) cusendbuf[ix,iy] = A[ix,iy,iz];
        end
        return nothing
    end

    # Read from the receive buffer on the host or device and store on the array on the device (x2d).
    function read_x2d!(curecvbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
        ix = (blockIdx().x-1) * blockDim().x + threadIdx().x + recvrangex[1] - 1
        iy = (blockIdx().y-1) * blockDim().y + threadIdx().y + recvrangey[1] - 1
        iz = (blockIdx().z-1) * blockDim().z + threadIdx().z + recvrangez[1] - 1
        if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
        if     (dim == 1) A[ix,iy,iz] = curecvbuf[iy,iz];
        elseif (dim == 2) A[ix,iy,iz] = curecvbuf[ix,iz];
        elseif (dim == 3) A[ix,iy,iz] = curecvbuf[ix,iy];
        end
        return nothing
    end

    # Write to the send buffer on the host from the array on the device (d2h).
    function write_d2h_async!(sendbuf::AbstractArray{T}, A::CuArray{T}, sendranges::Array{UnitRange{T2},1}, dim::Integer, custream::CuStream) where T <: GGNumber where T2 <: Integer
        params = Ref(CuMemcpy3D_st(
            # Source
            sizeof(T)*(sendranges[1][1]-1), sendranges[2][1]-1, sendranges[3][1]-1,        # srcXInBytes, srcY, srcZ
            0,                                                                             # srcLOD
            MEMORYTYPE_DEVICE,                                                             # srcMemoryType
            C_NULL, pointer(A),                                                            # srcHost, srcDevice
            C_NULL, C_NULL,                                                                # srcArray, reserved0
            size(A,1)*sizeof(T), size(A,2),                                                # srcPitch, srcHeight
            # Destination
            0, 0, 0,                                                                       # dstXInBytes, dstY, dstZ
            0,                                                                             # dstLOD
            MEMORYTYPE_HOST,                                                               # dstMemoryType
            pointer(sendbuf), CU_NULL,                                                     # dstHost, dstDevice
            C_NULL, C_NULL,                                                                # dstArray, reserved1
            sizeof(T)*length(sendranges[1]), length(sendranges[2]),                        # dstPitch, dstHeight
            # Extent of copy
            sizeof(T)*length(sendranges[1]), length(sendranges[2]), length(sendranges[3])  # WidthInBytes, Height, Depth
        ));
        CUDAdrv.@apicall(:cuMemcpy3DAsync, (Ptr{CuMemcpy3D_st}, CuStream_t), params, custream.handle);
    end

    # Read from the receive buffer on the host and store on the array on the device (h2d).  #TODO: SEE how to make work for 1D-3D and make comment here.
    function read_h2d_async!(recvbuf::AbstractArray{T}, A::CuArray{T}, recvranges::Array{UnitRange{T2},1}, dim::Integer, custream::CuStream) where T <: GGNumber where T2 <: Integer
        params = Ref(CuMemcpy3D_st(
            # Source
            0, 0, 0,                                                                       # srcXInBytes, srcY, srcZ
            0,                                                                             # srcLOD
            MEMORYTYPE_HOST,                                                               # srcMemoryType
            pointer(recvbuf), CU_NULL,                                                     # srcHost, srcDevice
            C_NULL, C_NULL,                                                                # srcArray, reserved0
            sizeof(T)*length(recvranges[1]), length(recvranges[2]),                        # srcPitch, srcHeight
            # Destination
            sizeof(T)*(recvranges[1][1]-1), recvranges[2][1]-1, recvranges[3][1]-1,        # dstXInBytes, dstY, dstZ
            0,                                                                             # dstLOD
            MEMORYTYPE_DEVICE,                                                             # dstMemoryType
            C_NULL, pointer(A),                                                            # dstHost, dstDevice
            C_NULL, C_NULL,                                                                # dstArray, reserved1
            size(A,1)*sizeof(T), size(A,2),                                                # dstPitch, dstHeight
            # Extent of copy
            sizeof(T)*length(recvranges[1]), length(recvranges[2]), length(recvranges[3])  # WidthInBytes, Height, Depth
        ));
        CUDAdrv.@apicall(:cuMemcpy3DAsync, (Ptr{CuMemcpy3D_st}, CuStream_t), params, custream.handle);
    end
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function irecv_halo!(n::Integer, dim::Integer, A::GGArray, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if cudaaware_MPI(dim) && is_cuarray(A)
            @enable_if_cuda req = MPI.Irecv!(curecvbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        else
            req = MPI.Irecv!(recvbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function isend_halo(n::Integer, dim::Integer, A::GGArray, i::Integer; tag::Integer=0)
    req = MPI.REQUEST_NULL;
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if cudaaware_MPI(dim) && is_cuarray(A)
            @enable_if_cuda req = MPI.Isend(cusendbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        else
            req = MPI.Isend(sendbuf_flat(n,dim,i,A), neighbor(n,dim), tag, comm());
        end
    end
    return req
end

function sendrecv_halo_local(n::Integer, dim::Integer, A::GGArray, i::Integer)
    if ol(dim,A) >= 2  # There is only a halo and thus a halo update if the overlap is at least 2...
        if cudaaware_MPI(dim) && is_cuarray(A)
            if n == 1
                @enable_if_cuda cumemcopy!(curecvbuf_flat(2,dim,i,A), cusendbuf_flat(1,dim,i,A));
            elseif n == 2
                @enable_if_cuda cumemcopy!(curecvbuf_flat(1,dim,i,A), cusendbuf_flat(2,dim,i,A));
            end
        else
            if n == 1
                memcopy!(recvbuf_flat(2,dim,i,A), sendbuf_flat(1,dim,i,A));
            elseif n == 2
                memcopy!(recvbuf_flat(1,dim,i,A), sendbuf_flat(2,dim,i,A));
            end
        end
    end
end

function memcopy!(dst::AbstractArray{T}, src::AbstractArray{T}) where T <: GGNumber
	if nthreads() > 1
		@threads for ix = 1:length(dst)  # NOTE: Set the number of threads e.g. as: export JULIA_NUM_THREADS=12
		    @inbounds dst[ix] = src[ix]  # NOTE: We fix here exceptionally the use of @inbounds as this copy between two flat vectors (which must have the right length) is considered safe.
		end
	else
		@inbounds dst .= src
	end
end

@static if ENABLE_CUDA
    function cumemcopy!(dst::CuArray{T}, src::CuArray{T}) where T <: GGNumber
    	@inbounds dst .= src
    end
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
end
