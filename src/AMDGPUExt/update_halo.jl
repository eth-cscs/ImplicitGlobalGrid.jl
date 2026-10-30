##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

ImplicitGlobalGrid.free_update_halo_rocbuffers(args...) = free_update_halo_rocbuffers(args...)
ImplicitGlobalGrid.init_rocbufs_arrays(args...) = init_rocbufs_arrays(args...)
ImplicitGlobalGrid.init_rocbufs(args...) = init_rocbufs(args...)
ImplicitGlobalGrid.reinterpret_rocbufs(args...) = reinterpret_rocbufs(args...)
ImplicitGlobalGrid.reallocate_undersized_rocbufs(args...) = reallocate_undersized_rocbufs(args...)
ImplicitGlobalGrid.register_unregistered_rocbufs(args...) = register_unregistered_rocbufs(args...)
ImplicitGlobalGrid.unregister_rocbufs(args...) = unregister_rocbufs(args...)
ImplicitGlobalGrid.get_rocsendbufs_raw(args...) = get_rocsendbufs_raw(args...)
ImplicitGlobalGrid.get_rocrecvbufs_raw(args...) = get_rocrecvbufs_raw(args...)
ImplicitGlobalGrid.gpusendbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpusendbuf(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpurecvbuf(n,dim,i,A)
ImplicitGlobalGrid.gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpusendbuf_flat(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpurecvbuf_flat(n,dim,i,A)

let
    global free_update_halo_rocbuffers, init_rocbufs_arrays, init_rocbufs, reinterpret_rocbufs, register_unregistered_rocbufs, unregister_rocbufs, reallocate_undersized_rocbufs
    global gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat
    rocsendbufs_raw = nothing
    rocrecvbufs_raw = nothing
    rocsendbufs_raw_h = nothing  # The host buffers registered for the GPU (nothing where not registered).
    rocrecvbufs_raw_h = nothing
    retired_hostbufs  = []       # Host buffers whose GPU wrappers were freed, kept alive until their registration is released: AMDGPU releases the registration only once the last wrapper of the memory is finalized and the device is done with it (i.e. possibly later than the call to unsafe_free!); if the memory were freed and reused before, the new owner would find it registered (with the wrong size).
    hipmalloc_ptrs    = Set{Ptr{Cvoid}}()  # Device buffers allocated with hipMalloc by alloc_rocbuf (IGG owns them; freed with hipFree by free_rocbuf!).

    function free_update_halo_rocbuffers()
        if !isnothing(rocsendbufs_raw_h)
            for i = 1:length(rocsendbufs_raw_h), n = 1:NNEIGHBORS_PER_DIM
                unregister_rocbufs(i, n);  # Release the registration of the host buffers, keeping them alive until it is released.
            end
        end
        free_rocbufs(rocsendbufs_raw)
        free_rocbufs(rocrecvbufs_raw)
        reset_roc_buffers()
    end

    function free_rocbufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if is_rocarray(bufs[i][n]) free_rocbuf!(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    # Allocate a zeroed device buffer of nelems elements of type T with plain hipMalloc, i.e. outside the AMDGPU.jl memory pool.
    # NOTE: the device buffers used for ROCm-aware MPI must not come from the AMDGPU.jl memory pool (hipMallocFromPoolAsync, behind ROCArray{T}(undef,...), AMDGPU.zeros, ...): the GPU transport layer of Cray MPICH (libmpi_gtl_hsa) does not recognize pool memory as device memory (IPC path: abort `gtlt_hsa_get_mem_handle: Assertion seg_info != NULL`; non-IPC path: segmentation fault in a CPU memcpy on the device pointer; verified with cray-mpich 9.1.0 / ROCm 7.2.3 on MI300A and MI250X, from Julia and from C). AMDGPU.jl has no option to bypass the pool (unlike CUDA.jl's JULIA_CUDA_MEMORY_POOL=none); once it provides one, this can go back to AMDGPU.zeros for the AMDGPU.jl versions providing it (version check). The wrapping ROCArray does not own the memory (own=false); the buffer is freed explicitly with free_rocbuf!.
    function alloc_rocbuf(T::DataType, nelems::Integer)
        ptr = Ref{Ptr{Cvoid}}(C_NULL)
        AMDGPU.HIP.hipMalloc(ptr, nelems*sizeof(T))
        push!(hipmalloc_ptrs, ptr[])
        buf = unsafe_wrap(ROCArray, Ptr{T}(ptr[]), (Int(nelems),); own=false)
        fill!(buf, zero(T)); AMDGPU.synchronize()  # NOTE: synchronizes the task-local stream, on which fill! was enqueued.
        return buf
    end

    # Free a device buffer allocated with alloc_rocbuf (hipFree, after synchronizing the task-local stream; a reinterpreted wrapper of such a buffer shares its pointer and is handled the same) or allocated by AMDGPU.jl (unsafe_free!).
    # NOTE: the callers (free_update_halo_rocbuffers, reallocate_rocbufs) run when no halo update is in progress, i.e. after all streams used by IGG have been synchronized, and the only reference to the buffer is the slot being freed or replaced (reinterpret_rocbufs rewrites the slot).
    function free_rocbuf!(buf)
        if is_rocarray(buf)
            p = (length(buf) > 0) ? Ptr{Cvoid}(pointer(buf)) : C_NULL
            if (p != C_NULL && p in hipmalloc_ptrs) AMDGPU.synchronize(); AMDGPU.HIP.hipFree(p); delete!(hipmalloc_ptrs, p)
            else AMDGPU.unsafe_free!(buf)
            end
        end
    end

    function reset_roc_buffers()
        rocsendbufs_raw = nothing
        rocrecvbufs_raw = nothing
        rocsendbufs_raw_h = nothing
        rocrecvbufs_raw_h = nothing
        prune_retired_hostbufs();  # NOTE: the retired host buffers whose registration is not yet released are kept.
    end

    function prune_retired_hostbufs()
        filter!(hostbuf_registered, retired_hostbufs);
    end


    # (AMDGPU functions)

    function init_rocbufs_arrays()
        rocsendbufs_raw = Array{Array{Any,1},1}();
        rocrecvbufs_raw = Array{Array{Any,1},1}();
        rocsendbufs_raw_h = Array{Array{Any,1},1}();
        rocrecvbufs_raw_h = Array{Array{Any,1},1}();
    end

    function init_rocbufs(T::DataType, fields::GGField...)
        while (length(rocsendbufs_raw) < length(fields)) push!(rocsendbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        while (length(rocrecvbufs_raw) < length(fields)) push!(rocrecvbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        while (length(rocsendbufs_raw_h) < length(fields)) push!(rocsendbufs_raw_h, [nothing, nothing]); end
        while (length(rocrecvbufs_raw_h) < length(fields)) push!(rocrecvbufs_raw_h, [nothing, nothing]); end
    end

    function reinterpret_rocbufs(T::DataType, i::Integer, n::Integer)
        if (eltype(rocsendbufs_raw[i][n]) != T) rocsendbufs_raw[i][n] = reinterpret(T, rocsendbufs_raw[i][n]); end
        if (eltype(rocrecvbufs_raw[i][n]) != T) rocrecvbufs_raw[i][n] = reinterpret(T, rocrecvbufs_raw[i][n]); end
    end

    function reallocate_undersized_rocbufs(T::DataType, i::Integer, max_halo_elems::Integer)
        if (!isnothing(rocsendbufs_raw) && length(rocsendbufs_raw[i][1]) < max_halo_elems)
            for n = 1:NNEIGHBORS_PER_DIM
                reallocate_rocbufs(T, i, n, max_halo_elems); GC.gc(); # Too small buffers had been replaced with larger ones; free the unused memory immediately.
            end
        end
    end

    function reallocate_rocbufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        free_rocbuf!(rocsendbufs_raw[i][n]); free_rocbuf!(rocrecvbufs_raw[i][n]);  # NOTE: buffers allocated with alloc_rocbuf are not freed by the GC.
        rocsendbufs_raw[i][n] = alloc_rocbuf(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        rocrecvbufs_raw[i][n] = alloc_rocbuf(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    # Register the host buffers of field i for the GPU (page-locked) if they are not registered (anymore), wrapping them into GPU buffers. NOTE: the host buffers can be reinterpreted (see reinterpret_bufs); the registration is pointer based and the GPU buffers get the current element type of the host buffers.
    function register_unregistered_rocbufs(i::Integer, sendbufs_raw, recvbufs_raw)
        for n = 1:NNEIGHBORS_PER_DIM
            if isnothing(rocsendbufs_raw_h[i][n]) rocsendbufs_raw[i][n] = register(ROCArray,sendbufs_raw[i][n]); rocsendbufs_raw_h[i][n] = sendbufs_raw[i][n]; end
            if isnothing(rocrecvbufs_raw_h[i][n]) rocrecvbufs_raw[i][n] = register(ROCArray,recvbufs_raw[i][n]); rocrecvbufs_raw_h[i][n] = recvbufs_raw[i][n]; end
        end
    end

    # Unregister the host buffers n of field i from the GPU if they are registered: freeing the GPU buffers wrapping them releases the registration (own=true, see register), possibly later (see retired_hostbufs); the GPU buffers are replaced with empty GPU buffers (they would be dangling once the host buffers are freed).
    function unregister_rocbufs(i::Integer, n::Integer)
        if (!isnothing(rocsendbufs_raw_h) && i <= length(rocsendbufs_raw_h))
            if !isnothing(rocsendbufs_raw_h[i][n]) T = eltype(rocsendbufs_raw[i][n]); AMDGPU.unsafe_free!(rocsendbufs_raw[i][n]); rocsendbufs_raw[i][n] = ROCArray{T}(undef,0); push!(retired_hostbufs, rocsendbufs_raw_h[i][n]); rocsendbufs_raw_h[i][n] = nothing; end
            if !isnothing(rocrecvbufs_raw_h[i][n]) T = eltype(rocrecvbufs_raw[i][n]); AMDGPU.unsafe_free!(rocrecvbufs_raw[i][n]); rocrecvbufs_raw[i][n] = ROCArray{T}(undef,0); push!(retired_hostbufs, rocrecvbufs_raw_h[i][n]); rocrecvbufs_raw_h[i][n] = nothing; end
            prune_retired_hostbufs();
        end
    end


    # (AMDGPU functions)

    function gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where T <: GGNumber
        return view(rocsendbufs_raw[i][n]::ROCVector{T},1:prod(halosize(dim,A)));
    end

    function gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where T <: GGNumber
        return view(rocrecvbufs_raw[i][n]::ROCVector{T},1:prod(halosize(dim,A)));
    end


    # (GPU functions)

    #TODO: see if remove T here and in other cases for CuArray, ROCArray or Array (but then it does not verify that CuArray/ROCArray is of type GGNumber) or if I should instead change GGArray to GGArrayUnion and create: GGArray = Array{T} where T <: GGNumber  and  GGCuArray = CuArray{T} where T <: GGNumber; This is however more difficult to read and understand for others.
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_rocsendbufs_raw, get_rocrecvbufs_raw
    get_rocsendbufs_raw() = deepcopy(rocsendbufs_raw)
    get_rocrecvbufs_raw() = deepcopy(rocrecvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

function ImplicitGlobalGrid.allocate_rocstreams(fields::GGField...)
    allocate_rocstreams_iwrite(fields...);
    allocate_rocstreams_iread(fields...);
end

ImplicitGlobalGrid.iwrite_sendbufs!(n::Integer, dim::Integer, F::ROCField{T}, i::Integer) where {T <: GGNumber} = iwrite_sendbufs!(n,dim,F,i)
ImplicitGlobalGrid.iread_recvbufs!(n::Integer, dim::Integer, F::ROCField{T}, i::Integer) where {T <: GGNumber} = iread_recvbufs!(n,dim,F,i)
ImplicitGlobalGrid.wait_iwrite(n::Integer, A::ROCField{T}, i::Integer) where {T <: GGNumber} = wait_iwrite(n,A,i)
ImplicitGlobalGrid.wait_iread(n::Integer, A::ROCField{T}, i::Integer) where {T <: GGNumber} = wait_iread(n,A,i)
ImplicitGlobalGrid.rocm_task_local_sync() = AMDGPU.synchronize()

let
    global iwrite_sendbufs!, allocate_rocstreams_iwrite, wait_iwrite

    rocstreams = Array{AMDGPU.HIPStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iwrite(n::Integer, A::ROCField{T}, i::Integer) where T <: GGNumber = AMDGPU.synchronize(rocstreams[n,i]; blocking=true);

    function allocate_rocstreams_iwrite(fields::GGField...)
        if length(fields) > size(rocstreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a ROCField
            rocstreams = [rocstreams [AMDGPU.HIPStream(:high) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(rocstreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, F::ROCField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            # DEBUG: the follow section needs perf testing
            if dim == 1  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the 3-D memcopy does not perform well for this extremely strided case).
                ranges = sendranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @roc gridsize=nblocks groupsize=nthreads stream=rocstreams[n,i] write_d2x!(gpusendbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                sendbuf = amdgpuaware_MPI(dim) ? gpusendbuf_flat(n,dim,i,F) : sendbuf_flat(n,dim,i,F);
                write_d2x_async!(amdgpuaware_MPI(dim), sendbuf, A, sendranges(n,dim,F), rocstreams[n,i]);
            end
        end
    end
end

let
    global iread_recvbufs!, allocate_rocstreams_iread, wait_iread, rocstream_iread

    rocstreams = Array{AMDGPU.HIPStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iread(n::Integer, A::ROCField{T}, i::Integer) where T <: GGNumber = AMDGPU.synchronize(rocstreams[n,i]; blocking=true);

    rocstream_iread(n::Integer, i::Integer) = rocstreams[n,i];  # The stream on which the receive buffer n of field i is unpacked (iread_recvbufs!).

    function allocate_rocstreams_iread(fields::GGField...)
        if length(fields) > size(rocstreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a ROCField
            rocstreams = [rocstreams [AMDGPU.HIPStream(:high) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(rocstreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, F::ROCField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            # DEBUG: the follow section needs perf testing
            if dim == 1  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the 3-D memcopy does not perform well for this extremely strided case).
                ranges = recvranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @roc gridsize=nblocks groupsize=nthreads stream=rocstreams[n,i] read_x2d!(gpurecvbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                recvbuf = amdgpuaware_MPI(dim) ? gpurecvbuf_flat(n,dim,i,F) : recvbuf_flat(n,dim,i,F);
                read_x2d_async!(amdgpuaware_MPI(dim), recvbuf, A, recvranges(n,dim,F), rocstreams[n,i]);
            end
        end
    end

end


# (AMDGPU functions)

# Write to the send buffer on the host or device from the array on the device (d2x).
function ImplicitGlobalGrid.write_d2x!(gpusendbuf::ROCDeviceArray{T}, A::ROCDeviceArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
    ix = (AMDGPU.workgroupIdx().x-1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x + sendrangex[1] - 1
    iy = (AMDGPU.workgroupIdx().y-1) * AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y + sendrangey[1] - 1
    iz = (AMDGPU.workgroupIdx().z-1) * AMDGPU.workgroupDim().z + AMDGPU.workitemIdx().z + sendrangez[1] - 1
    if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
    gpusendbuf[ix-(sendrangex[1]-1),iy-(sendrangey[1]-1),iz-(sendrangez[1]-1)] = A[ix,iy,iz];
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function ImplicitGlobalGrid.read_x2d!(gpurecvbuf::ROCDeviceArray{T}, A::ROCDeviceArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
    ix = (AMDGPU.workgroupIdx().x-1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x + recvrangex[1] - 1
    iy = (AMDGPU.workgroupIdx().y-1) * AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y + recvrangey[1] - 1
    iz = (AMDGPU.workgroupIdx().z-1) * AMDGPU.workgroupDim().z + AMDGPU.workitemIdx().z + recvrangez[1] - 1
    if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
    A[ix,iy,iz] = gpurecvbuf[ix-(recvrangex[1]-1),iy-(recvrangey[1]-1),iz-(recvrangez[1]-1)];
    return nothing
end

# Write to the send buffer on the host or device from the array on the device (d2x).
function ImplicitGlobalGrid.write_d2x_async!(rocmaware::Bool, sendbuf::AbstractArray{T}, A::AnyROCArray{T}, sendranges::Array{UnitRange{T2},1}, rocstream::AMDGPU.HIPStream) where T <: GGNumber where T2 <: Integer
    dstMemoryType = rocmaware ? AMDGPU.Mem.HIPBuffer : AMDGPU.Mem.HostBuffer
    AMDGPU.Mem.unsafe_copy3d!(
        pointer(sendbuf), dstMemoryType,
        pointer(A), AMDGPU.Mem.HIPBuffer,
        length(sendranges[1]), length(sendranges[2]), length(sendranges[3]);
        srcPos=(sendranges[1][1], sendranges[2][1], sendranges[3][1]),
        dstPitch=sizeof(T) * length(sendranges[1]), dstHeight=length(sendranges[2]),
        srcPitch=sizeof(T) * size(A, 1), srcHeight=size(A, 2),
        async=true, stream=rocstream
    )
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function ImplicitGlobalGrid.read_x2d_async!(rocmaware::Bool, recvbuf::AbstractArray{T}, A::AnyROCArray{T}, recvranges::Array{UnitRange{T2},1}, rocstream::AMDGPU.HIPStream) where T <: GGNumber where T2 <: Integer
    srcMemoryType = rocmaware ? AMDGPU.Mem.HIPBuffer : AMDGPU.Mem.HostBuffer
    AMDGPU.Mem.unsafe_copy3d!(
        pointer(A), AMDGPU.Mem.HIPBuffer,
        pointer(recvbuf), srcMemoryType,
        length(recvranges[1]), length(recvranges[2]), length(recvranges[3]);
        dstPos=(recvranges[1][1], recvranges[2][1], recvranges[3][1]),
        dstPitch=sizeof(T) * size(A, 1), dstHeight=size(A, 2),
        srcPitch=sizeof(T) * length(recvranges[1]), srcHeight=length(recvranges[2]),
        async=true, stream=rocstream
    )
    return nothing
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

# Copy the send buffer src into the receive buffer dst of field i (local exchange), enqueued on the stream on which dst is subsequently unpacked (iread_recvbufs!(nr,...)), so that the unpack is ordered after the copy (on the task-local stream, nothing would order them).
# NOTE: a device-to-device copy of raw pointers rather than copyto! under AMDGPU.stream!(f, stream): before AMDGPU.jl 2.10, the latter does not switch the stream (f runs on the task-local stream). The buffers are IGG's own device buffers (allocated by alloc_rocbuf), and AMDGPU.jl does not track the streams using an array, so using raw pointers has no side effects.
function ImplicitGlobalGrid.gpumemcopy!(dst::ROCArray{T}, src::ROCArray{T}, nr::Integer, i::Integer) where T <: GGNumber
    if length(dst) != length(src) error("gpumemcopy!: the receive and send buffers differ in length.") end  # NOTE: should never happen (both are views of the halo size of the same field and dimension).
    GC.@preserve dst src begin
        AMDGPU.HIP.hipMemcpyDtoDAsync(pointer(dst), pointer(src), length(src)*sizeof(T), rocstream_iread(nr,i))
    end
    return dst
end
