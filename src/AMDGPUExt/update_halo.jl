##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

ImplicitGlobalGrid.free_update_halo_rocbuffers(args...) = free_update_halo_rocbuffers(args...)
ImplicitGlobalGrid.init_rocbufs_arrays(args...) = init_rocbufs_arrays(args...)
ImplicitGlobalGrid.init_rocbufs(args...) = init_rocbufs(args...)
ImplicitGlobalGrid.reinterpret_rocbufs(args...) = reinterpret_rocbufs(args...)
ImplicitGlobalGrid.reallocate_undersized_rocbufs(args...) = reallocate_undersized_rocbufs(args...)
ImplicitGlobalGrid.reregister_rocbufs(args...) = reregister_rocbufs(args...)
ImplicitGlobalGrid.get_rocsendbufs_raw(args...) = get_rocsendbufs_raw(args...)
ImplicitGlobalGrid.get_rocrecvbufs_raw(args...) = get_rocrecvbufs_raw(args...)
ImplicitGlobalGrid.gpusendbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpusendbuf(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpurecvbuf(n,dim,i,A)
ImplicitGlobalGrid.gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpusendbuf_flat(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::ROCField{T}) where {T <: GGNumber} = gpurecvbuf_flat(n,dim,i,A)

let
    global free_update_halo_rocbuffers, init_rocbufs_arrays, init_rocbufs, reinterpret_rocbufs, reregister_rocbufs, reallocate_undersized_rocbufs
    global gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat
    rocsendbufs_raw = nothing
    rocrecvbufs_raw = nothing
    # INFO: no need for roc host buffers

    function free_update_halo_rocbuffers()
        free_rocbufs(rocsendbufs_raw)
        free_rocbufs(rocrecvbufs_raw)
        # INFO: no need for roc host buffers
        reset_roc_buffers()
    end

    function free_rocbufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if is_rocarray(bufs[i][n]) AMDGPU.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end # DEBUG: unsafe_free should be managed in AMDGPU
                end
            end
        end
    end

    # INFO: no need for roc host buffers
    # function unregister_rocbufs(bufs)
    # end

    function reset_roc_buffers()
        rocsendbufs_raw = nothing
        rocrecvbufs_raw = nothing
        # INFO: no need for roc host buffers
    end


    # (AMDGPU functions)

    function init_rocbufs_arrays()
        rocsendbufs_raw = Array{Array{Any,1},1}();
        rocrecvbufs_raw = Array{Array{Any,1},1}();
        # INFO: no need for roc host buffers
    end

    function init_rocbufs(T::DataType, fields::GGField...)
        while (length(rocsendbufs_raw) < length(fields)) push!(rocsendbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        while (length(rocrecvbufs_raw) < length(fields)) push!(rocrecvbufs_raw, [ROCArray{T}(undef,0), ROCArray{T}(undef,0)]); end
        # INFO: no need for roc host buffers
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
        rocsendbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        rocrecvbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_rocbufs(T::DataType, i::Integer, n::Integer, sendbufs_raw, recvbufs_raw)
        # INFO: no need for roc host buffers
        rocsendbufs_raw[i][n] = register(ROCArray,sendbufs_raw[i][n]);
        rocrecvbufs_raw[i][n] = register(ROCArray,recvbufs_raw[i][n]);
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
            # DEBUG 2: commenting read_h2d_async! for now
            # if dim == 1 || amdgpuaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = sendranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @roc gridsize=nblocks groupsize=nthreads stream=rocstreams[n,i] write_d2x!(gpusendbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            # else
            #     write_d2h_async!(sendbuf_flat(n,dim,i,F), A, sendranges(n,dim,F), rocstreams[n,i]);
            # end
        end
    end
end

let
    global iread_recvbufs!, allocate_rocstreams_iread, wait_iread

    rocstreams = Array{AMDGPU.HIPStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iread(n::Integer, A::ROCField{T}, i::Integer) where T <: GGNumber = AMDGPU.synchronize(rocstreams[n,i]; blocking=true);

    function allocate_rocstreams_iread(fields::GGField...)
        if length(fields) > size(rocstreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a ROCField
            rocstreams = [rocstreams [AMDGPU.HIPStream(:high) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(rocstreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, F::ROCField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            # DEBUG: the follow section needs perf testing
            # DEBUG 2: commenting read_h2d_async! for now
            # if dim == 1 || amdgpuaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = recvranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @roc gridsize=nblocks groupsize=nthreads stream=rocstreams[n,i] read_x2d!(gpurecvbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            # else
            #     read_h2d_async!(recvbuf_flat(n,dim,i,F), A, recvranges(n,dim,F), rocstreams[n,i]);
            # end
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

# Write to the send buffer on the host from the array on the device (d2h).
function ImplicitGlobalGrid.write_d2h_async!(sendbuf::AbstractArray{T}, A::AnyROCArray{T}, sendranges::Array{UnitRange{T2},1}, rocstream::AMDGPU.HIPStream) where T <: GGNumber where T2 <: Integer
    buf_view = reshape(sendbuf, Tuple(length.(sendranges)))
    AMDGPU.Mem.unsafe_copy3d!(
        pointer(sendbuf), AMDGPU.Mem.HostBuffer,
        pointer(A), typeof(A.buf),
        length(sendranges[1]), length(sendranges[2]), length(sendranges[3]);
        srcPos=(sendranges[1][1], sendranges[2][1], sendranges[3][1]),
        dstPitch=sizeof(T) * size(buf_view, 1), dstHeight=size(buf_view, 2),
        srcPitch=sizeof(T) * size(A, 1), srcHeight=size(A, 2),
        async=true, stream=rocstream
    )
    return nothing
end

# Read from the receive buffer on the host and store on the array on the device (h2d).
function ImplicitGlobalGrid.read_h2d_async!(recvbuf::AbstractArray{T}, A::AnyROCArray{T}, recvranges::Array{UnitRange{T2},1}, rocstream::AMDGPU.HIPStream) where T <: GGNumber where T2 <: Integer
    buf_view = reshape(recvbuf, Tuple(length.(recvranges)))
    AMDGPU.Mem.unsafe_copy3d!(
        pointer(A), typeof(A.buf),
        pointer(recvbuf), AMDGPU.Mem.HostBuffer,
        length(recvranges[1]), length(recvranges[2]), length(recvranges[3]);
        dstPos=(recvranges[1][1], recvranges[2][1], recvranges[3][1]),
        dstPitch=sizeof(T) * size(A, 1), dstHeight=size(A, 2),
        srcPitch=sizeof(T) * size(buf_view, 1), srcHeight=size(buf_view, 2),
        async=true, stream=rocstream
    )
    return nothing
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function ImplicitGlobalGrid.gpumemcopy!(dst::ROCArray{T}, src::ROCArray{T}) where T <: GGNumber
    @inbounds AMDGPU.copyto!(dst, src)
end
