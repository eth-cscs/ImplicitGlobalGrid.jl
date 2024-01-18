##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

ImplicitGlobalGrid.free_update_halo_cubuffers(args...) = free_update_halo_cubuffers(args...)
ImplicitGlobalGrid.init_cubufs_arrays(args...) = init_cubufs_arrays(args...)
ImplicitGlobalGrid.init_cubufs(args...) = init_cubufs(args...)
ImplicitGlobalGrid.reinterpret_cubufs(args...) = reinterpret_cubufs(args...)
ImplicitGlobalGrid.reallocate_undersized_cubufs(args...) = reallocate_undersized_cubufs(args...)
ImplicitGlobalGrid.reregister_cubufs(args...) = reregister_cubufs(args...)
ImplicitGlobalGrid.get_cusendbufs_raw(args...) = get_cusendbufs_raw(args...)
ImplicitGlobalGrid.get_curecvbufs_raw(args...) = get_curecvbufs_raw(args...)
ImplicitGlobalGrid.gpusendbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where {T <: GGNumber} = gpusendbuf(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where {T <: GGNumber} = gpurecvbuf(n,dim,i,A)
ImplicitGlobalGrid.gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where {T <: GGNumber} = gpusendbuf_flat(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where {T <: GGNumber} = gpurecvbuf_flat(n,dim,i,A)

let
    global free_update_halo_cubuffers, init_cubufs_arrays, init_cubufs, reinterpret_cubufs, reregister_cubufs, reallocate_undersized_cubufs
    global gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat
    cusendbufs_raw = nothing
    curecvbufs_raw = nothing
    cusendbufs_raw_h = nothing
    curecvbufs_raw_h = nothing

    function free_update_halo_cubuffers()
        free_cubufs(cusendbufs_raw)
        free_cubufs(curecvbufs_raw)
        unregister_cubufs(cusendbufs_raw_h)
        unregister_cubufs(curecvbufs_raw_h)
        reset_cu_buffers()
    end

    function free_cubufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if is_cuarray(bufs[i][n]) CUDA.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    function unregister_cubufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if (isa(bufs[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    function reset_cu_buffers()
        cusendbufs_raw = nothing
        curecvbufs_raw = nothing
        cusendbufs_raw_h = nothing
        curecvbufs_raw_h = nothing
    end


    # (CUDA functions)

    function init_cubufs_arrays()
        cusendbufs_raw = Array{Array{Any,1},1}();
        curecvbufs_raw = Array{Array{Any,1},1}();
        cusendbufs_raw_h = Array{Array{Any,1},1}();
        curecvbufs_raw_h = Array{Array{Any,1},1}();
    end

    function init_cubufs(T::DataType, fields::GGField...)
        while (length(cusendbufs_raw) < length(fields)) push!(cusendbufs_raw, [CuArray{T}(undef,0), CuArray{T}(undef,0)]); end
        while (length(curecvbufs_raw) < length(fields)) push!(curecvbufs_raw, [CuArray{T}(undef,0), CuArray{T}(undef,0)]); end
        while (length(cusendbufs_raw_h) < length(fields)) push!(cusendbufs_raw_h, [[], []]); end
        while (length(curecvbufs_raw_h) < length(fields)) push!(curecvbufs_raw_h, [[], []]); end
    end

    function reinterpret_cubufs(T::DataType, i::Integer, n::Integer)
        if (eltype(cusendbufs_raw[i][n]) != T) cusendbufs_raw[i][n] = reinterpret(T, cusendbufs_raw[i][n]); end
        if (eltype(curecvbufs_raw[i][n]) != T) curecvbufs_raw[i][n] = reinterpret(T, curecvbufs_raw[i][n]); end
    end

    function reallocate_undersized_cubufs(T::DataType, i::Integer, max_halo_elems::Integer)
        if (!isnothing(cusendbufs_raw) && length(cusendbufs_raw[i][1]) < max_halo_elems)
            for n = 1:NNEIGHBORS_PER_DIM
                reallocate_cubufs(T, i, n, max_halo_elems); GC.gc(); # Too small buffers had been replaced with larger ones; free the unused memory immediately.
            end
        end
    end

    function reallocate_cubufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        cusendbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        curecvbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_cubufs(T::DataType, i::Integer, n::Integer, sendbufs_raw, recvbufs_raw)
        if (isa(cusendbufs_raw_h[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(cusendbufs_raw_h[i][n]); cusendbufs_raw_h[i][n] = []; end # It is always initialized registered... if (cusendbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        if (isa(curecvbufs_raw_h[i][n],CUDA.Mem.HostBuffer)) CUDA.Mem.unregister(curecvbufs_raw_h[i][n]); curecvbufs_raw_h[i][n] = []; end # It is always initialized registered... if (curecvbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        cusendbufs_raw[i][n], cusendbufs_raw_h[i][n] = register(CuArray,sendbufs_raw[i][n]);
        curecvbufs_raw[i][n], curecvbufs_raw_h[i][n] = register(CuArray,recvbufs_raw[i][n]);
    end


    # (CUDA functions)

    function gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where T <: GGNumber
        return view(cusendbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
    end

    function gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where T <: GGNumber
        return view(curecvbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
    end


    # (GPU functions)

    #TODO: see if remove T here and in other cases for CuArray, ROCArray or Array (but then it does not verify that CuArray/ROCArray is of type GGNumber) or if I should instead change GGArray to GGArrayUnion and create: GGArray = Array{T} where T <: GGNumber  and  GGCuArray = CuArray{T} where T <: GGNumber; This is however more difficult to read and understand for others.
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T}) where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_cusendbufs_raw, get_curecvbufs_raw
    get_cusendbufs_raw()  = deepcopy(cusendbufs_raw)
    get_curecvbufs_raw()  = deepcopy(curecvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

function ImplicitGlobalGrid.allocate_custreams(fields::GGField...)
    allocate_custreams_iwrite(fields...);
    allocate_custreams_iread(fields...);
end

let
    global iwrite_sendbufs!, allocate_custreams_iwrite, wait_iwrite

    custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iwrite(n::Integer, A::CuField{T}, i::Integer) where T <: GGNumber = CUDA.synchronize(custreams[n,i]);

    function allocate_custreams_iwrite(fields::GGField...)
        if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuField
            custreams = [custreams [CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, F::CuField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            if dim == 1 || cudaaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = sendranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] write_d2x!(gpusendbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                write_d2h_async!(sendbuf_flat(n,dim,i,F), A, sendranges(n,dim,F), custreams[n,i]);
            end
        end
    end
end

let
    global iread_recvbufs!, allocate_custreams_iread, wait_iread

    custreams = Array{CuStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iread(n::Integer, A::CuField{T}, i::Integer) where T <: GGNumber = CUDA.synchronize(custreams[n,i]);

    function allocate_custreams_iread(fields::GGField...)
        if length(fields) > size(custreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuField
            custreams = [custreams [CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(custreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, F::CuField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            if dim == 1 || cudaaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = recvranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=custreams[n,i] read_x2d!(gpurecvbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                read_h2d_async!(recvbuf_flat(n,dim,i,F), A, recvranges(n,dim,F), custreams[n,i]);
            end
        end
    end
end


# (CUDA functions)

# Write to the send buffer on the host or device from the array on the device (d2x).
function ImplicitGlobalGrid.write_d2x!(gpusendbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x + sendrangex[1] - 1
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y + sendrangey[1] - 1
    iz = (CUDA.blockIdx().z-1) * CUDA.blockDim().z + CUDA.threadIdx().z + sendrangez[1] - 1
    if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
    gpusendbuf[ix-(sendrangex[1]-1),iy-(sendrangey[1]-1),iz-(sendrangez[1]-1)] = A[ix,iy,iz];
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function ImplicitGlobalGrid.read_x2d!(gpurecvbuf::CuDeviceArray{T}, A::CuDeviceArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x + recvrangex[1] - 1
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y + recvrangey[1] - 1
    iz = (CUDA.blockIdx().z-1) * CUDA.blockDim().z + CUDA.threadIdx().z + recvrangez[1] - 1
    if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
    A[ix,iy,iz] = gpurecvbuf[ix-(recvrangex[1]-1),iy-(recvrangey[1]-1),iz-(recvrangez[1]-1)];
    return nothing
end

# Write to the send buffer on the host from the array on the device (d2h).
function ImplicitGlobalGrid.write_d2h_async!(sendbuf::AbstractArray{T}, A::CuArray{T}, sendranges::Array{UnitRange{T2},1}, custream::CuStream) where T <: GGNumber where T2 <: Integer
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
function ImplicitGlobalGrid.read_h2d_async!(recvbuf::AbstractArray{T}, A::CuArray{T}, recvranges::Array{UnitRange{T2},1}, custream::CuStream) where T <: GGNumber where T2 <: Integer
    CUDA.Mem.unsafe_copy3d!(
        pointer(A), CUDA.Mem.Device, pointer(recvbuf), CUDA.Mem.Host,
        length(recvranges[1]), length(recvranges[2]), length(recvranges[3]);
        dstPos=(recvranges[1][1], recvranges[2][1], recvranges[3][1]),
        srcPitch=sizeof(T)*length(recvranges[1]), srcHeight=length(recvranges[2]),
        dstPitch=sizeof(T)*size(A,1), dstHeight=size(A,2),
        async=true, stream=custream
    )
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function ImplicitGlobalGrid.gpumemcopy!(dst::CuArray{T}, src::CuArray{T}) where T <: GGNumber
    @inbounds CUDA.copyto!(dst, src)
end

