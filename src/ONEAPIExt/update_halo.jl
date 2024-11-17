##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

ImplicitGlobalGrid.free_update_halo_onebuffers(args...) = free_update_halo_onebuffers(args...)
ImplicitGlobalGrid.init_onebufs_arrays(args...) = init_onebufs_arrays(args...)
ImplicitGlobalGrid.init_onebufs(args...) = init_onebufs(args...)
ImplicitGlobalGrid.reinterpret_onebufs(args...) = reinterpret_onebufs(args...)
ImplicitGlobalGrid.reallocate_undersized_onebufs(args...) = reallocate_undersized_onebufs(args...)
ImplicitGlobalGrid.reregister_onebufs(args...) = reregister_onebufs(args...)
ImplicitGlobalGrid.get_onesendbufs_raw(args...) = get_onesendbufs_raw(args...)
ImplicitGlobalGrid.get_onerecvbufs_raw(args...) = get_onerecvbufs_raw(args...)
ImplicitGlobalGrid.gpusendbuf(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where {T <: GGNumber} = gpusendbuf(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where {T <: GGNumber} = gpurecvbuf(n,dim,i,A)
ImplicitGlobalGrid.gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where {T <: GGNumber} = gpusendbuf_flat(n,dim,i,A)
ImplicitGlobalGrid.gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where {T <: GGNumber} = gpurecvbuf_flat(n,dim,i,A)

let
    global free_update_halo_onebuffers, init_onebufs_arrays, init_onebufs, reinterpret_onebufs, reregister_onebufs, reallocate_undersized_onebufs
    global gpusendbuf, gpurecvbuf, gpusendbuf_flat, gpurecvbuf_flat
    onesendbufs_raw = nothing
    onerecvbufs_raw = nothing
    onesendbufs_raw_h = nothing
    onerecvbufs_raw_h = nothing

    function free_update_halo_onebuffers()
        free_onebufs(onesendbufs_raw)
        free_onebufs(onerecvbufs_raw)
        unregister_onebufs(onesendbufs_raw_h)
        unregister_onebufs(onerecvbufs_raw_h)
        reset_one_buffers()
    end

    function free_onebufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if is_onearray(bufs[i][n]) oneAPI.unsafe_free!(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    function unregister_onebufs(bufs)
        if (bufs !== nothing)
            for i = 1:length(bufs)
                for n = 1:length(bufs[i])
                    if (isa(bufs[i][n],oneAPI.Mem.HostBuffer)) oneAPI.Mem.unregister(bufs[i][n]); bufs[i][n] = []; end
                end
            end
        end
    end

    function reset_one_buffers()
        onesendbufs_raw = nothing
        onerecvbufs_raw = nothing
        onesendbufs_raw_h = nothing
        onerecvbufs_raw_h = nothing
    end


    # (oneAPI functions)

    function init_onebufs_arrays()
        onesendbufs_raw = Array{Array{Any,1},1}();
        onerecvbufs_raw = Array{Array{Any,1},1}();
        onesendbufs_raw_h = Array{Array{Any,1},1}();
        onerecvbufs_raw_h = Array{Array{Any,1},1}();
    end

    function init_onebufs(T::DataType, fields::GGField...)
        while (length(onesendbufs_raw) < length(fields)) push!(onesendbufs_raw, [oneArray{T}(undef,0), oneArray{T}(undef,0)]); end
        while (length(onerecvbufs_raw) < length(fields)) push!(onerecvbufs_raw, [oneArray{T}(undef,0), oneArray{T}(undef,0)]); end
        while (length(onesendbufs_raw_h) < length(fields)) push!(onesendbufs_raw_h, [[], []]); end
        while (length(onerecvbufs_raw_h) < length(fields)) push!(onerecvbufs_raw_h, [[], []]); end
    end

    function reinterpret_onebufs(T::DataType, i::Integer, n::Integer)
        if (eltype(onesendbufs_raw[i][n]) != T) onesendbufs_raw[i][n] = reinterpret(T, onesendbufs_raw[i][n]); end
        if (eltype(onerecvbufs_raw[i][n]) != T) onerecvbufs_raw[i][n] = reinterpret(T, onerecvbufs_raw[i][n]); end
    end

    function reallocate_undersized_onebufs(T::DataType, i::Integer, max_halo_elems::Integer)
        if (!isnothing(onesendbufs_raw) && length(onesendbufs_raw[i][1]) < max_halo_elems)
            for n = 1:NNEIGHBORS_PER_DIM
                reallocate_onebufs(T, i, n, max_halo_elems); GC.gc(); # Too small buffers had been replaced with larger ones; free the unused memory immediately.
            end
        end
    end

    function reallocate_onebufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        onesendbufs_raw[i][n] = oneAPI.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        onerecvbufs_raw[i][n] = oneAPI.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_onebufs(T::DataType, i::Integer, n::Integer, sendbufs_raw, recvbufs_raw)
        if (isa(onesendbufs_raw_h[i][n],oneAPI.Mem.HostBuffer)) oneAPI.Mem.unregister(onesendbufs_raw_h[i][n]); onesendbufs_raw_h[i][n] = []; end # It is always initialized registered... if (cusendbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        if (isa(onerecvbufs_raw_h[i][n],oneAPI.Mem.HostBuffer)) oneAPI.Mem.unregister(onerecvbufs_raw_h[i][n]); onerecvbufs_raw_h[i][n] = []; end # It is always initialized registered... if (curecvbufs_raw_h[i][n].bytesize > 32*sizeof(T))
        onesendbufs_raw[i][n], onesendbufs_raw_h[i][n] = register(oneArray,sendbufs_raw[i][n]);
        onerecvbufs_raw[i][n], onerecvbufs_raw_h[i][n] = register(oneArray,recvbufs_raw[i][n]);
    end


    # (oneAPI functions)

    function gpusendbuf_flat(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where T <: GGNumber
        return view(onesendbufs_raw[i][n]::oneVector{T},1:prod(halosize(dim,A)));
    end

    function gpurecvbuf_flat(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where T <: GGNumber
        return view(onerecvbufs_raw[i][n]::CuVector{T},1:prod(halosize(dim,A)));
    end


    # (GPU functions)

    #TODO: see if remove T here and in other cases for CuArray, ROCArray or Array (but then it does not verify that CuArray/ROCArray is of type GGNumber) or if I should instead change GGArray to GGArrayUnion and create: GGArray = Array{T} where T <: GGNumber  and  GGCuArray = CuArray{T} where T <: GGNumber; This is however more difficult to read and understand for others.
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::oneField{T}) where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_onesendbufs_raw, get_onerecvbufs_raw
    get_onesendbufs_raw()  = deepcopy(onesendbufs_raw)
    get_onerecvbufs_raw()  = deepcopy(onerecvbufs_raw)
end


##----------------------------------------------
## FUNCTIONS TO WRITE AND READ SEND/RECV BUFFERS

function ImplicitGlobalGrid.allocate_onestreams(fields::GGField...)
    allocate_onestreams_iwrite(fields...);
    allocate_onestreams_iread(fields...);
end

ImplicitGlobalGrid.iwrite_sendbufs!(n::Integer, dim::Integer, F::oneField{T}, i::Integer) where {T <: GGNumber} = iwrite_sendbufs!(n,dim,F,i)
ImplicitGlobalGrid.iread_recvbufs!(n::Integer, dim::Integer, F::oneField{T}, i::Integer) where {T <: GGNumber} = iread_recvbufs!(n,dim,F,i)
ImplicitGlobalGrid.wait_iwrite(n::Integer, A::oneField{T}, i::Integer) where {T <: GGNumber} = wait_iwrite(n,A,i)
ImplicitGlobalGrid.wait_iread(n::Integer, A::oneField{T}, i::Integer) where {T <: GGNumber} = wait_iread(n,A,i)

let
    global iwrite_sendbufs!, allocate_onestreams_iwrite, wait_iwrite

    onestreams = Array{oneStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iwrite(n::Integer, A::oneField{T}, i::Integer) where T <: GGNumber = oneAPI.synchronize(onestreams[n,i]; blocking=true);

    function allocate_onestreams_iwrite(fields::GGField...)
        if length(fields) > size(onestreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuField
            onestreams = [onestreams [oneStream(; flags=ONEAPI.STREAM_NON_BLOCKING, priority=oneAPI.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(onestreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iwrite_sendbufs!(n::Integer, dim::Integer, F::oneField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            if dim == 1 || oneapiaware_MPI(dim) # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = sendranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=onestreams[n,i] write_d2x!(gpusendbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                write_d2h_async!(sendbuf_flat(n,dim,i,F), A, sendranges(n,dim,F), onestreams[n,i]);
            end
        end
    end
end

let
    global iread_recvbufs!, allocate_onestreams_iread, wait_iread

    onestreams = Array{oneStream}(undef, NNEIGHBORS_PER_DIM, 0)

    wait_iread(n::Integer, A::oneField{T}, i::Integer) where T <: GGNumber = oneAPI.synchronize(onestreams[n,i]; blocking=true);

    function allocate_onestreams_iread(fields::GGField...)
        if length(fields) > size(onestreams,2)  # Note: for simplicity, we create a stream for every field even if it is not a CuField
            onestreams = [onestreams [oneStream(; flags=ONEAPI.STREAM_NON_BLOCKING, priority=oneAPI.priority_range()[end]) for n=1:NNEIGHBORS_PER_DIM, i=1:(length(fields)-size(onestreams,2))]];  # Create (additional) maximum priority nonblocking streams to enable overlap with computation kernels.
        end
    end

    function iread_recvbufs!(n::Integer, dim::Integer, F::oneField{T}, i::Integer) where T <: GGNumber
        A, halowidths = F;
        if ol(dim,A) >= 2*halowidths[dim] # There is only a halo and thus a halo update if the overlap is at least 2 times the halowidth...
            if dim == 1 || oneapiaware_MPI(dim)  # Use a custom copy kernel for the first dimension to obtain a good copy performance (the CUDA 3-D memcopy does not perform well for this extremely strided case).
                ranges = recvranges(n, dim, F);
                nthreads = (dim==1) ? (1, 32, 1) : (32, 1, 1);
                halosize = [r[end] - r[1] + 1 for r in ranges];
                nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                @cuda blocks=nblocks threads=nthreads stream=onestreams[n,i] read_x2d!(gpurecvbuf(n,dim,i,F), A, ranges[1], ranges[2], ranges[3], dim);
            else
                read_h2d_async!(recvbuf_flat(n,dim,i,F), A, recvranges(n,dim,F), onestreams[n,i]);
            end
        end
    end
end


# (CUDA functions)

# Write to the send buffer on the host or device from the array on the device (d2x).
function ImplicitGlobalGrid.write_d2x!(gpusendbuf::oneArray{T}, A::oneArray{T}, sendrangex::UnitRange{Int64}, sendrangey::UnitRange{Int64}, sendrangez::UnitRange{Int64},  dim::Integer) where T <: GGNumber
    ix = (oneAPI.blockIdx().x-1) * oneAPI.blockDim().x + oneAPI.threadIdx().x + sendrangex[1] - 1
    iy = (oneAPI.blockIdx().y-1) * oneAPI.blockDim().y + oneAPI.threadIdx().y + sendrangey[1] - 1
    iz = (oneAPI.blockIdx().z-1) * oneAPI.blockDim().z + oneAPI.threadIdx().z + sendrangez[1] - 1
    if !(ix in sendrangex && iy in sendrangey && iz in sendrangez) return nothing; end
    gpusendbuf[ix-(sendrangex[1]-1),iy-(sendrangey[1]-1),iz-(sendrangez[1]-1)] = A[ix,iy,iz];
    return nothing
end

# Read from the receive buffer on the host or device and store on the array on the device (x2d).
function ImplicitGlobalGrid.read_x2d!(gpurecvbuf::oneArray{T}, A::oneArray{T}, recvrangex::UnitRange{Int64}, recvrangey::UnitRange{Int64}, recvrangez::UnitRange{Int64}, dim::Integer) where T <: GGNumber
    ix = (oneAPI.blockIdx().x-1) * oneAPI.blockDim().x + oneAPI.threadIdx().x + recvrangex[1] - 1
    iy = (oneAPI.blockIdx().y-1) * oneAPI.blockDim().y + oneAPI.threadIdx().y + recvrangey[1] - 1
    iz = (oneAPI.blockIdx().z-1) * oneAPI.blockDim().z + oneAPI.threadIdx().z + recvrangez[1] - 1
    if !(ix in recvrangex && iy in recvrangey && iz in recvrangez) return nothing; end
    A[ix,iy,iz] = gpurecvbuf[ix-(recvrangex[1]-1),iy-(recvrangey[1]-1),iz-(recvrangez[1]-1)];
    return nothing
end

# Write to the send buffer on the host from the array on the device (d2h).
function ImplicitGlobalGrid.write_d2h_async!(sendbuf::AbstractArray{T}, A::oneArray{T}, sendranges::Array{UnitRange{T2},1}, onestream::oneStream) where T <: GGNumber where T2 <: Integer
    oneAPI.Mem.unsafe_copy3d!(
        pointer(sendbuf), oneAPI.Mem.Host, pointer(A), oneAPI.Mem.Device,
        length(sendranges[1]), length(sendranges[2]), length(sendranges[3]);
        srcPos=(sendranges[1][1], sendranges[2][1], sendranges[3][1]),
        srcPitch=sizeof(T)*size(A,1), srcHeight=size(A,2),
        dstPitch=sizeof(T)*length(sendranges[1]), dstHeight=length(sendranges[2]),
        async=true, stream=onestream
    )
end

# Read from the receive buffer on the host and store on the array on the device (h2d).
function ImplicitGlobalGrid.read_h2d_async!(recvbuf::AbstractArray{T}, A::CuArray{T}, recvranges::Array{UnitRange{T2},1}, onestream::oneStream) where T <: GGNumber where T2 <: Integer
    oneAPI.Mem.unsafe_copy3d!(
        pointer(A), oneAPI.Mem.Device, pointer(recvbuf), oneAPI.Mem.Host,
        length(recvranges[1]), length(recvranges[2]), length(recvranges[3]);
        dstPos=(recvranges[1][1], recvranges[2][1], recvranges[3][1]),
        srcPitch=sizeof(T)*length(recvranges[1]), srcHeight=length(recvranges[2]),
        dstPitch=sizeof(T)*size(A,1), dstHeight=size(A,2),
        async=true, stream=onestream
    )
end


##------------------------------
## FUNCTIONS TO SEND/RECV FIELDS

function ImplicitGlobalGrid.gpumemcopy!(dst::oneArray{T}, src::oneArray{T}) where T <: GGNumber
    @inbounds oneAPI.copyto!(dst, src)
end

