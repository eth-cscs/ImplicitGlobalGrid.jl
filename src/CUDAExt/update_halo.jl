##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

let
    global free_update_halo_cubuffers, reset_cu_buffers, free_cubufs, unregister_cubufs
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

    function reallocate_cubufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        cusendbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        curecvbufs_raw[i][n] = CUDA.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_cubufs(T::DataType, i::Integer, n::Integer)
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
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T} where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::CuField{T} where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_cusendbufs_raw, get_curecvbufs_raw
    get_cusendbufs_raw()  = deepcopy(cusendbufs_raw)
    get_curecvbufs_raw()  = deepcopy(curecvbufs_raw)
end