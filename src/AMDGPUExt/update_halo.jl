##---------------------------------------
## FUNCTIONS RELATED TO BUFFER ALLOCATION

# NOTE: CUDA and AMDGPU buffers live and are dealt with independently, enabling the support of usage of CUDA and AMD GPUs at the same time.

let
    global free_update_halo_rocbuffers, reset_roc_buffers, free_rocbufs
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

    function reallocate_rocbufs(T::DataType, i::Integer, n::Integer, max_halo_elems::Integer)
        rocsendbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY); # Ensure that the amount of allocated memory is a multiple of 4*sizeof(T) (sizeof(Float64)/sizeof(Float16) = 4). So, we can always correctly reinterpret the raw buffers even if next time sizeof(T) is greater.
        rocrecvbufs_raw[i][n] = AMDGPU.zeros(T, Int(ceil(max_halo_elems/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);
    end

    function reregister_rocbufs(T::DataType, i::Integer, n::Integer)
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
    function gpusendbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T} where T <: GGNumber
        return reshape(gpusendbuf_flat(n,dim,i,A), halosize(dim,A));
    end

    function gpurecvbuf(n::Integer, dim::Integer, i::Integer, A::ROCField{T} where T <: GGNumber
        return reshape(gpurecvbuf_flat(n,dim,i,A), halosize(dim,A));
    end


    # Make sendbufs_raw and recvbufs_raw accessible for unit testing.
    global get_rocsendbufs_raw, get_rocrecvbufs_raw
    get_rocsendbufs_raw() = deepcopy(rocsendbufs_raw)
    get_rocrecvbufs_raw() = deepcopy(rocrecvbufs_raw)
end