export gather!

@doc """
    gather!(A, A_global)
    gather!(A, A_global; root=0)

Gather a CPU-array `A` from each member of the Cartesian grid of MPI processes into a one large CPU-array `A_global` on the root process (default: `0`).

!!! note "Memory usage note"
    `gather!` allocates at first call an internal buffer of the size of `A_global` and keeps it alive until [`finalize_global_grid`](@ref) is called. A (re-)allocation occurs only if `gather!` is called with a larger `A_global` than in any previous call since the call to [`init_global_grid`](@ref). This is an optimisation to minimize (re-)allocation, which is very important as `gather!` is typically called in the main loop of a simulation and its performance is critical for the overall application performance.
"""
gather!

let
    global gather!, free_gather_buffer
    A_all_buf = zeros(0);

    "Free the buffer used by gather!."
    function free_gather_buffer()
        A_all_buf = nothing;
        GC.gc();
        A_all_buf = zeros(0);
    end

    function gather!(A::Array{T}, A_global::Union{Array{T}, Nothing}; root::Integer=0) where T <: GGNumber
        check_initialized();
        cart_gather!(A, A_global, me(), global_grid().dims, comm(); root=root);
        return nothing
    end

    function cart_gather!(A::Array{T}, A_global::Union{Array{T}, Nothing}, me::Integer, dims::Array{T2}, comm_cart::MPI.Comm; root::Integer=0, tag::Integer=0, ndims::Integer=NDIMS_MPI) where T <: GGNumber where T2 <: Integer
        nprocs = prod(dims);
        if me != root
            req = MPI.REQUEST_NULL;
            req = MPI.Isend(A, root, tag, comm_cart);
            MPI.Wait!(req);
        else # (me == root)
            A_global === nothing && error("The input argument A_global can't be `nothing` on the root")
            if length(A_global) != nprocs*length(A) error("The input argument A_global must be of length nprocs*length(A)") end
            if (eltype(A_all_buf) != T)
                A_all_buf = reinterpret(T, A_all_buf);
            end
            if length(A_all_buf) < nprocs*length(A)                       # Allocate only if the buffer is not large enough
                free_gather_buffer();                                     # Free the memory of the old buffer immediately as it can typically go up to the order of the total available memory.
                A_all_buf = zeros(T, Int(ceil(nprocs*length(A)/GG_ALLOC_GRANULARITY))*GG_ALLOC_GRANULARITY);  # Ensure that the amount of allocated memory is a multiple of GG_ALLOC_GRANULARITY*sizeof(T). So, we can always correctly reinterpret A_all_buf even if next time sizeof(T) is greater.
            end
            A_all_flat = view(A_all_buf,1:nprocs*length(A));              # Create a 1D-view on the amount of memory needed from A_all_buf.
            reqs = fill(MPI.REQUEST_NULL, nprocs);
            for p in [0:root-1; root+1:nprocs-1]
                cs = Cint[-1,-1,-1];
                MPI.Cart_coords!(comm_cart, p, cs);
                offset = cs[1]*length(A) + cs[2]*dims[1]*length(A) + cs[3]*dims[1]*dims[2]*length(A)
                A_c = view(A_all_flat, 1+offset:length(A)+offset);
                reqs[p+1] = MPI.Irecv!(A_c, p, tag, comm_cart);           # Irev! requires a contigous (SubArray) buffer (that is not both reshaped and reinterpreted)...
            end
            cs = MPI.Cart_coords(comm_cart);
            A_all = reshape(A_all_flat, (length(A), dims[1], dims[2], dims[3])); # Create a 4D-view on the amount of memory needed from A_all_buf.
            A_all[:,cs[1]+1,cs[2]+1,cs[3]+1] .= A[:];
            if (nprocs>1) MPI.Waitall!(reqs); end
            nx, ny, nz = size(view(A,:,:,:));
            for cz = 0:size(A_all,4)-1, cy = 0:size(A_all,3)-1, cx = 0:size(A_all,2)-1
                A_global[(1:nx).+cx*nx, (1:ny).+cy*ny, (1:nz).+cz*nz] .= reshape(A_all[:,cx+1,cy+1,cz+1],nx,ny,nz); # Store the data at the right place in A_global (works for 1D-3D, e.g. if 2D: nz=1, cz=0...)
            end
        end
    end
end
