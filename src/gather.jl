export gather!

"""
    gather!(A, A_global)
    gather!(A, A_global; root=0)

!!! note "Advanced"
        gather!(A, A_global, comm; root=0)

Gather an array `A` from each member of the Cartesian grid of MPI processes into one large array `A_global` on the root process (default: `0`). The size of the global array `size(A_global)` must be equal to `size(A) .* dims`, where `dims` is the number of processes in each dimension of the Cartesian grid, defined in [`init_global_grid`](@ref).

!!! note "Advanced"
    If the argument `comm` is given, then this communicator is used for the gather operation and `dims` extracted from it. 

!!! note "Memory requirements"
    The memory for the global array only needs to be allocated on the root process; the argument `A_global` can be `nothing` on the other processes.
"""
function gather!(A::AbstractArray{T,N}, A_global::Union{AbstractArray{T,N},Nothing}; root::Integer=0) where {T,N}
    check_initialized();
    gather!(A, A_global, comm(); root=root);
    return nothing
end:
    

function gather!(A::AbstractArray{T,N}, A_global::Union{AbstractArray{T,N},Nothing}, comm::MPI.Comm; root::Integer=0) where {T,N}
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        if (A_global === nothing) error("The input argument A_global can't be `nothing` on the root.") end
        if (size(A_global) != (dims .* size(A))) error("The size of the global array `size(A_global)` must be equal to `size(A) .* dims`.") end
        # Make subtype for gather
        offset  = Tuple(0 for _ in 1:N)
        subtype = MPI.Types.create_subarray(size(A_global), size(A), offset, MPI.Datatype(eltype(A_global)))
        subtype = MPI.Types.create_resized(subtype, 0, size(A, 1) * Base.elsize(A_global))
        MPI.Types.commit!(subtype)
        # Make VBuffer for collective communication
        counts  = fill(Cint(1), reverse(dims)) # Gather one subarray from each MPI rank
        displs  = zeros(Cint, reverse(dims))   # Reverse dims since MPI Cart comm is row-major
        csizes  = cumprod(size(A)[2:end] .* dims[1:end-1])
        strides = (1, csizes...)
        for I in CartesianIndices(displs)
            offset = reverse(Tuple(I - oneunit(I)))
            displs[I] = sum(offset .* strides)
        end
        recvbuf = MPI.VBuffer(A_global, vec(counts), vec(displs), subtype)
        MPI.Gatherv!(A, recvbuf, comm; root)
    else
        MPI.Gatherv!(A, nothing, comm; root)
    end
    return
end
