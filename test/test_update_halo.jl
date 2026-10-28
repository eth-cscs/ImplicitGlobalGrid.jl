# NOTE: All tests of this file can be run with any number of processes.
# Nearly all of the functionality can however be verified with one single process
# (thanks to the usage of periodic boundaries in most of the full halo update tests).

using Test
import MPI, Polyester
using CUDA, AMDGPU
using CellArrays, StaticArrays
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require, longnameof

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

SArray{S, T, N, L}(a::Number) where {S, T, N, L}          = SArray{S, T, N, L}(ntuple(_ -> a, Val(L)))
celldims                                                  = (2, 2)
Cell{T}                                                   = SMatrix{celldims..., T, prod(celldims)}
cellzeros0(::Type{T}, dims::Integer...) where {T<:Number} = (A=CPUCellArray{Cell{T},0}(undef, dims); A.data.=0.0; A)
cellzeros1(::Type{T}, dims::Integer...) where {T<:Number} = (A=CPUCellArray{Cell{T},1}(undef, dims); A.data.=0.0; A)
cellzeros0(dims::Integer...)                              = cellzeros0(Float64, dims...)
cellzeros1(dims::Integer...)                              = cellzeros1(Float64, dims...)

array_types              = ["CPU"]
cellarray_types          = ["CPU cell (B=0)", "CPU cell (B=1)"]
gpu_array_types          = []
gpu_cellarray_types      = []
device_types             = ["auto"]
celldevice_types         = ["auto", "auto"]
gpu_device_types         = []
allocators               = Function[zeros]
cellallocators           = Function[cellzeros0, cellzeros1]
cpu_allocators           = Function[zeros]
cpu_cellallocators       = Function[cellzeros0, cellzeros1]
gpu_allocators           = []
gpu_cellallocators       = []
ArrayConstructors        = [Array]
CellArrayConstructors    = [CPUCellArray, CPUCellArray]
CPUArrayConstructors     = [Array]
CPUCellArrayConstructors = [CPUCellArray, CPUCellArray]
GPUArrayConstructors     = []
GPUCellArrayConstructors = []
CPUArray                 = Array
if test_cuda
    @define_CuCellArray
    cuzeros = CUDA.zeros
    cucellzeros0(::Type{T}, dims::Integer...) where {T<:Number} = (A=CuCellArray{Cell{T},0}(undef, dims); A.data.=0.0; A)
    cucellzeros1(::Type{T}, dims::Integer...) where {T<:Number} = (A=CuCellArray{Cell{T},1}(undef, dims); A.data.=0.0; A)
    cucellzeros0(dims::Integer...)                              = cucellzeros0(Float64, dims...)
    cucellzeros1(dims::Integer...)                              = cucellzeros1(Float64, dims...)
    push!(array_types, "CUDA")
    push!(cellarray_types, "CUDA cell (B=0)", "CUDA cell (B=1)")
    push!(gpu_array_types, "CUDA")
    # push!(gpu_cellarray_types, "CUDA cell (B=0)", "CUDA cell (B=1)")
    push!(device_types, "CUDA")
    push!(celldevice_types, "CUDA", "CUDA")
    push!(gpu_device_types, "CUDA")
    push!(allocators, cuzeros)
    push!(cpu_allocators, zeros)
    push!(cpu_cellallocators, cellzeros0, cellzeros1)
    push!(cellallocators, cucellzeros0, cucellzeros1)
    push!(gpu_allocators, cuzeros)
    # push!(gpu_cellallocators, cucellzeros0, cucellzeros1)
    push!(ArrayConstructors, CuArray)
    push!(CellArrayConstructors, CuCellArray, CuCellArray)
    push!(CPUArrayConstructors, Array)
    push!(CPUCellArrayConstructors, CPUCellArray, CPUCellArray)
    push!(GPUArrayConstructors, CuArray)
    # push!(GPUCellArrayConstructors, CuCellArray, CuCellArray)
end
if test_amdgpu
    @define_ROCCellArray
    roczeros = AMDGPU.zeros
    roccellzeros0(::Type{T}, dims::Integer...) where {T<:Number} = (A=ROCCellArray{Cell{T},0}(undef, dims); A.data.=0.0; A)
    roccellzeros1(::Type{T}, dims::Integer...) where {T<:Number} = (A=ROCCellArray{Cell{T},1}(undef, dims); A.data.=0.0; A)
    roccellzeros0(dims::Integer...)                              = roccellzeros0(Float64, dims...)
    roccellzeros1(dims::Integer...)                              = roccellzeros1(Float64, dims...)
    push!(array_types, "AMDGPU")
    push!(cellarray_types, "AMDGPU cell (B=0)", "AMDGPU cell (B=1)")
    push!(gpu_array_types, "AMDGPU")
    # push!(gpu_cellarray_types, "AMDGPU cell (B=0)", "AMDGPU cell (B=1)")
    push!(device_types, "AMDGPU")
    push!(celldevice_types, "AMDGPU", "AMDGPU")
    push!(gpu_device_types, "AMDGPU")
    push!(allocators, roczeros)
    push!(cpu_allocators, zeros)
    push!(cpu_cellallocators, cellzeros0, cellzeros1)
    push!(cellallocators, roccellzeros0, roccellzeros1)
    push!(gpu_allocators, roczeros)
    # push!(gpu_cellallocators, roccellzeros0, roccellzeros1)
    push!(ArrayConstructors, ROCArray)
    push!(CellArrayConstructors, ROCCellArray, ROCCellArray)
    push!(CPUArrayConstructors, Array)
    push!(CPUCellArrayConstructors, CPUCellArray, CPUCellArray)
    push!(GPUArrayConstructors, ROCArray)
    # push!(GPUCellArrayConstructors, ROCCellArray, ROCCellArray)
end
allocators_union           = [allocators... cellallocators...]
cpu_allocators_union       = [cpu_allocators... cpu_cellallocators...]
# gpu_allocators_union       = [gpu_allocators... gpu_cellallocators...]
ArrayConstructors_union    = [ArrayConstructors... CellArrayConstructors...]
CPUArrayConstructors_union = [CPUArrayConstructors... CPUCellArrayConstructors...]
# GPUArrayConstructors_union = [GPUArrayConstructors... GPUCellArrayConstructors...]
array_types_union          = [array_types... cellarray_types...]
# gpu_array_types_union      = [gpu_array_types... gpu_cellarray_types...]
device_types_union         = [device_types... celldevice_types...]
# gpu_device_types_union     = [gpu_device_types... gpu_device_types... gpu_device_types...]

## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.
ndims_mpi = GG.NDIMS_MPI;
nneighbors_per_dim = GG.NNEIGHBORS_PER_DIM; # Should be 2 (one left and one right neighbor).
nx = 7;
ny = 5;
nz = 6;
dx = 1.0
dy = 1.0
dz = 1.0

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. argument check (arrays: $array_type)" for (array_type, device_type, zeros) in zip(array_types_union, device_types_union, allocators_union)
        init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
        P   = zeros(nx,  ny,  nz  );
        Sxz = zeros(nx-2,ny-1,nz-2);
        A   = zeros(nx-1,ny+2,nz+1);
        A2  = A;
        Z   = zeros(ComplexF64, nx-1,ny+2,nz+1);
        Z2  = Z;
        @test_throws ErrorException update_halo!(P, Sxz, A)                     # Error: Sxz has no halo.
        @test_throws ErrorException update_halo!(P, Sxz, A, Sxz)                # Error: Sxz and Sxz have no halo.
        @test_throws ErrorException update_halo!(A, (A=P, halowidths=(1,0,1)))  # Error: P has an invalid halowidth (less than 1).
        @test_throws ErrorException update_halo!(A, (A=P, halowidths=(2,2,2)))  # Error: P has no halo.
        @test_throws ErrorException update_halo!((A=A, halowidths=(0,3,2)), (A=P, halowidths=(2,2,2)))  # Error: A and P have no halo.
        if !occursin("cell", array_type)                                        # CellArrays do not support the following sanity checks:
            @test_throws ErrorException update_halo!(P, A, A)                   # Error: A is given twice.
            @test_throws ErrorException update_halo!(P, A, A2)                  # Error: A2 is duplicate of A (an alias; it points to the same memory).
            @test_throws ErrorException update_halo!(P, A, A, A2)               # Error: the second A and A2 are duplicates of the first A.
            @test_throws ErrorException update_halo!(Z, Z2)                     # Error: Z2 is duplicate of Z (an alias; it points to the same memory).
        end
        @test_throws ErrorException update_halo!(Z, P)                          # Error: P is of different type than Z.
        @test_throws ErrorException update_halo!(Z, P, A)                       # Error: P and A are of different type than Z.
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "2. buffer allocation (arrays: $array_type)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
        init_global_grid(nx, ny, nz, periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
        P = zeros(nx,  ny,  nz  );
        A = zeros(nx-1,ny+2,nz+1);
        B = zeros(Float32,    nx+1, ny+2, nz+3);
        C = zeros(Float32,    nx+1, ny+1, nz+1);
        Z = zeros(ComplexF16, nx,   ny,   nz  );
        Y = zeros(ComplexF16, nx-1, ny+2, nz+1);
        D = view(P, 2:length(P)-1);
        E = @view A[2:end-1, 2:end-1, 2:end-1];
        P, A, B, C, Z, Y, D, E = GG.wrap_field.((P, A, B, C, Z, Y, D, E));
        halowidths = (3,1,2);
        A_hw, Z_hw = GG.wrap_field(A.A, halowidths), GG.wrap_field(Z.A, halowidths);
        @testset "free buffers" begin
            @require GG.get_sendbufs_raw() === nothing
            @require GG.get_recvbufs_raw() === nothing
            GG.allocate_bufs(P);
            @require GG.get_sendbufs_raw() !== nothing
            @require GG.get_recvbufs_raw() !== nothing
            GG.free_update_halo_buffers();
            @test GG.get_sendbufs_raw() === nothing
            @test GG.get_recvbufs_raw() === nothing
        end;
        @testset "allocate single" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(P);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 1                                    # 1 array
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(P)...])[2:end])  # required length: max halo elements in any of the dimensions
                end
            end
        end;
        @testset "allocate single (Complex)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(Z);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 1                                    # 1 array
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(Z)...])[2:end])  # required length: max halo elements in any of the dimensions
                end
            end
        end;
        @testset "allocate single (contiguous view)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(D);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 1                                    # 1 array
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(D)...])[2:end])  # required length: max halo elements in any of the dimensions
                end
            end
        end;
        @testset "allocate single (non-contiguous view)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(E);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 1                                    # 1 array
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(E)...])[2:end])  # required length: max halo elements in any of the dimensions
                end
            end
        end;
        @testset "allocate single (halowidth > 1)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(A_hw);
            max_halo_elems = maximum((size(A,1)*size(A,2)*halowidths[3], size(A,1)*size(A,3)*halowidths[2], size(A,2)*size(A,3)*halowidths[1]));
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 1                                    # 1 array
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= max_halo_elems                   # required length: max halo elements in any of the dimensions
                end
            end
        end;
        @testset "keep 1st, allocate 2nd" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(P);
            GG.allocate_bufs(A, P);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 2                                    # 2 arrays
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                @test length(bufs_raw[2])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(A)...])[2:end])  # required length: max halo elements in any of the dimensions
                    @test length(bufs_raw[2][n]) >= prod(sort([size(P)...])[2:end])  # ...
                end
            end
        end;
        @testset "keep 1st, allocate 2nd (Complex)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(Z);
            GG.allocate_bufs(Y, Z);
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 2                                    # 2 arrays
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                @test length(bufs_raw[2])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(Y)...])[2:end])  # required length: max halo elements in any of the dimensions
                    @test length(bufs_raw[2][n]) >= prod(sort([size(Z)...])[2:end])  # ...
                end
            end
        end;
        @testset "reinterpret (no allocation)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(A, P);
            GG.allocate_bufs(B, C);                                                  # The new arrays contain Float32 (A, and P were Float64); B and C have a halo with more elements than A and P had, but they require less space in memory
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 2                                    # Still 2 arrays: B, C (even though they are different then before: was A and P)
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                @test length(bufs_raw[2])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(B)...])[2:end])  # required length: max halo elements in any of the dimensions
                    @test length(bufs_raw[2][n]) >= prod(sort([size(C)...])[2:end])  # ...
                end
                @test all([eltype(bufs_raw[i][n]) == Float32 for i=1:length(bufs_raw), n=1:nneighbors_per_dim])
            end
        end;
        @testset "reinterpret (no allocation) (Complex)" begin
            GG.free_update_halo_buffers();
            GG.allocate_bufs(A, P);
            GG.allocate_bufs(Y, Z);                                                  # The new arrays contain Float32 (A, and P were Float64); B and C have a halo with more elements than A and P had, but they require less space in memory
            for bufs_raw in [GG.get_sendbufs_raw(), GG.get_recvbufs_raw()]
                @test length(bufs_raw)       == 2                                    # Still 2 arrays: B, C (even though they are different then before: was A and P)
                @test length(bufs_raw[1])    == nneighbors_per_dim                   # 2 neighbors per dimension
                @test length(bufs_raw[2])    == nneighbors_per_dim                   # 2 neighbors per dimension
                for n = 1:nneighbors_per_dim
                    @test length(bufs_raw[1][n]) >= prod(sort([size(Y)...])[2:end])  # required length: max halo elements in any of the dimensions
                    @test length(bufs_raw[2][n]) >= prod(sort([size(Z)...])[2:end])  # ...
                end
                @test all([eltype(bufs_raw[i][n]) == ComplexF16 for i=1:length(bufs_raw), n=1:nneighbors_per_dim])
            end
        end;
        @testset "(cu/roc)sendbuf / (cu/roc)recvbuf" begin
            sendbuf, recvbuf = (GG.sendbuf, GG.recvbuf);
            if array_type in ["CUDA", "AMDGPU"]
                sendbuf, recvbuf = (GG.gpusendbuf, GG.gpurecvbuf);
            end
            GG.free_update_halo_buffers();
            GG.allocate_bufs(A, P);
            for dim = 1:ndims(A), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,1,A))    .== prod(size(A)[1:ndims(A).!=dim]))
                @test all(length(recvbuf(n,dim,1,A))    .== prod(size(A)[1:ndims(A).!=dim]))
                @test all(size(sendbuf(n,dim,1,A))[dim] .== A.halowidths[dim])
                @test all(size(recvbuf(n,dim,1,A))[dim] .== A.halowidths[dim])
            end
            for dim = 1:ndims(P), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,2,P))    .== prod(size(P)[1:ndims(P).!=dim]))
                @test all(length(recvbuf(n,dim,2,P))    .== prod(size(P)[1:ndims(P).!=dim]))
                @test all(size(sendbuf(n,dim,2,P))[dim] .== P.halowidths[dim])
                @test all(size(recvbuf(n,dim,2,P))[dim] .== P.halowidths[dim])
            end
        end;
        @testset "(cu/roc)sendbuf / (cu/roc)recvbuf (Complex)" begin
            sendbuf, recvbuf = (GG.sendbuf, GG.recvbuf);
            if array_type in ["CUDA", "AMDGPU"]
                sendbuf, recvbuf = (GG.gpusendbuf, GG.gpurecvbuf);
            end
            GG.free_update_halo_buffers();
            GG.allocate_bufs(Y, Z);
            for dim = 1:ndims(Y), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,1,Y))    .== prod(size(Y)[1:ndims(Y).!=dim]))
                @test all(length(recvbuf(n,dim,1,Y))    .== prod(size(Y)[1:ndims(Y).!=dim]))
                @test all(size(sendbuf(n,dim,1,Y))[dim] .== Y.halowidths[dim])
                @test all(size(recvbuf(n,dim,1,Y))[dim] .== Y.halowidths[dim])
            end
            for dim = 1:ndims(Z), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,2,Z))    .== prod(size(Z)[1:ndims(Z).!=dim]))
                @test all(length(recvbuf(n,dim,2,Z))    .== prod(size(Z)[1:ndims(Z).!=dim]))
                @test all(size(sendbuf(n,dim,2,Z))[dim] .== Z.halowidths[dim])
                @test all(size(recvbuf(n,dim,2,Z))[dim] .== Z.halowidths[dim])
            end
        end;
        @testset "(cu/roc)sendbuf / (cu/roc)recvbuf (views)" begin
            sendbuf, recvbuf = (GG.sendbuf, GG.recvbuf);
            if array_type in ["CUDA", "AMDGPU"]
                sendbuf, recvbuf = (GG.gpusendbuf, GG.gpurecvbuf);
            end
            GG.free_update_halo_buffers();
            GG.allocate_bufs(D, P);
            for dim = 1:ndims(D), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,1,D))    .== prod(size(D)[1:ndims(D).!=dim]))
                @test all(length(recvbuf(n,dim,1,D))    .== prod(size(D)[1:ndims(D).!=dim]))
                @test all(size(sendbuf(n,dim,1,D))[dim] .== D.halowidths[dim])
                @test all(size(recvbuf(n,dim,1,D))[dim] .== D.halowidths[dim])
            end
            for dim = 1:ndims(E), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,2,E))    .== prod(size(E)[1:ndims(E).!=dim]))
                @test all(length(recvbuf(n,dim,2,E))    .== prod(size(E)[1:ndims(E).!=dim]))
                @test all(size(sendbuf(n,dim,2,E))[dim] .== E.halowidths[dim])
                @test all(size(recvbuf(n,dim,2,E))[dim] .== E.halowidths[dim])
            end
        end;
        @testset "(cu/roc)sendbuf / (cu/roc)recvbuf (halowidth > 1)" begin
            sendbuf, recvbuf = (GG.sendbuf, GG.recvbuf);
            if array_type in ["CUDA", "AMDGPU"]
                sendbuf, recvbuf = (GG.gpusendbuf, GG.gpurecvbuf);
            end
            GG.free_update_halo_buffers();
            GG.allocate_bufs(A_hw);
            for dim = 1:ndims(A_hw), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,1,A_hw))    .== prod(size(A_hw)[1:ndims(A_hw).!=dim])*A_hw.halowidths[dim])
                @test all(length(recvbuf(n,dim,1,A_hw))    .== prod(size(A_hw)[1:ndims(A_hw).!=dim])*A_hw.halowidths[dim])
                @test all(size(sendbuf(n,dim,1,A_hw))[dim] .== A_hw.halowidths[dim])
                @test all(size(recvbuf(n,dim,1,A_hw))[dim] .== A_hw.halowidths[dim])
            end
        end;
        @testset "(cu/roc)sendbuf / (cu/roc)recvbuf (halowidth > 1, Complex)" begin
            sendbuf, recvbuf = (GG.sendbuf, GG.recvbuf);
            if array_type in ["CUDA", "AMDGPU"]
                sendbuf, recvbuf = (GG.gpusendbuf, GG.gpurecvbuf);
            end
            GG.free_update_halo_buffers();
            GG.allocate_bufs(Z_hw);
            for dim = 1:ndims(Z_hw), n = 1:nneighbors_per_dim
                @test all(length(sendbuf(n,dim,1,Z_hw))    .== prod(size(Z_hw)[1:ndims(Z_hw).!=dim])*Z_hw.halowidths[dim])
                @test all(length(recvbuf(n,dim,1,Z_hw))    .== prod(size(Z_hw)[1:ndims(Z_hw).!=dim])*Z_hw.halowidths[dim])
                @test all(size(sendbuf(n,dim,1,Z_hw))[dim] .== Z_hw.halowidths[dim])
                @test all(size(recvbuf(n,dim,1,Z_hw))[dim] .== Z_hw.halowidths[dim])
            end
        end;
        @testset "GPU buffers after CPU arrays used the same buffer slot (issue #140)" begin
            if array_type in ["CUDA", "AMDGPU"]
                gpuaware = (array_type=="CUDA") ? any(GG.cudaaware_MPI()) : any(GG.amdgpuaware_MPI());
                P_cpu = GG.wrap_field(Base.zeros(nx,   ny,   nz  ));                                       # Same size as P (CPU array).
                Q_cpu = GG.wrap_field(Base.zeros(nx+3, ny+3, nz+3));                                       # Larger halo than P (CPU array): forces a reallocation of the host buffers.
                Q     = GG.wrap_field(zeros(nx+3, ny+3, nz+3));
                # The CPU array sizes the host buffers of the slot before the GPU array uses it:
                GG.free_update_halo_buffers();
                GG.allocate_bufs(P_cpu);
                GG.allocate_bufs(P);
                for dim = 1:ndims(P), n = 1:nneighbors_per_dim
                    @test length(GG.gpusendbuf(n,dim,1,P)) == prod(size(P)[1:ndims(P).!=dim])
                    @test length(GG.gpurecvbuf(n,dim,1,P)) == prod(size(P)[1:ndims(P).!=dim])
                end
                # The GPU array registers the host buffers of the slot, then a larger CPU array replaces them: the GPU buffers must wrap the current host buffers for the following (smaller and larger) GPU arrays:
                GG.free_update_halo_buffers();
                GG.allocate_bufs(P);
                GG.allocate_bufs(Q_cpu);
                GG.allocate_bufs(P);
                for dim = 1:ndims(P), n = 1:nneighbors_per_dim
                    @test length(GG.gpusendbuf(n,dim,1,P)) == prod(size(P)[1:ndims(P).!=dim])
                    @test length(GG.gpurecvbuf(n,dim,1,P)) == prod(size(P)[1:ndims(P).!=dim])
                    if !gpuaware                                                                           # Without GPU-aware MPI, the GPU buffers are the host buffers (registered for the GPU): what is written to the host buffers must be visible through the GPU buffers.
                        GG.sendbuf_flat(n,dim,1,P) .= eltype(P)( n*dim);
                        GG.recvbuf_flat(n,dim,1,P) .= eltype(P)(-n*dim);
                        @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P)) .== eltype(P)( n*dim))
                        @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P)) .== eltype(P)(-n*dim))
                    end
                end
                GG.allocate_bufs(Q);
                for dim = 1:ndims(Q), n = 1:nneighbors_per_dim
                    @test length(GG.gpusendbuf(n,dim,1,Q)) == prod(size(Q)[1:ndims(Q).!=dim])
                    @test length(GG.gpurecvbuf(n,dim,1,Q)) == prod(size(Q)[1:ndims(Q).!=dim])
                end
                # The host buffers are reinterpreted for another element type before the GPU array requires them (B contains Float32; P_cpu Float64):
                GG.free_update_halo_buffers();
                GG.allocate_bufs(P_cpu);
                GG.allocate_bufs(B);
                for dim = 1:ndims(B), n = 1:nneighbors_per_dim
                    @test length(GG.gpusendbuf(n,dim,1,B)) == prod(size(B)[1:ndims(B).!=dim])
                    @test length(GG.gpurecvbuf(n,dim,1,B)) == prod(size(B)[1:ndims(B).!=dim])
                    @test eltype(GG.gpusendbuf(n,dim,1,B)) == Float32
                    @test eltype(GG.gpurecvbuf(n,dim,1,B)) == Float32
                end
            end
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "3. data transfer components" begin
        @testset "memcopy! (use_polyester: $use_polyester)" for use_polyester in (false, true) # NOTE: the Polyester code path is only taken if Julia is started with multiple threads.
            src   = rand(nx, ny, nz);
            dst   = zeros(nx, ny, nz);
            GG.memcopy!(dst, src, use_polyester);
            @test dst == src
            dst  .= 0.0;
            GG.memcopy!(view(dst, :, 2, :), view(src, :, 2, :), use_polyester);  # Copy of a y-face (Cartesian indexing).
            @test dst[:, 2, :] == src[:, 2, :]
            @test all(dst[:, [1; 3:end], :] .== 0.0)
            src_c = rand(ComplexF64, nx, ny, nz);
            dst_c = zeros(ComplexF64, nx, ny, nz);
            GG.memcopy!(dst_c, src_c, use_polyester);                              # NOTE: Complex numbers are never copied with Polyester.
            @test dst_c == src_c
        end;
        @testset "iwrite_sendbufs! / iread_recvbufs!" begin
            @testset "sendranges / recvranges (arrays: $array_type)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(2,2,3), quiet=true, init_MPI=false, device_type=device_type);
                P   = zeros(nx,  ny,  nz  );
                A   = zeros(nx-1,ny+2,nz+1);
                P, A = GG.wrap_field.((P, A));
                @test GG.sendranges(1, 1, P) == [                    2:2,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(2, 1, P) == [size(P,1)-1:size(P,1)-1,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(1, 2, P) == [            1:size(P,1),                     2:2,             1:size(P,3)]
                @test GG.sendranges(2, 2, P) == [            1:size(P,1), size(P,2)-1:size(P,2)-1,             1:size(P,3)]
                @test GG.sendranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     3:3]
                @test GG.sendranges(2, 3, P) == [            1:size(P,1),             1:size(P,2), size(P,3)-2:size(P,3)-2]
                @test GG.recvranges(1, 1, P) == [                    1:1,             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(2, 1, P) == [    size(P,1):size(P,1),             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 2, P) == [            1:size(P,1),                     1:1,             1:size(P,3)]
                @test GG.recvranges(2, 2, P) == [            1:size(P,1),     size(P,2):size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     1:1]
                @test GG.recvranges(2, 3, P) == [            1:size(P,1),             1:size(P,2),     size(P,3):size(P,3)]
                @test_throws ErrorException  GG.sendranges(1, 1, A)
                @test_throws ErrorException  GG.sendranges(2, 1, A)
                @test GG.sendranges(1, 2, A) == [            1:size(A,1),                     4:4,             1:size(A,3)]
                @test GG.sendranges(2, 2, A) == [            1:size(A,1), size(A,2)-3:size(A,2)-3,             1:size(A,3)]
                @test GG.sendranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     4:4]
                @test GG.sendranges(2, 3, A) == [            1:size(A,1),             1:size(A,2), size(A,3)-3:size(A,3)-3]
                @test_throws ErrorException  GG.recvranges(1, 1, A)
                @test_throws ErrorException  GG.recvranges(2, 1, A)
                @test GG.recvranges(1, 2, A) == [            1:size(A,1),                     1:1,             1:size(A,3)]
                @test GG.recvranges(2, 2, A) == [            1:size(A,1),     size(A,2):size(A,2),             1:size(A,3)]
                @test GG.recvranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     1:1]
                @test GG.recvranges(2, 3, A) == [            1:size(A,1),             1:size(A,2),     size(A,3):size(A,3)]
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "sendranges / recvranges (halowidth > 1, $array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
                nx = 13;
                ny = 9;
                nz = 9;
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(6,4,4), halowidths=(3,1,2), quiet=true, init_MPI=false, device_type=device_type);
                P   = zeros(nx,  ny,  nz  );
                A   = zeros(nx-1,ny+2,nz+1);
                P, A = GG.wrap_field.((P, A));
                @test GG.sendranges(1, 1, P) == [                    4:6,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(2, 1, P) == [size(P,1)-5:size(P,1)-3,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(1, 2, P) == [            1:size(P,1),                     4:4,             1:size(P,3)]
                @test GG.sendranges(2, 2, P) == [            1:size(P,1), size(P,2)-3:size(P,2)-3,             1:size(P,3)]
                @test GG.sendranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     3:4]
                @test GG.sendranges(2, 3, P) == [            1:size(P,1),             1:size(P,2), size(P,3)-3:size(P,3)-2]
                @test GG.recvranges(1, 1, P) == [                    1:3,             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(2, 1, P) == [  size(P,1)-2:size(P,1),             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 2, P) == [            1:size(P,1),                     1:1,             1:size(P,3)]
                @test GG.recvranges(2, 2, P) == [            1:size(P,1),     size(P,2):size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     1:2]
                @test GG.recvranges(2, 3, P) == [            1:size(P,1),             1:size(P,2),   size(P,3)-1:size(P,3)]
                @test_throws ErrorException  GG.sendranges(1, 1, A)
                @test_throws ErrorException  GG.sendranges(2, 1, A)
                @test GG.sendranges(1, 2, A) == [            1:size(A,1),                     6:6,             1:size(A,3)]
                @test GG.sendranges(2, 2, A) == [            1:size(A,1), size(A,2)-5:size(A,2)-5,             1:size(A,3)]
                @test GG.sendranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     4:5]
                @test GG.sendranges(2, 3, A) == [            1:size(A,1),             1:size(A,2), size(A,3)-4:size(A,3)-3]
                @test_throws ErrorException  GG.recvranges(1, 1, A)
                @test_throws ErrorException  GG.recvranges(2, 1, A)
                @test GG.recvranges(1, 2, A) == [            1:size(A,1),                     1:1,             1:size(A,3)]
                @test GG.recvranges(2, 2, A) == [            1:size(A,1),     size(A,2):size(A,2),             1:size(A,3)]
                @test GG.recvranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     1:2]
                @test GG.recvranges(2, 3, A) == [            1:size(A,1),             1:size(A,2),   size(A,3)-1:size(A,3)]
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "write_h2h! / read_h2h!" begin
                init_global_grid(nx, ny, nz; quiet=true, init_MPI=false);
                P  = zeros(nx,  ny,  nz  );
                P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P2 = zeros(size(P));
                halowidths = (1,1,1)
                # (dim=1)
                buf = zeros(halowidths[1], size(P,2), size(P,3));
                ranges = [2:2, 1:size(P,2), 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 1);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 1);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=2)
                buf = zeros(size(P,1), halowidths[2], size(P,3));
                ranges = [1:size(P,1), 3:3, 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 2);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 2);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=3)
                buf = zeros(size(P,1), size(P,2), halowidths[3]);
                ranges = [1:size(P,1), 1:size(P,2), 4:4];
                GG.write_h2h!(buf, P, ranges, 3);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 3);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "write_h2h! / read_h2h! (halowidth > 1)" begin
                init_global_grid(nx, ny, nz; quiet=true, init_MPI=false);
                P  = zeros(nx,  ny,  nz  );
                P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P2 = zeros(size(P));
                halowidths = (3,1,2);
                # (dim=1)
                buf = zeros(halowidths[1], size(P,2), size(P,3));
                ranges = [4:6, 1:size(P,2), 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 1);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 1);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=2)
                buf = zeros(size(P,1), halowidths[2], size(P,3));
                ranges = [1:size(P,1), 4:4, 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 2);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 2);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=3)
                buf = zeros(size(P,1), size(P,2), halowidths[3]);
                ranges = [1:size(P,1), 1:size(P,2), 3:4];
                GG.write_h2h!(buf, P, ranges, 3);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 3);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                finalize_global_grid(finalize_MPI=false);
            end;
            @static if test_cuda || test_amdgpu
                @testset "write_d2x! / write_d2x_async! / write_d2h_async! / read_x2d! / read_x2d_async! / read_h2d_async! (arrays: $array_type)" for (array_type, device_type, gpuzeros, GPUArray) in zip(gpu_array_types, gpu_device_types, gpu_allocators, GPUArrayConstructors)
                    init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                    P  = zeros(nx,  ny,  nz  );
                    P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                    P  = GPUArray(P);
                    halowidths = (1,3,1)
                    if array_type == "CUDA"
                        # (dim=1)
                        dim = 1;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(halowidths[dim], size(P,2), size(P,3));
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [2:2, 1:size(P,2), 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2x_async!(false, buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_x2d_async!(false, buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.unregister(buf_h);
                        # (dim=2)
                        dim = 2;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), halowidths[dim], size(P,3));
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [1:size(P,1), 2:4, 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2x_async!(false, buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_x2d_async!(false, buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.unregister(buf_h);
                        # (dim=3)
                        dim = 3
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), size(P,2), halowidths[dim]);
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [1:size(P,1), 1:size(P,2), 4:4];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2x_async!(false, buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_x2d_async!(false, buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.unregister(buf_h);
                    elseif array_type == "AMDGPU"
                        # (dim=1)
                        dim = 1;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(halowidths[dim], size(P,2), size(P,3));
                        buf_d = GG.register(ROCArray,buf);
                        ranges = [2:2, 1:size(P,2), 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @roc gridsize=nblocks groupsize=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @roc gridsize=nblocks groupsize=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        rocstream = AMDGPU.HIPStream();
                        GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        AMDGPU.unsafe_free!(buf_d);
                        # (dim=2)
                        dim = 2;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), halowidths[dim], size(P,3));
                        buf_d = GG.register(ROCArray,buf);
                        ranges = [1:size(P,1), 2:4, 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @roc gridsize=nblocks groupsize=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @roc gridsize=nblocks groupsize=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        rocstream = AMDGPU.HIPStream();
                        GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        AMDGPU.unsafe_free!(buf_d);
                        # (dim=3)
                        dim = 3
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), size(P,2), halowidths[dim]);
                        buf_d = GG.register(ROCArray,buf);
                        ranges = [1:size(P,1), 1:size(P,2), 4:4];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @roc gridsize=nblocks groupsize=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                         @roc gridsize=nblocks groupsize=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        rocstream = AMDGPU.HIPStream();
                        GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        AMDGPU.unsafe_free!(buf_d);
                    end
                    finalize_global_grid(finalize_MPI=false);
                end;
            end
            @testset "iwrite_sendbufs! (arrays: $array_type)" for (array_type, device_type, zeros, Array) in zip(array_types, device_types, allocators, ArrayConstructors)
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                P = zeros(nx,  ny,  nz  );
                A = zeros(nx-1,ny+2,nz+1);
                P .= Array([iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
                A .= Array([iz*1e2 + iy*1e1 + ix for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]);
                P, A = GG.wrap_field.((P, A));
                GG.allocate_bufs(P, A);
                if     (array_type == "CUDA")   GG.allocate_custreams(P, A);
                elseif (array_type == "AMDGPU") GG.allocate_rocstreams(P, A);
                else                            GG.allocate_tasks(P, A);
                end
                dim = 1
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[3:4,:,:][:]))) # DEBUG: here and later, CPUArray is needed to avoid error in AMDGPU because of mapreduce
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== (eltype(A)(0.0),)))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[3:4,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== (eltype(A)(0.0),))
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[end-3:end-2,:,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== (eltype(A)(0.0),)))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[end-3:end-2,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== (eltype(A)(0.0),))
                end
                dim = 2
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,2,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,4,:][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,2,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,4,:][:]))
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,end-1,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,end-3,:][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,end-1,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,end-3,:][:]))
                end
                dim = 3
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,:,3][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,:,4][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,3][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,4][:]))
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,:,end-2][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,:,end-3][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,end-2][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,end-3][:]))
                end
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "iread_recvbufs! (arrays: $array_type)" for (array_type, device_type, zeros, Array) in zip(array_types, device_types, allocators, ArrayConstructors)
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                P = zeros(nx,  ny,  nz  );
                A = zeros(nx-1,ny+2,nz+1);
                P, A = GG.wrap_field.((P, A));
                GG.allocate_bufs(P, A);
                if     (array_type == "CUDA")   GG.allocate_custreams(P, A);
                elseif (array_type == "AMDGPU") GG.allocate_rocstreams(P, A);
                else                            GG.allocate_tasks(P, A);
                end
                dim = 1
                for n = 1:nneighbors_per_dim
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        GG.gpurecvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.gpurecvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    else
                        GG.recvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.recvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    end
                end
                n = 1
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[1:2,:,:][:])))
                    @test all(CPUArray(                          0.0 .== Array(A.A[1:2,:,:][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[1:2,:,:][:]))
                    @test all(                       0.0 .== CPUArray(A.A[1:2,:,:][:]))
                end
                n = 2
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[end-1:end,:,:][:])))
                    @test all(CPUArray(                          0.0 .== Array(A.A[end-1:end,:,:][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[end-1:end,:,:][:]))
                    @test all(                       0.0 .== CPUArray(A.A[end-1:end,:,:][:]))
                end
                dim = 2
                for n = 1:nneighbors_per_dim
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        GG.gpurecvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.gpurecvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    else
                        GG.recvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.recvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    end
                end
                n = 1
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[:,1,:][:])))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,2,A) .== Array(A.A[:,1,:][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,1,:][:]))
                    @test all(GG.recvbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,1,:][:]))
                end
                n = 2
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[:,end,:][:])))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,2,A) .== Array(A.A[:,end,:][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,end,:][:]))
                    @test all(GG.recvbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,end,:][:]))
                end
                dim = 3
                for n = 1:nneighbors_per_dim
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        GG.gpurecvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.gpurecvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    else
                        GG.recvbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                        GG.recvbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                    end
                end
                n = 1
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[:,:,1][:])))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,2,A) .== Array(A.A[:,:,1][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,1][:]))
                    @test all(GG.recvbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,1][:]))
                end
                n = 2
                GG.iread_recvbufs!(n, dim, P, 1);
                GG.iread_recvbufs!(n, dim, A, 2);
                GG.wait_iread(n, P, 1);
                GG.wait_iread(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,1,P) .== Array(P.A[:,:,end][:])))
                    @test all(CPUArray(GG.gpurecvbuf_flat(n,dim,2,A) .== Array(A.A[:,:,end][:])))
                else
                    @test all(GG.recvbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,end][:]))
                    @test all(GG.recvbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,end][:]))
                end
                finalize_global_grid(finalize_MPI=false);
            end;
            if (nprocs==1)
                @testset "sendrecv_halo_local (arrays: $array_type)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
                    init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                    P = zeros(nx,  ny,  nz  );
                    A = zeros(nx-1,ny+2,nz+1);
                    P, A = GG.wrap_field.((P, A));
                    GG.allocate_bufs(P, A);
                    dim = 1
                    for n = 1:nneighbors_per_dim
                        if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                            GG.gpusendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.gpusendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        else
                            GG.sendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.sendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        end
                    end
                    for n = 1:nneighbors_per_dim
                        GG.sendrecv_halo_local(n, dim, P, 1);
                        GG.sendrecv_halo_local(n, dim, A, 2);
                    end
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,1,P) .== GG.gpusendbuf_flat(2,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,2,A) .== (eltype(A)(0.0),)));  # There is no halo (ol(dim,A) < 2).
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,1,P) .== GG.gpusendbuf_flat(1,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,2,A) .== (eltype(A)(0.0),)));  # There is no halo (ol(dim,A) < 2).
                    else
                        @test all(GG.recvbuf_flat(1,dim,1,P) .== GG.sendbuf_flat(2,dim,1,P));
                        @test all(GG.recvbuf_flat(1,dim,2,A) .== (eltype(A)(0.0),));  # There is no halo (ol(dim,A) < 2).
                        @test all(GG.recvbuf_flat(2,dim,1,P) .== GG.sendbuf_flat(1,dim,1,P));
                        @test all(GG.recvbuf_flat(2,dim,2,A) .== (eltype(A)(0.0),));  # There is no halo (ol(dim,A) < 2).
                    end
                    dim = 2
                    for n = 1:nneighbors_per_dim
                        if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                            GG.gpusendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.gpusendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        else
                            GG.sendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.sendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        end
                    end
                    for n = 1:nneighbors_per_dim
                        GG.sendrecv_halo_local(n, dim, P, 1);
                        GG.sendrecv_halo_local(n, dim, A, 2);
                    end
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,1,P) .== GG.gpusendbuf_flat(2,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,2,A) .== GG.gpusendbuf_flat(2,dim,2,A)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,1,P) .== GG.gpusendbuf_flat(1,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,2,A) .== GG.gpusendbuf_flat(1,dim,2,A)));
                    else
                        @test all(GG.recvbuf_flat(1,dim,1,P) .== GG.sendbuf_flat(2,dim,1,P));
                        @test all(GG.recvbuf_flat(1,dim,2,A) .== GG.sendbuf_flat(2,dim,2,A));
                        @test all(GG.recvbuf_flat(2,dim,1,P) .== GG.sendbuf_flat(1,dim,1,P));
                        @test all(GG.recvbuf_flat(2,dim,2,A) .== GG.sendbuf_flat(1,dim,2,A));
                    end
                    dim = 3
                    for n = 1:nneighbors_per_dim
                        if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                            GG.gpusendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.gpusendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        else
                            GG.sendbuf_flat(n,dim,1,P) .= dim*1e2 + n*1e1 + 1;
                            GG.sendbuf_flat(n,dim,2,A) .= dim*1e2 + n*1e1 + 2;
                        end
                    end
                    for n = 1:nneighbors_per_dim
                        GG.sendrecv_halo_local(n, dim, P, 1);
                        GG.sendrecv_halo_local(n, dim, A, 2);
                    end
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,1,P) .== GG.gpusendbuf_flat(2,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,2,A) .== GG.gpusendbuf_flat(2,dim,2,A)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,1,P) .== GG.gpusendbuf_flat(1,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,2,A) .== GG.gpusendbuf_flat(1,dim,2,A)));
                    else
                        @test all(GG.recvbuf_flat(1,dim,1,P) .== GG.sendbuf_flat(2,dim,1,P));
                        @test all(GG.recvbuf_flat(1,dim,2,A) .== GG.sendbuf_flat(2,dim,2,A));
                        @test all(GG.recvbuf_flat(2,dim,1,P) .== GG.sendbuf_flat(1,dim,1,P));
                        @test all(GG.recvbuf_flat(2,dim,2,A) .== GG.sendbuf_flat(1,dim,2,A));
                    end
                    finalize_global_grid(finalize_MPI=false);
                end
            end
        end;
        if (nprocs>1)
            @testset "irecv_halo! / isend_halo (arrays: $array_type, dim: $dim)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators), dim in 1:ndims_mpi
                dimxyz  = [(d == dim) ? 0 : 1 for d=1:ndims_mpi]; # NOTE: all processes are placed along the dimension dim.
                periods = [(d == dim) ? 1 : 0 for d=1:ndims_mpi];
                nxyz    = (8, 8, 8);                                  # NOTE: periodicity requires a local size of at least 2*overlap-1 in every dimension.
                me, dims, nprocs, coords, comm = init_global_grid(nxyz...; dimx=dimxyz[1], dimy=dimxyz[2], dimz=dimxyz[3], periodx=periods[1], periody=periods[2], periodz=periods[3], overlaps=(4,4,4), halowidths=(2,1,2), quiet=true, init_MPI=false, device_type=device_type);
                P   = zeros(nxyz...);
                A   = zeros(nxyz...);
                P, A = GG.wrap_field.((P, A));
                GG.allocate_bufs(P, A);
                for n = 1:nneighbors_per_dim
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        GG.gpusendbuf(n,dim,1,P) .= 9.0;
                        GG.gpurecvbuf(n,dim,1,P) .= 0;
                        GG.gpusendbuf(n,dim,2,A) .= 9.0;
                        GG.gpurecvbuf(n,dim,2,A) .= 0;
                    else
                        GG.sendbuf(n,dim,1,P) .= 9.0;
                        GG.recvbuf(n,dim,1,P) .= 0;
                        GG.sendbuf(n,dim,2,A) .= 9.0;
                        GG.recvbuf(n,dim,2,A) .= 0;
                    end
                end
                # DEBUG: Filling arrays is async (at least on AMDGPU); sync is needed.
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim))
                    CUDA.synchronize()
                elseif (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    AMDGPU.synchronize()
                end
                reqs  = fill(MPI.REQUEST_NULL, 2, nneighbors_per_dim, 2);
                for n = 1:nneighbors_per_dim
                    reqs[1,n,1] = GG.irecv_halo!(n, dim, P, 1);
                    reqs[2,n,1] = GG.irecv_halo!(n, dim, A, 2);
                    reqs[1,n,2] = GG.isend_halo(n, dim, P, 1);
                    reqs[2,n,2] = GG.isend_halo(n, dim, A, 2);
                end
                @test all(reqs .!= [MPI.REQUEST_NULL])
                MPI.Waitall(reqs[:]);
                for n = 1:nneighbors_per_dim
                    if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                        @test all(CPUArray(GG.gpurecvbuf(n,dim,1,P) .== 9.0))
                        @test all(CPUArray(GG.gpurecvbuf(n,dim,2,A) .== 9.0))
                    else
                        @test all(GG.recvbuf(n,dim,1,P) .== 9.0)
                        @test all(GG.recvbuf(n,dim,2,A) .== 9.0)
                    end
                end
                finalize_global_grid(finalize_MPI=false);
            end;
        end
    end;

    # (Backup field filled with encoded coordinates and set boundary to zeros; then update halo and compare with backuped field; it should be the same again, except for the boundaries that are not halos)
    @testset "4. halo update (arrays: $array_type)" for (array_type, device_type, zeros, Array, CPUArray) in zip(array_types_union, device_types_union, cpu_allocators_union, ArrayConstructors_union, CPUArrayConstructors_union)
        @testset "basic grid (default: periodic)" begin
            @testset "1D" begin
                init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx);
                P    .= eltype(P).([x_g(ix,dx,P) for ix=1:size(P,1)]);
                P_ref = copy(P);
                P[[1, end]] .= (eltype(P)(0.0),);
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !(P == P_ref)
                update_halo!(P);
                @test (P == P_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny);
                P    .= eltype(P).([y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2)]);
                P_ref = copy(P);
                P[[1, end],       :] .= (eltype(P)(0.0),);
                P[       :,[1, end]] .= (eltype(P)(0.0),);
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !(P == P_ref)
                update_halo!(P);
                @test (P == P_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= eltype(P).([z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
                P_ref = copy(P);
                P[[1, end],       :,       :] .= (eltype(P)(0.0),);
                P[       :,[1, end],       :] .= (eltype(P)(0.0),);
                P[       :,       :,[1, end]] .= (eltype(P)(0.0),);
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !(P == P_ref)
                update_halo!(P);
                @test (P == P_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= eltype(P).([z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
                P_ref = copy(P);
                P[[1,2, end-1,end],       :,       :] .= (eltype(P)(0.0),);
                P[               :,[1, end],       :] .= (eltype(P)(0.0),);
                P[               :,       :,[1, end]] .= (eltype(P)(0.0),);
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !(P == P_ref)
                update_halo!(P);
                @test (P == P_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (not periodic)" begin
                me, dims, nprocs, coords = init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= eltype(P).([z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
                P_ref = copy(P);
                P[[1, end],       :,       :] .= (eltype(P)(0.0),);
                P[       :,[1, end],       :] .= (eltype(P)(0.0),);
                P[       :,       :,[1, end]] .= (eltype(P)(0.0),);
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !(P == P_ref)
                update_halo!(P);
                @test all(CPUArray(P)[2:end-1,2:end-1,2:end-1] .== CPUArray(P_ref)[2:end-1,2:end-1,2:end-1])
                if (coords[1] ==         0) @test all(CPUArray(P)[  1,  :,  :] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[      1,2:end-1,2:end-1] .== CPUArray(P_ref)[      1,2:end-1,2:end-1]); end  # Verifcation of corner values would be cumbersome here; it is already sufficiently covered in the periodic tests.
                if (coords[1] == dims[1]-1) @test all(CPUArray(P)[end,  :,  :] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[    end,2:end-1,2:end-1] .== CPUArray(P_ref)[    end,2:end-1,2:end-1]); end
                if (coords[2] ==         0) @test all(CPUArray(P)[  :,  1,  :] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[2:end-1,      1,2:end-1] .== CPUArray(P_ref)[2:end-1,      1,2:end-1]); end
                if (coords[2] == dims[2]-1) @test all(CPUArray(P)[  :,end,  :] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[2:end-1,    end,2:end-1] .== CPUArray(P_ref)[2:end-1,    end,2:end-1]); end
                if (coords[3] ==         0) @test all(CPUArray(P)[  :,  :,  1] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[2:end-1,2:end-1,      1] .== CPUArray(P_ref)[2:end-1,2:end-1,      1]); end
                if (coords[3] == dims[3]-1) @test all(CPUArray(P)[  :,  :,end] .== (eltype(P)(0.0),)); else @test all(CPUArray(P)[2:end-1,2:end-1,    end] .== CPUArray(P_ref)[2:end-1,2:end-1,    end]); end
                finalize_global_grid(finalize_MPI=false);
            end;
        end;
        @testset "staggered grid (default: periodic)" begin
            @testset "1D" begin
                init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                Vx     = zeros(nx+1);
                Vx    .= eltype(Vx).([x_g(ix,dx,Vx) for ix=1:size(Vx,1)]);
                Vx_ref = copy(Vx);
                Vx[[1, end]] .= (eltype(Vx)(0.0),);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !(Vx == Vx_ref)
                update_halo!(Vx);
                @test (Vx == Vx_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                Vy     = zeros(nx,ny+1);
                Vy    .= eltype(Vy).([y_g(iy,dy,Vy)*1e1 + x_g(ix,dx,Vy) for ix=1:size(Vy,1), iy=1:size(Vy,2)]);
                Vy_ref = copy(Vy);
                Vy[[1, end],       :] .= (eltype(Vy)(0.0),);
                Vy[       :,[1, end]] .= (eltype(Vy)(0.0),);
                Vy     = Array(Vy);
                Vy_ref = Array(Vy_ref);
                @require !(Vy == Vy_ref)
                update_halo!(Vy);
                @test (Vy == Vy_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !(Vz == Vz_ref)
                update_halo!(Vz);
                @test (Vz == Vz_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= eltype(Vx).([z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)]);
                Vx_ref = copy(Vx);
                Vx[[1,2, end-1,end],       :,       :] .= (eltype(Vx)(0.0),);
                Vx[               :,[1, end],       :] .= (eltype(Vx)(0.0),);
                Vx[               :,       :,[1, end]] .= (eltype(Vx)(0.0),);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !(Vx == Vx_ref)
                update_halo!(Vx);
                @test (Vx == Vx_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (not periodic)" begin
                me, dims, nprocs, coords = init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !(Vz == Vz_ref)
                update_halo!(Vz);
                @test all(CPUArray(Vz)[2:end-1,2:end-1,2:end-1] .== CPUArray(Vz_ref)[2:end-1,2:end-1,2:end-1])
                if (coords[1] ==         0) @test all(CPUArray(Vz)[  1,  :,  :] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[      1,2:end-1,2:end-1] .== CPUArray(Vz_ref)[      1,2:end-1,2:end-1]); end  # Verifcation of corner values would be cumbersome here; it is already sufficiently covered in the periodic tests.
                if (coords[1] == dims[1]-1) @test all(CPUArray(Vz)[end,  :,  :] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[    end,2:end-1,2:end-1] .== CPUArray(Vz_ref)[    end,2:end-1,2:end-1]); end
                if (coords[2] ==         0) @test all(CPUArray(Vz)[  :,  1,  :] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[2:end-1,      1,2:end-1] .== CPUArray(Vz_ref)[2:end-1,      1,2:end-1]); end
                if (coords[2] == dims[2]-1) @test all(CPUArray(Vz)[  :,end,  :] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[2:end-1,    end,2:end-1] .== CPUArray(Vz_ref)[2:end-1,    end,2:end-1]); end
                if (coords[3] ==         0) @test all(CPUArray(Vz)[  :,  :,  1] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[2:end-1,2:end-1,      1] .== CPUArray(Vz_ref)[2:end-1,2:end-1,      1]); end
                if (coords[3] == dims[3]-1) @test all(CPUArray(Vz)[  :,  :,end] .== (eltype(Vz)(0.0),)); else @test all(CPUArray(Vz)[2:end-1,2:end-1,    end] .== CPUArray(Vz_ref)[2:end-1,2:end-1,    end]); end
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D (no halo in one dim)" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                A     = zeros(nx-1,ny+2);
                A    .= eltype(A).([y_g(iy,dy,A)*1e1 + x_g(ix,dx,A) for ix=1:size(A,1), iy=1:size(A,2)]);
                A_ref = copy(A);
                A[[1, end],       :] .= (eltype(A)(0.0),);
                A[       :,[1, end]] .= (eltype(A)(0.0),);
                A     = Array(A);
                A_ref = Array(A_ref);
                @require !(A == A_ref)
                update_halo!(A);
                @test all(CPUArray(A)[2:end-1,:] .== CPUArray(A_ref)[2:end-1,:])
                @test all(CPUArray(A)[[1, end],:] .== (eltype(A)(0.0),))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (no halo in one dim)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                A     = zeros(nx+2,ny-1,nz+1);
                A    .= eltype(A).([z_g(iz,dz,A)*1e2 + y_g(iy,dy,A)*1e1 + x_g(ix,dx,A) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]);
                A_ref = copy(A);
                A[[1, end],       :,       :] .= (eltype(A)(0.0),);
                A[       :,[1, end],       :] .= (eltype(A)(0.0),);
                A[       :,       :,[1, end]] .= (eltype(A)(0.0),);
                A     = Array(A);
                A_ref = Array(A_ref);
                @require !(A == A_ref)
                update_halo!(A);
                @test all(CPUArray(A)[:,2:end-1,:] .== CPUArray(A_ref)[:,2:end-1,:])
                @test all(CPUArray(A)[:,[1, end],:] .== (eltype(A)(0.0),))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (Complex)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(ComplexF16,nx,ny,nz+1);
                Vz    .= eltype(Vz).([(1+im)*(z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz)) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !(Vz == Vz_ref)
                update_halo!(Vz);
                @test (Vz == Vz_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            if !occursin("cell", array_type) # NOTE: views of CellArrays are not tested.
                @testset "1D (contiguous view)" begin
                    init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                    P_buf = zeros(nx+2);
                    P_buf.= (eltype(P_buf)(-1.0),)
                    P     = @view P_buf[2:end-1];
                    P    .= eltype(P).([x_g(ix,dx,P) for ix=1:size(P,1)]);
                    P_ref = CPUArray(P);
                    P[[1, end]] .= (eltype(P)(0.0),);
                    P_buf = Array(P_buf);                # NOTE: the view is created on the array of the tested type.
                    P     = @view P_buf[2:end-1];
                    @require !(CPUArray(P) == P_ref)
                    update_halo!(P);
                    @test (CPUArray(P) == P_ref)
                    @test all(CPUArray(P_buf)[[1, end]] .== (eltype(P_buf)(-1.0),))
                    finalize_global_grid(finalize_MPI=false);
                end;
                @testset "3D (non-contiguous view)" begin
                    init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                    Vz_buf = Array(zeros(nx+2,ny+2,nz+3));
                    Vz     = @view Vz_buf[2:end-1,2:end-1,2:end-1];
                    @test_throws ErrorException update_halo!(Vz) # Error: Vz is non-contiguous.
                    finalize_global_grid(finalize_MPI=false);
                end;
            end
            @testset "3D (changing datatype)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vx     = zeros(Float32,nx+1,ny,nz);
                Vx    .= eltype(Vx).([z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)]);
                Vx_ref = copy(Vx);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz) .== CPUArray(Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz) .== CPUArray(Vz_ref))
                Vx[[1, end],       :,       :] .= (eltype(Vx)(0.0),);
                Vx[       :,[1, end],       :] .= (eltype(Vx)(0.0),);
                Vx[       :,       :,[1, end]] .= (eltype(Vx)(0.0),);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vx) .== CPUArray(Vx_ref))
                update_halo!(Vx);
                @test all(CPUArray(Vx) .== CPUArray(Vx_ref))
                #TODO: added for GPU - quick fix:
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz) .== CPUArray(Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz) .== CPUArray(Vz_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (changing datatype) (Complex)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vx     = zeros(ComplexF64,nx+1,ny,nz);
                Vx    .= eltype(Vx).([(1+im)*(z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx)) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)]);
                Vx_ref = copy(Vx);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz) .== CPUArray(Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz) .== CPUArray(Vz_ref))
                Vx[[1, end],       :,       :] .= (eltype(Vx)(0.0),);
                Vx[       :,[1, end],       :] .= (eltype(Vx)(0.0),);
                Vx[       :,       :,[1, end]] .= (eltype(Vx)(0.0),);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vx) .== CPUArray(Vx_ref))
                update_halo!(Vx);
                @test all(CPUArray(Vx) .== CPUArray(Vx_ref))
                #TODO: added for GPU - quick fix:
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz) .== CPUArray(Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz) .== CPUArray(Vz_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (two fields simultaneously)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= eltype(Vx).([z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)]);
                Vx_ref = copy(Vx);
                Vz[[1, end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[       :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[       :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vx[[1, end],       :,       :] .= (eltype(Vx)(0.0),);
                Vx[       :,[1, end],       :] .= (eltype(Vx)(0.0),);
                Vx[       :,       :,[1, end]] .= (eltype(Vx)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !(Vz == Vz_ref)
                @require !(Vx == Vx_ref)
                update_halo!(Vz, Vx);
                @test (Vz == Vz_ref)
                @test (Vx == Vx_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (two fields simultaneously, non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= eltype(Vz).([z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)]);
                Vz_ref = copy(Vz);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= eltype(Vx).([z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)]);
                Vx_ref = copy(Vx);
                Vz[[1,2, end-1,end],       :,       :] .= (eltype(Vz)(0.0),);
                Vz[               :,[1, end],       :] .= (eltype(Vz)(0.0),);
                Vz[               :,       :,[1, end]] .= (eltype(Vz)(0.0),);
                Vx[[1,2, end-1,end],       :,       :] .= (eltype(Vx)(0.0),);
                Vx[               :,[1, end],       :] .= (eltype(Vx)(0.0),);
                Vx[               :,       :,[1, end]] .= (eltype(Vx)(0.0),);
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !(Vz == Vz_ref)
                @require !(Vx == Vx_ref)
                update_halo!(Vz, Vx);
                @test (Vz == Vz_ref)
                @test (Vx == Vx_ref)
                finalize_global_grid(finalize_MPI=false);
            end;
        end;
    end;

    @testset "5. halo update (buffers used before for other arrays) (arrays: $array_type)" for (array_type, device_type, zeros, Array, CPUArray) in zip(array_types_union, device_types_union, cpu_allocators_union, ArrayConstructors_union, CPUArrayConstructors_union)
        @testset "3D (CPU array of the same size used the buffers before)" begin                          # issue #140
            init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
            P_cpu = zeros(nx, ny, nz);
            update_halo!(P_cpu);                                                                           # The CPU array sizes the host buffers of the slot.
            P     = zeros(nx, ny, nz);
            P    .= eltype(P).([z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
            P_ref = copy(P);
            P[[1, end],       :,       :] .= (eltype(P)(0.0),);
            P[       :,[1, end],       :] .= (eltype(P)(0.0),);
            P[       :,       :,[1, end]] .= (eltype(P)(0.0),);
            P     = Array(P);
            P_ref = Array(P_ref);
            @require !(P == P_ref)
            update_halo!(P);
            @test (P == P_ref)
            finalize_global_grid(finalize_MPI=false);
        end;
        @testset "3D (buffers reallocated for a larger CPU array in between)" begin
            init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
            P     = zeros(nx, ny, nz);
            P    .= eltype(P).([z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
            P_ref = copy(P);
            P[[1, end],       :,       :] .= (eltype(P)(0.0),);
            P[       :,[1, end],       :] .= (eltype(P)(0.0),);
            P[       :,       :,[1, end]] .= (eltype(P)(0.0),);
            P     = Array(P);
            P_ref = Array(P_ref);
            @require !(P == P_ref)
            update_halo!(P);                                                                               # The array sizes the host buffers of the slot (and registers them for the GPU if it is a GPU array).
            @test (P == P_ref)
            Q_cpu = zeros(nx+3, ny+3, nz+3);
            update_halo!(Q_cpu);                                                                           # The larger CPU array replaces the host buffers of the slot.
            P2     = zeros(nx, ny, nz);
            P2    .= eltype(P2).([z_g(iz,dz,P2)*1e2 + y_g(iy,dy,P2)*1e1 + x_g(ix,dx,P2) for ix=1:size(P2,1), iy=1:size(P2,2), iz=1:size(P2,3)]);
            P2_ref = copy(P2);
            P2[[1, end],       :,       :] .= (eltype(P2)(0.0),);
            P2[       :,[1, end],       :] .= (eltype(P2)(0.0),);
            P2[       :,       :,[1, end]] .= (eltype(P2)(0.0),);
            P2     = Array(P2);
            P2_ref = Array(P2_ref);
            @require !(P2 == P2_ref)
            update_halo!(P2);                                                                              # The array must use the current host buffers (not the replaced ones).
            @test (P2 == P2_ref)
            Q     = zeros(nx+3, ny+3, nz+3);
            Q    .= eltype(Q).([z_g(iz,dz,Q)*1e2 + y_g(iy,dy,Q)*1e1 + x_g(ix,dx,Q) for ix=1:size(Q,1), iy=1:size(Q,2), iz=1:size(Q,3)]);
            Q_ref = copy(Q);
            Q[[1, end],       :,       :] .= (eltype(Q)(0.0),);
            Q[       :,[1, end],       :] .= (eltype(Q)(0.0),);
            Q[       :,       :,[1, end]] .= (eltype(Q)(0.0),);
            Q     = Array(Q);
            Q_ref = Array(Q_ref);
            @require !(Q == Q_ref)
            update_halo!(Q);                                                                               # The larger array must get buffers of sufficient size.
            @test (Q == Q_ref)
            finalize_global_grid(finalize_MPI=false);
        end;
    end;
end;

## Test tear down
MPI.Finalize()
