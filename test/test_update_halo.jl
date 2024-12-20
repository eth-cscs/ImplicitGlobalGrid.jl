# NOTE: All tests of this file can be run with any number of processes.
# Nearly all of the functionality can however be verified with one single process
# (thanks to the usage of periodic boundaries in most of the full halo update tests).

push!(LOAD_PATH, "../src")
using Test
import MPI, Polyester
using CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require, longnameof

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

array_types          = ["CPU"]
gpu_array_types      = []
device_types         = ["auto"]
gpu_device_types     = []
allocators           = Function[zeros]
gpu_allocators       = []
ArrayConstructors    = [Array]
GPUArrayConstructors = []
CPUArray             = Array
if test_cuda
    cuzeros = CUDA.zeros
    push!(array_types, "CUDA")
    push!(gpu_array_types, "CUDA")
    push!(device_types, "CUDA")
    push!(gpu_device_types, "CUDA")
    push!(allocators, cuzeros)
    push!(gpu_allocators, cuzeros)
    push!(ArrayConstructors, CuArray)
    push!(GPUArrayConstructors, CuArray)
end
if test_amdgpu
    roczeros = AMDGPU.zeros
    push!(array_types, "AMDGPU")
    push!(gpu_array_types, "AMDGPU")
    push!(device_types, "AMDGPU")
    push!(gpu_device_types, "AMDGPU")
    push!(allocators, roczeros)
    push!(gpu_allocators, roczeros)
    push!(ArrayConstructors, ROCArray)
    push!(GPUArrayConstructors, ROCArray)
end

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
    @testset "1. argument check ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
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
        @test_throws ErrorException update_halo!(P, A, A)                       # Error: A is given twice.
        @test_throws ErrorException update_halo!(P, A, A2)                      # Error: A2 is duplicate of A (an alias; it points to the same memory).
        @test_throws ErrorException update_halo!(P, A, A, A2)                   # Error: the second A and A2 are duplicates of the first A.
        @test_throws ErrorException update_halo!(Z, Z2)                         # Error: Z2 is duplicate of Z (an alias; it points to the same memory).
        @test_throws ErrorException update_halo!(Z, P)                          # Error: P is of different type than Z.
        @test_throws ErrorException update_halo!(Z, P, A)                       # Error: P and A are of different type than Z.
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "2. buffer allocation ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
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
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "3. data transfer components" begin
        @testset "iwrite_sendbufs! / iread_recvbufs!" begin
            @testset "sendranges / recvranges ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
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
                @testset "write_d2x! / write_d2h_async! / read_x2d! / read_h2d_async! ($array_type arrays)" for (array_type, device_type, gpuzeros, GPUArray) in zip(gpu_array_types, gpu_device_types, gpu_allocators, GPUArrayConstructors)
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
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
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
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
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
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
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
                        # buf .= 0.0; # DEBUG: diabling read_x2x_async! tests for now in AMDGPU backend because there is an issue most likely in HIP
                        # P2  .= 0.0;
                        # rocstream = AMDGPU.HIPStream();
                        # GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        # GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        # AMDGPU.unsafe_free!(buf_d);
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
                        # buf .= 0.0; # DEBUG: diabling read_x2x_async! tests for now in AMDGPU backend because there is an issue most likely in HIP
                        # P2  .= 0.0;
                        # rocstream = AMDGPU.HIPStream();
                        # GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        # GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        # AMDGPU.unsafe_free!(buf_d);
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
                        # buf .= 0.0; # DEBUG: diabling read_x2x_async! tests for now in AMDGPU backend because there is an issue most likely in HIP
                        # P2  .= 0.0;
                        # rocstream = AMDGPU.HIPStream();
                        # GG.write_d2h_async!(buf, P, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        # GG.read_h2d_async!(buf, P2, ranges, rocstream); AMDGPU.synchronize();
                        # @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        # AMDGPU.unsafe_free!(buf_d);
                    end
                    finalize_global_grid(finalize_MPI=false);
                end;
            end
            @testset "iwrite_sendbufs! ($array_type arrays)" for (array_type, device_type, zeros, Array) in zip(array_types, device_types, allocators, ArrayConstructors)
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
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== 0.0))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[3:4,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== 0.0)
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[end-3:end-2,:,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== 0.0))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[end-3:end-2,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== 0.0)
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
            @testset "iread_recvbufs! ($array_type arrays)" for (array_type, device_type, zeros, Array) in zip(array_types, device_types, allocators, ArrayConstructors)
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
                @testset "sendrecv_halo_local ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
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
                        @test all(CPUArray(GG.gpurecvbuf_flat(1,dim,2,A) .== 0.0));  # There is no halo (ol(dim,A) < 2).
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,1,P) .== GG.gpusendbuf_flat(1,dim,1,P)));
                        @test all(CPUArray(GG.gpurecvbuf_flat(2,dim,2,A) .== 0.0));  # There is no halo (ol(dim,A) < 2).
                    else
                        @test all(GG.recvbuf_flat(1,dim,1,P) .== GG.sendbuf_flat(2,dim,1,P));
                        @test all(GG.recvbuf_flat(1,dim,2,A) .== 0.0);  # There is no halo (ol(dim,A) < 2).
                        @test all(GG.recvbuf_flat(2,dim,1,P) .== GG.sendbuf_flat(1,dim,1,P));
                        @test all(GG.recvbuf_flat(2,dim,2,A) .== 0.0);  # There is no halo (ol(dim,A) < 2).
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
            @testset "irecv_halo! / isend_halo ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
                me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz; dimy=1, dimz=1, periodx=1, overlaps=(4,4,4), halowidths=(2,1,2), quiet=true, init_MPI=false, device_type=device_type);
                P   = zeros(nx,ny,nz);
                A   = zeros(nx,ny,nz);
                P, A = GG.wrap_field.((P, A));
                dim = 1;
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
                MPI.Waitall!(reqs[:]);
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
    @testset "4. halo update ($array_type arrays)" for (array_type, device_type, Array) in zip(array_types, device_types, ArrayConstructors)
        @testset "basic grid (default: periodic)" begin
            @testset "1D" begin
                init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx);
                P    .= [x_g(ix,dx,P) for ix=1:size(P,1)];
                P_ref = copy(P);
                P[[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref)) # DEBUG: CPUArray needed here and onwards as mapreduce! is failing on AMDGPU (see https://github.com/JuliaGPU/AMDGPU.jl/issues/210)
                update_halo!(P);
                @test all(CPUArray(P .== P_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny);
                P    .= [y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2)];
                P_ref = copy(P);
                P[[1, end],       :] .= 0.0;
                P[       :,[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref))
                update_halo!(P);
                @test all(CPUArray(P .== P_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P_ref = copy(P);
                P[[1, end],       :,       :] .= 0.0;
                P[       :,[1, end],       :] .= 0.0;
                P[       :,       :,[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref))
                update_halo!(P);
                @test all(CPUArray(P .== P_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P_ref = copy(P);
                P[[1,2, end-1,end],       :,       :] .= 0.0;
                P[               :,[1, end],       :] .= 0.0;
                P[               :,       :,[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref))
                update_halo!(P);
                @test all(CPUArray(P .== P_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (not periodic)" begin
                me, dims, nprocs, coords = init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                P     = zeros(nx, ny, nz);
                P    .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P_ref = copy(P);
                P[[1, end],       :,       :] .= 0.0;
                P[       :,[1, end],       :] .= 0.0;
                P[       :,       :,[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref))
                update_halo!(P);
                @test all(CPUArray(P[2:end-1,2:end-1,2:end-1] .== P_ref[2:end-1,2:end-1,2:end-1]))
                if (coords[1] ==         0) @test all(CPUArray(P[  1,  :,  :] .== 0.0)); else @test all(CPUArray(P[      1,2:end-1,2:end-1] .== P_ref[      1,2:end-1,2:end-1])); end  # Verifcation of corner values would be cumbersome here; it is already sufficiently covered in the periodic tests.
                if (coords[1] == dims[1]-1) @test all(CPUArray(P[end,  :,  :] .== 0.0)); else @test all(CPUArray(P[    end,2:end-1,2:end-1] .== P_ref[    end,2:end-1,2:end-1])); end
                if (coords[2] ==         0) @test all(CPUArray(P[  :,  1,  :] .== 0.0)); else @test all(CPUArray(P[2:end-1,      1,2:end-1] .== P_ref[2:end-1,      1,2:end-1])); end
                if (coords[2] == dims[2]-1) @test all(CPUArray(P[  :,end,  :] .== 0.0)); else @test all(CPUArray(P[2:end-1,    end,2:end-1] .== P_ref[2:end-1,    end,2:end-1])); end
                if (coords[3] ==         0) @test all(CPUArray(P[  :,  :,  1] .== 0.0)); else @test all(CPUArray(P[2:end-1,2:end-1,      1] .== P_ref[2:end-1,2:end-1,      1])); end
                if (coords[3] == dims[3]-1) @test all(CPUArray(P[  :,  :,end] .== 0.0)); else @test all(CPUArray(P[2:end-1,2:end-1,    end] .== P_ref[2:end-1,2:end-1,    end])); end
                finalize_global_grid(finalize_MPI=false);
            end;
        end;
        @testset "staggered grid (default: periodic)" begin
            @testset "1D" begin
                init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                Vx     = zeros(nx+1);
                Vx    .= [x_g(ix,dx,Vx) for ix=1:size(Vx,1)];
                Vx_ref = copy(Vx);
                Vx[[1, end]] .= 0.0;
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vx .== Vx_ref))
                update_halo!(Vx);
                @test all(CPUArray(Vx .== Vx_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                Vy     = zeros(nx,ny+1);
                Vy    .= [y_g(iy,dy,Vy)*1e1 + x_g(ix,dx,Vy) for ix=1:size(Vy,1), iy=1:size(Vy,2)];
                Vy_ref = copy(Vy);
                Vy[[1, end],       :] .= 0.0;
                Vy[       :,[1, end]] .= 0.0;
                Vy     = Array(Vy);
                Vy_ref = Array(Vy_ref);
                @require !all(CPUArray(Vy .== Vy_ref))
                update_halo!(Vy);
                @test all(CPUArray(Vy .== Vy_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= 0.0;
                Vz[       :,[1, end],       :] .= 0.0;
                Vz[       :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz .== Vz_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= [z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)];
                Vx_ref = copy(Vx);
                Vx[[1,2, end-1,end],       :,       :] .= 0.0;
                Vx[               :,[1, end],       :] .= 0.0;
                Vx[               :,       :,[1, end]] .= 0.0;
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vx .== Vx_ref))
                update_halo!(Vx);
                @test all(CPUArray(Vx .== Vx_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (not periodic)" begin
                me, dims, nprocs, coords = init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= 0.0;
                Vz[       :,[1, end],       :] .= 0.0;
                Vz[       :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz[2:end-1,2:end-1,2:end-1] .== Vz_ref[2:end-1,2:end-1,2:end-1]))
                if (coords[1] ==         0) @test all(CPUArray(Vz[  1,  :,  :] .== 0.0)); else @test all(CPUArray(Vz[      1,2:end-1,2:end-1] .== Vz_ref[      1,2:end-1,2:end-1])); end  # Verifcation of corner values would be cumbersome here; it is already sufficiently covered in the periodic tests.
                if (coords[1] == dims[1]-1) @test all(CPUArray(Vz[end,  :,  :] .== 0.0)); else @test all(CPUArray(Vz[    end,2:end-1,2:end-1] .== Vz_ref[    end,2:end-1,2:end-1])); end
                if (coords[2] ==         0) @test all(CPUArray(Vz[  :,  1,  :] .== 0.0)); else @test all(CPUArray(Vz[2:end-1,      1,2:end-1] .== Vz_ref[2:end-1,      1,2:end-1])); end
                if (coords[2] == dims[2]-1) @test all(CPUArray(Vz[  :,end,  :] .== 0.0)); else @test all(CPUArray(Vz[2:end-1,    end,2:end-1] .== Vz_ref[2:end-1,    end,2:end-1])); end
                if (coords[3] ==         0) @test all(CPUArray(Vz[  :,  :,  1] .== 0.0)); else @test all(CPUArray(Vz[2:end-1,2:end-1,      1] .== Vz_ref[2:end-1,2:end-1,      1])); end
                if (coords[3] == dims[3]-1) @test all(CPUArray(Vz[  :,  :,end] .== 0.0)); else @test all(CPUArray(Vz[2:end-1,2:end-1,    end] .== Vz_ref[2:end-1,2:end-1,    end])); end
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "2D (no halo in one dim)" begin
                init_global_grid(nx, ny, 1; periodx=1, periody=1, quiet=true, init_MPI=false, device_type=device_type);
                A     = zeros(nx-1,ny+2);
                A    .= [y_g(iy,dy,A)*1e1 + x_g(ix,dx,A) for ix=1:size(A,1), iy=1:size(A,2)];
                A_ref = copy(A);
                A[[1, end],       :] .= 0.0;
                A[       :,[1, end]] .= 0.0;
                A     = Array(A);
                A_ref = Array(A_ref);
                @require !all(CPUArray(A .== A_ref))
                update_halo!(A);
                @test all(CPUArray(A[2:end-1,:] .== A_ref[2:end-1,:]))
                @test all(CPUArray(A[[1, end],:] .== 0.0))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (no halo in one dim)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                A     = zeros(nx+2,ny-1,nz+1);
                A    .= [z_g(iz,dz,A)*1e2 + y_g(iy,dy,A)*1e1 + x_g(ix,dx,A) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)];
                A_ref = copy(A);
                A[[1, end],       :,       :] .= 0.0;
                A[       :,[1, end],       :] .= 0.0;
                A[       :,       :,[1, end]] .= 0.0;
                A     = Array(A);
                A_ref = Array(A_ref);
                @require !all(CPUArray(A .== A_ref))
                update_halo!(A);
                @test all(CPUArray(A[:,2:end-1,:] .== A_ref[:,2:end-1,:]))
                @test all(CPUArray(A[:,[1, end],:] .== 0.0))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (Complex)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(ComplexF16,nx,ny,nz+1);
                Vz    .= [(1+im)*(z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz)) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= 0.0;
                Vz[       :,[1, end],       :] .= 0.0;
                Vz[       :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz .== Vz_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "1D (contiguous view)" begin
                init_global_grid(nx, 1, 1; periodx=1, quiet=true, init_MPI=false, device_type=device_type);
                P_buf = -1.0 .+ zeros(nx+2);
                P     = @view P_buf[2:end-1];
                P    .= [x_g(ix,dx,P) for ix=1:size(P,1)];
                P_ref = copy(P);
                P[[1, end]] .= 0.0;
                P     = Array(P);
                P_ref = Array(P_ref);
                @require !all(CPUArray(P .== P_ref)) # DEBUG: CPUArray needed here and onwards as mapreduce! is failing on AMDGPU (see https://github.com/JuliaGPU/AMDGPU.jl/issues/210)
                update_halo!(P);
                @test all(CPUArray(P .== P_ref))
                @test all(CPUArray(P_buf[[1, end]] .== -1.0))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (non-contiguous view)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz_buf = -1.0 .+ zeros(nx+2,ny+2,nz+3);
                Vz     = @view Vz_buf[2:end-1,2:end-1,2:end-1];
                Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vz[[1, end],       :,       :] .= 0.0;
                Vz[       :,[1, end],       :] .= 0.0;
                Vz[       :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                update_halo!(Vz);
                @test all(CPUArray(Vz .== Vz_ref))
                @test all(CPUArray(Vz_buf[[1, end],:,:] .== -1.0))
                @test all(CPUArray(Vz_buf[:,[1, end],:] .== -1.0))
                @test all(CPUArray(Vz_buf[:,:,[1, end]] .== -1.0))
                finalize_global_grid(finalize_MPI=false);
            end;
            # @testset "3D (changing datatype)" begin
            #     init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
            #     Vz     = zeros(nx,ny,nz+1);
            #     Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
            #     Vz_ref = copy(Vz);
            #     Vx     = zeros(Float32,nx+1,ny,nz);
            #     Vx    .= [z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)];
            #     Vx_ref = copy(Vx);
            #     Vz[[1, end],       :,       :] .= 0.0;
            #     Vz[       :,[1, end],       :] .= 0.0;
            #     Vz[       :,       :,[1, end]] .= 0.0;
            #     Vz     = Array(Vz);
            #     Vz_ref = Array(Vz_ref);
            #     @require !all(Vz .== Vz_ref)
            #     update_halo!(Vz);
            #     @test all(Vz .== Vz_ref)
            #     Vx[[1, end],       :,       :] .= 0.0;
            #     Vx[       :,[1, end],       :] .= 0.0;
            #     Vx[       :,       :,[1, end]] .= 0.0;
            #     Vx     = Array(Vx);
            #     Vx_ref = Array(Vx_ref);
            #     @require !all(Vx .== Vx_ref)
            #     update_halo!(Vx);
            #     @test all(Vx .== Vx_ref)
            #     #TODO: added for GPU - quick fix:
            #     Vz     = zeros(nx,ny,nz+1);
            #     Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
            #     Vz_ref = copy(Vz);
            #     Vz[[1, end],       :,       :] .= 0.0;
            #     Vz[       :,[1, end],       :] .= 0.0;
            #     Vz[       :,       :,[1, end]] .= 0.0;
            #     Vz     = Array(Vz);
            #     Vz_ref = Array(Vz_ref);
            #     @require !all(Vz .== Vz_ref)
            #     update_halo!(Vz);
            #     @test all(Vz .== Vz_ref)
            #     finalize_global_grid(finalize_MPI=false);
            # end;
            # @testset "3D (changing datatype) (Complex)" begin
            #     init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
            #     Vz     = zeros(nx,ny,nz+1);
            #     Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
            #     Vz_ref = copy(Vz);
            #     Vx     = zeros(ComplexF64,nx+1,ny,nz);
            #     Vx    .= [(1+im)*(z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx)) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)];
            #     Vx_ref = copy(Vx);
            #     Vz[[1, end],       :,       :] .= 0.0;
            #     Vz[       :,[1, end],       :] .= 0.0;
            #     Vz[       :,       :,[1, end]] .= 0.0;
            #     Vz     = Array(Vz);
            #     Vz_ref = Array(Vz_ref);
            #     @require !all(Vz .== Vz_ref)
            #     update_halo!(Vz);
            #     @test all(Vz .== Vz_ref)
            #     Vx[[1, end],       :,       :] .= 0.0;
            #     Vx[       :,[1, end],       :] .= 0.0;
            #     Vx[       :,       :,[1, end]] .= 0.0;
            #     Vx     = Array(Vx);
            #     Vx_ref = Array(Vx_ref);
            #     @require !all(Vx .== Vx_ref)
            #     update_halo!(Vx);
            #     @test all(Vx .== Vx_ref)
            #     #TODO: added for GPU - quick fix:
            #     Vz     = zeros(nx,ny,nz+1);
            #     Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
            #     Vz_ref = copy(Vz);
            #     Vz[[1, end],       :,       :] .= 0.0;
            #     Vz[       :,[1, end],       :] .= 0.0;
            #     Vz[       :,       :,[1, end]] .= 0.0;
            #     Vz     = Array(Vz);
            #     Vz_ref = Array(Vz_ref);
            #     @require !all(Vz .== Vz_ref)
            #     update_halo!(Vz);
            #     @test all(Vz .== Vz_ref)
            #     finalize_global_grid(finalize_MPI=false);
            # end;
            @testset "3D (two fields simultaneously)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= [z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)];
                Vx_ref = copy(Vx);
                Vz[[1, end],       :,       :] .= 0.0;
                Vz[       :,[1, end],       :] .= 0.0;
                Vz[       :,       :,[1, end]] .= 0.0;
                Vx[[1, end],       :,       :] .= 0.0;
                Vx[       :,[1, end],       :] .= 0.0;
                Vx[       :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                @require !all(CPUArray(Vx .== Vx_ref))
                update_halo!(Vz, Vx);
                @test all(CPUArray(Vz .== Vz_ref))
                @test all(CPUArray(Vx .== Vx_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "3D (two fields simultaneously, non-default overlap and halowidth)" begin
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                Vz     = zeros(nx,ny,nz+1);
                Vz    .= [z_g(iz,dz,Vz)*1e2 + y_g(iy,dy,Vz)*1e1 + x_g(ix,dx,Vz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)];
                Vz_ref = copy(Vz);
                Vx     = zeros(nx+1,ny,nz);
                Vx    .= [z_g(iz,dz,Vx)*1e2 + y_g(iy,dy,Vx)*1e1 + x_g(ix,dx,Vx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)];
                Vx_ref = copy(Vx);
                Vz[[1,2, end-1,end],       :,       :] .= 0.0;
                Vz[               :,[1, end],       :] .= 0.0;
                Vz[               :,       :,[1, end]] .= 0.0;
                Vx[[1,2, end-1,end],       :,       :] .= 0.0;
                Vx[               :,[1, end],       :] .= 0.0;
                Vx[               :,       :,[1, end]] .= 0.0;
                Vz     = Array(Vz);
                Vz_ref = Array(Vz_ref);
                Vx     = Array(Vx);
                Vx_ref = Array(Vx_ref);
                @require !all(CPUArray(Vz .== Vz_ref))
                @require !all(CPUArray(Vx .== Vx_ref))
                update_halo!(Vz, Vx);
                @test all(CPUArray(Vz .== Vz_ref))
                @test all(CPUArray(Vx .== Vx_ref))
                finalize_global_grid(finalize_MPI=false);
            end;
        end;
    end;
end;

## Test tear down
MPI.Finalize()