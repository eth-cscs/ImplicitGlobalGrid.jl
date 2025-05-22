push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require
macro coords(i) :(GG.global_grid().coords[$i]) end


## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD);
@require nprocs == 1  # NOTE: these tests require nprocs == 1.

@testset "$(basename(@__FILE__))" begin
    @testset "1. grid functions" begin
        lx = 8;
        ly = 8;
        lz = 8;
        nx = 5;
        ny = 5;
        nz = 5;
        P   = zeros(nx,  ny,  nz  );
        Vx  = zeros(nx+1,ny,  nz  );
        Vz  = zeros(nx,  ny,  nz+1);
        A   = zeros(nx+2,ny,  nz+2);
        Sxz = zeros(nx-2,ny-1,nz-2);
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, quiet=true, init_MPI=false);
        @testset "nx_g / ny_g / nz_g" begin
            @test nx_g() == nx
            @test ny_g() == ny
            @test nz_g() == nz-2
            @test nx_g(P) == nx
            @test ny_g(P) == ny
            @test nz_g(P) == nz-2
            @test nx_g(Vx) == nx+1
            @test ny_g(Vx) == ny
            @test nz_g(Vx) == nz-2
            @test nx_g(Vz) == nx
            @test ny_g(Vz) == ny
            @test nz_g(Vz) == nz-2+1
            @test nx_g(A) == nx+2
            @test ny_g(A) == ny
            @test nz_g(A) == nz-2+2
            @test nx_g(Sxz) == nx-2
            @test ny_g(Sxz) == ny-1
            @test nz_g(Sxz) == nz-2-2
        end;
        @testset "ix_g / iy_g / iz_g" begin
            # (for P)
            @test [ix_g(ix, P) for ix = 1:size(P,1)] == [1, 2, 3, 4, 5]
            @test [iy_g(iy, P) for iy = 1:size(P,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, P) for iz = 1:size(P,3)] == [3, 1, 2, 3, 1]
            @test [iz_g(iz, P; wrap_periodic=false) for iz = 1:size(P,3)] == [0, 1, 2, 3, 4]
            # (for Vx)
            @test [ix_g(ix, Vx) for ix = 1:size(Vx,1)] == [1, 2, 3, 4, 5, 6]
            @test [iy_g(iy, Vx) for iy = 1:size(Vx,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, Vx) for iz = 1:size(Vx,3)] == [3, 1, 2, 3, 1]
            @test [iz_g(iz, Vx; wrap_periodic=false) for iz = 1:size(Vx,3)] == [0, 1, 2, 3, 4]
            # (for Vz)
            @test [ix_g(ix, Vz) for ix = 1:size(Vz,1)] == [1, 2, 3, 4, 5]
            @test [iy_g(iy, Vz) for iy = 1:size(Vz,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, Vz) for iz = 1:size(Vz,3)] == [3, 1, 2, 3, 1, 2] #Alternative: [2, 3, 1, 2, 3, 1]
            @test [iz_g(iz, Vz; wrap_periodic=false) for iz = 1:size(Vz,3)] == [0, 1, 2, 3, 4, 5] #Alternative: [-1, 0, 1, 2, 3, 4]
            # (for A)
            @test [ix_g(ix, A) for ix = 1:size(A,1)] == [1, 2, 3, 4, 5, 6, 7]
            @test [iy_g(iy, A) for iy = 1:size(A,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, A) for iz = 1:size(A,3)] == [2, 3, 1, 2, 3, 1, 2]
            @test [iz_g(iz, A; wrap_periodic=false) for iz = 1:size(A,3)] == [-1, 0, 1, 2, 3, 4, 5]
            # (for Sxz)
            @test [ix_g(ix, Sxz) for ix = 1:size(Sxz,1)] == [1, 2, 3]
            @test [iy_g(iy, Sxz) for iy = 1:size(Sxz,2)] == [1, 2, 3, 4]
            @test [iz_g(iz, Sxz) for iz = 1:size(Sxz,3)] == [1, 2, 3]
            @test [iz_g(iz, Sxz; wrap_periodic=false) for iz = 1:size(Sxz,3)] == [1, 2, 3]
        end;
        @testset "x_g / y_g / z_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            # (for P)
            @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [8.0, 0.0, 4.0, 8.0, 0.0]
            @test [z_g(iz,dz,P; wrap_periodic=false) for iz = 1:size(P,3)] == [-4.0, 0.0, 4.0, 8.0, 12.0]
            # (for Vx)
            @test [x_g(ix,dx,Vx) for ix = 1:size(Vx,1)] == [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
            @test [y_g(iy,dy,Vx) for iy = 1:size(Vx,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Vx) for iz = 1:size(Vx,3)] == [8.0, 0.0, 4.0, 8.0, 0.0]
            @test [z_g(iz,dz,Vx; wrap_periodic=false) for iz = 1:size(Vx,3)] == [-4.0, 0.0, 4.0, 8.0, 12.0]
            # (for Vz)
            @test [x_g(ix,dx,Vz) for ix = 1:size(Vz,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Vz) for iy = 1:size(Vz,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Vz) for iz = 1:size(Vz,3)] == [ 6.0, -2.0,  2.0,  6.0, -2.0,  2.0]
            #                       base grid (z dim):        [ 8.0,  0.0,  4.0,  8.0,  0.0]
            #                       possible alternative:  [6.0,  10.0,  2.0,  6.0, 10.0,  2.0]  # This would be a possible alternative way to define {x,y,z}_g; however, we decided that the grid should start at -2.0 in this case and have more overlap be at the end (the advantage is that it is more in agreement with the non-periodic case: +1 cell, means starts one earlier... furthermore, the extents_g function relies on this)
            #                       wrong:                 [ 6.0, -2.0,  2.0,  6.0, 10.0,  2.0]  # The 2nd and the 2nd-last cell must be the same due to the overlap of 3.
            @test [z_g(iz,dz,Vz; wrap_periodic=false) for iz = 1:size(Vz,3)] == [-6.0, -2.0, 2.0, 6.0, 10.0, 14.0]
            # (for A)
            @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [4.0, 8.0, 0.0, 4.0, 8.0, 0.0, 4.0]
            #                       base grid (z dim):        [8.0, 0.0, 4.0, 8.0, 0.0]
            @test [z_g(iz,dz,A; wrap_periodic=false) for iz = 1:size(A,3)] == [-8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0]
            # (for Sxz)
            @test [x_g(ix,dx,Sxz) for ix = 1:size(Sxz,1)] ==   [2.0, 4.0, 6.0]
            #                       base grid (x dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Sxz) for iy = 1:size(Sxz,2)] == [1.0, 3.0, 5.0, 7.0]
            #                       base grid (y dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Sxz) for iz = 1:size(Sxz,3)] == [0.0, 4.0, 8.0]
            #                       base grid (z dim):  [8.0, 0.0, 4.0, 8.0, 0.0]
            @test [z_g(iz,dz,Sxz; wrap_periodic=false) for iz = 1:size(Sxz,3)] == [0.0, 4.0, 8.0]
        end;
        @testset "extents" begin
            @testset "default" begin
                @test extents() == (1:5, 1:5, 2:4)
                @test extents(P) == (1:5, 1:5, 2:4)
                @test extents(Vx) == (1:6, 1:5, 2:4)
                @test extents(Vz) == (1:5, 1:5, 2:4)
                @test extents(A) == (1:7, 1:5, 3:5)
                @test extents(Sxz) == (1:3, 1:4, 1:3)
            end;
            @testset "fix_global_boundaries" begin
                @test extents(; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(P; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(Vx; fix_global_boundaries=false) == (1:6, 1:5, 1:5)
                @test extents(Vz; fix_global_boundaries=false) == (1:5, 1:5, 1:6)
                @test extents(A; fix_global_boundaries=false) == (1:7, 1:5, 1:7)
                @test extents(Sxz; fix_global_boundaries=false) == (1:3, 1:4, 1:3)
            end;
            @testset "overlap" begin
                @test extents(0) == (1:5, 1:5, 2:4)
                @test extents(1) == (1:5, 1:5, 2:4)
                @test extents(P, 0) == (1:5, 1:5, 2:4)
                @test extents(Vx, 1) == (1:6, 1:5, 2:4) 
                @test extents(Vz, 0) == (1:5, 1:5, 2:4)
                @test extents(A, 1) == (1:7, 1:5, 3:5)
            end;
        end;
        @testset "extents_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            @testset "fix_global_boundaries" begin
                @test extents_g(; fix_global_boundaries=false) == (1:5, 1:5, 0:4)
                @test extents_g(P; fix_global_boundaries=false) == (1:5, 1:5, 0:4)
                @test extents_g(Vx; fix_global_boundaries=false) == (1:6, 1:5, 0:4)
                @test extents_g(Vz; fix_global_boundaries=false) == (1:5, 1:5, 0:5)
                @test extents_g(A; fix_global_boundaries=false) == (1:7, 1:5, -1:5)
                @test extents_g(Sxz; fix_global_boundaries=false) == (1:3, 1:4, 1:3)
            end;
            @testset "default" begin
                @test extents_g() == (1:5, 1:5, 1:3)
                @test extents_g(1) == (1:5, 1:5, 1:3)
                @test extents_g(P) == (1:5, 1:5, 1:3)
                @test extents_g(Vx) == (1:6, 1:5, 1:3)
                @test extents_g(Vz) == (1:5, 1:5, 1:3)
                @test extents_g(A) == (1:7, 1:5, 1:3)
                @test extents_g(Sxz) == (1:3, 1:4, 1:3)
            end;
            @testset "overlap" begin
                @test extents_g(0) == (1:5, 1:5, 1:3)
                @test extents_g(1) == (1:5, 1:5, 1:3)
                @test extents_g(P, 0) == (1:5, 1:5, 1:3)
                @test extents_g(Vx, 1) == (1:6, 1:5, 1:3)
                @test extents_g(Vz, 0) == (1:5, 1:5, 1:3)
                @test extents_g(A, 1) == (1:7, 1:5, 1:3)
            end;
            @testset "coordinates" begin
                @test extents_g(; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:4.0:8.0)
                @test extents_g(1; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:4.0:8.0)
                @test extents_g(P; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:4.0:8.0)
                @test extents_g(Vx; dxyz=(dx, dy, dz)) == (-1.0:2.0:9.0, 0.0:2.0:8.0, 0.0:4.0:8.0)
                @test extents_g(Vz; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, -2.0:4.0:6.0)
                @test extents_g(A; dxyz=(dx, dy, dz)) == (-2.0:2.0:10.0, 0.0:2.0:8.0, 0.0:4.0:8.0)
            end;
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "2. grid functions with non-default overlap" begin
        lx = 8;
        ly = 8;
        lz = 8;
        nx = 5;
        ny = 5;
        nz = 8;
        P   = zeros(nx,  ny,  nz  );
        Vx  = zeros(nx+1,ny,  nz  );
        Vz  = zeros(nx,  ny,  nz+1);
        A   = zeros(nx+2,ny,  nz+2);
        Sxz = zeros(nx-2,ny-1,nz-2);
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, origin=(0.0,-5.0,0.0), centery=true, overlaps=(3,2,3), quiet=true, init_MPI=false);
        @testset "nx_g / ny_g / nz_g" begin
            @test nx_g() == nx
            @test ny_g() == ny
            @test nz_g() == nz-3
            @test nx_g(P) == nx
            @test ny_g(P) == ny
            @test nz_g(P) == nz-3
            @test nx_g(Vx) == nx+1
            @test ny_g(Vx) == ny
            @test nz_g(Vx) == nz-3
            @test nx_g(Vz) == nx
            @test ny_g(Vz) == ny
            @test nz_g(Vz) == nz-3+1
            @test nx_g(A) == nx+2
            @test ny_g(A) == ny
            @test nz_g(A) == nz-3+2
            @test nx_g(Sxz) == nx-2
            @test ny_g(Sxz) == ny-1
            @test nz_g(Sxz) == nz-3-2
        end;
        @testset "ix_g / iy_g / iz_g" begin
            # (for P)
            @test [ix_g(ix, P) for ix = 1:size(P,1)] == [1, 2, 3, 4, 5]
            @test [iy_g(iy, P) for iy = 1:size(P,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, P) for iz = 1:size(P,3)] == [5, 1, 2, 3, 4, 5, 1, 2] #Alternative: [4, 5, 1, 2, 3, 4, 5, 1]
            @test [iz_g(iz, P; wrap_periodic=false) for iz = 1:size(P,3)] == [0, 1, 2, 3, 4, 5, 6, 7] #Alternative: [-1, 0, 1, 2, 3, 4, 5, 6] 
            # (for Vx)
            @test [ix_g(ix, Vx) for ix = 1:size(Vx,1)] == [1, 2, 3, 4, 5, 6]
            @test [iy_g(iy, Vx) for iy = 1:size(Vx,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, Vx) for iz = 1:size(Vx,3)] == [5, 1, 2, 3, 4, 5, 1, 2] #Alternative: [4, 5, 1, 2, 3, 4, 5, 1]
            @test [iz_g(iz, Vx; wrap_periodic=false) for iz = 1:size(Vx,3)] == [0, 1, 2, 3, 4, 5, 6, 7]#Alternative: [-1, 0, 1, 2, 3, 4, 5, 6]
            # (for Vz)
            @test [ix_g(ix, Vz) for ix = 1:size(Vz,1)] == [1, 2, 3, 4, 5]
            @test [iy_g(iy, Vz) for iy = 1:size(Vz,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, Vz) for iz = 1:size(Vz,3)] == [4, 5, 1, 2, 3, 4, 5, 1, 2]
            @test [iz_g(iz, Vz; wrap_periodic=false) for iz = 1:size(Vz,3)] == [-1, 0, 1, 2, 3, 4, 5, 6, 7]
            # (for A)
            @test [ix_g(ix, A) for ix = 1:size(A,1)] == [1, 2, 3, 4, 5, 6, 7]
            @test [iy_g(iy, A) for iy = 1:size(A,2)] == [1, 2, 3, 4, 5]
            @test [iz_g(iz, A) for iz = 1:size(A,3)] == [4, 5, 1, 2, 3, 4, 5, 1, 2, 3] #Alternative: [3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
            @test [iz_g(iz, A; wrap_periodic=false) for iz = 1:size(A,3)] == [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8] #Alternative: [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
            # (for Sxz)
            @test [ix_g(ix, Sxz) for ix = 1:size(Sxz,1)] == [1, 2, 3]
            @test [iy_g(iy, Sxz) for iy = 1:size(Sxz,2)] == [1, 2, 3, 4]
            @test [iz_g(iz, Sxz) for iz = 1:size(Sxz,3)] == [1, 2, 3, 4, 5, 1] #Alternative: [5, 1, 2, 3, 4, 5]
            @test [iz_g(iz, Sxz; wrap_periodic=false) for iz = 1:size(Sxz,3)] == [1, 2, 3, 4, 5, 6] #Alternative: [0, 1, 2, 3, 4, 5]
        end;
        @testset "x_g / y_g / z_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            # (for P)
            @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [-9.0, -7.0, -5.0, -3.0, -1.0]
            @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0] #Alternative: [6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
            @test [z_g(iz,dz,P; wrap_periodic=false) for iz = 1:size(P,3)] == [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] #Alternative: [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            # (for Vx)
            @test [x_g(ix,dx,Vx) for ix = 1:size(Vx,1)] == [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
            @test [y_g(iy,dy,Vx) for iy = 1:size(Vx,2)] == [-9.0, -7.0, -5.0, -3.0, -1.0]
            @test [z_g(iz,dz,Vx) for iz = 1:size(Vx,3)] == [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0] #Alternative: [6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
            @test [z_g(iz,dz,Vx; wrap_periodic=false) for iz = 1:size(Vx,3)] == [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] #Alternative: [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            # (for Vz)
            @test [x_g(ix,dx,Vz) for ix = 1:size(Vz,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Vz) for iy = 1:size(Vz,2)] == [-9.0, -7.0, -5.0, -3.0, -1.0]
            @test [z_g(iz,dz,Vz) for iz = 1:size(Vz,3)] == [5.0, 7.0, -1.0, 1.0, 3.0, 5.0, 7.0, -1.0, 1.0] #Alternative: [7.0, 9.0, 1.0, 3.0, 5.0, 7.0, 9.0, 1.0, 3.0]
            #                       base grid (z dim):       [8.0, 0.0, 2.0, 4.0, 6.0. 8.0, 0.0, 2.0]
            #                       possible alternative:  [7.0,-1.0, 1.0, 3.0, 5.0, 7.0,-1.0, 1.0, 3.0]
            @test [z_g(iz,dz,Vz; wrap_periodic=false) for iz = 1:size(Vz,3)] == [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0] #Alternative: [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
            # (for A)
            @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [-9.0, -7.0, -5.0, -3.0, -1.0]
            @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0, 4.0] #Alternative: [4.0, 6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0]
            #                       base grid (z dim):        [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0]
            @test [z_g(iz,dz,A; wrap_periodic=false) for iz = 1:size(A,3)] == [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0] #Alternative: [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
            # (for Sxz)
            @test [x_g(ix,dx,Sxz) for ix = 1:size(Sxz,1)] ==   [2.0, 4.0, 6.0]
            #                       base grid (x dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Sxz) for iy = 1:size(Sxz,2)] == [-8.0, -6.0, -4.0, -2.0]
            #                       base grid (y dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Sxz) for iz = 1:size(Sxz,3)] == [0.0, 2.0, 4.0, 6.0, 8.0, 0.0] #Alternative: [8.0, 0.0, 2.0, 4.0, 6.0, 8.0]
            #                       base grid (z dim):  [6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
            @test [z_g(iz,dz,Sxz; wrap_periodic=false) for iz = 1:size(Sxz,3)] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0] #Alternative: [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
        end;
        @testset "extents" begin
            @testset "default" begin
                @test extents() == (1:5, 1:5, 2:6)
                @test extents(P) == (1:5, 1:5, 2:6)
                @test extents(Vx) == (1:6, 1:5, 2:6)
                @test extents(Vz) == (1:5, 1:5, 3:7)
                @test extents(A) == (1:7, 1:5, 3:7)
                @test extents(Sxz) == (1:3, 1:4, 1:5)
            end;
            @testset "fix_global_boundaries" begin
                @test extents(; fix_global_boundaries=false) == (1:5, 1:5, 1:8)
                @test extents(P; fix_global_boundaries=false) == (1:5, 1:5, 1:8)
                @test extents(Vx; fix_global_boundaries=false) == (1:6, 1:5, 1:8)
                @test extents(Vz; fix_global_boundaries=false) == (1:5, 1:5, 1:9)
                @test extents(A; fix_global_boundaries=false) == (1:7, 1:5, 1:10)
                @test extents(Sxz; fix_global_boundaries=false) == (1:3, 1:4, 1:6)
            end;
            @testset "overlap" begin
                @test extents(0) == (1:5, 1:5, 2:6)
                @test extents(1) == (1:5, 1:5, 2:6)
                @test extents(2) == (1:5, 1:5, 2:6)
                @test extents(P, 0) == (1:5, 1:5, 2:6)
                @test extents(Vx, 1) == (1:6, 1:5, 2:6)
                @test extents(Vz, 1) == (1:5, 1:5, 3:7)
                @test extents(A, 2) == (1:7, 1:5, 3:7)
            end;
        end;
        @testset "extents_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            @testset "default" begin
                @test extents_g() == (1:5, 1:5, 1:5)
                @test extents_g(1) == (1:5, 1:5, 1:5)
                @test extents_g(P) == (1:5, 1:5, 1:5)
                @test extents_g(Vx) == (1:6, 1:5, 1:5)
                @test extents_g(Vz) == (1:5, 1:5, 1:5)
                @test extents_g(A) == (1:7, 1:5, 1:5)
                @test extents_g(Sxz) == (1:3, 1:4, 1:5)
            end;
            @testset "fix_global_boundaries" begin
                @test extents_g(; fix_global_boundaries=false) == (1:5, 1:5, 0:7)
                @test extents_g(P; fix_global_boundaries=false) == (1:5, 1:5, 0:7)
                @test extents_g(Vx; fix_global_boundaries=false) == (1:6, 1:5, 0:7)
                @test extents_g(Vz; fix_global_boundaries=false) == (1:5, 1:5, -1:7)
                @test extents_g(A; fix_global_boundaries=false) == (1:7, 1:5, -1:8)
                @test extents_g(Sxz; fix_global_boundaries=false) == (1:3, 1:4, 1:6)
            end;
            @testset "overlap" begin
                @test extents_g(0) == (1:5, 1:5, 1:5)
                @test extents_g(1) == (1:5, 1:5, 1:5)
                @test extents_g(2) == (1:5, 1:5, 1:5)
                @test extents_g(P, 0) == (1:5, 1:5, 1:5)
                @test extents_g(Vx, 1) == (1:6, 1:5, 1:5)
                @test extents_g(Vz, 1) == (1:5, 1:5, 1:5)
                @test extents_g(A, 2) == (1:7, 1:5, 1:5)
            end;
            @testset "coordinates" begin
                @test extents_g(; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, -9.0:2.0:-1.0, 0.0:2.0:8.0)
                @test extents_g(1; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, -9.0:2.0:-1.0, 0.0:2.0:8.0)
                @test extents_g(P; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, -9.0:2.0:-1.0, 0.0:2.0:8.0)
                @test extents_g(Vx; dxyz=(dx, dy, dz)) == (-1.0:2.0:9.0, -9.0:2.0:-1.0, 0.0:2.0:8.0)
                @test extents_g(Vz; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, -9.0:2.0:-1.0, -1.0:2.0:7.0)
                @test extents_g(A; dxyz=(dx, dy, dz)) == (-2.0:2.0:10.0, -9.0:2.0:-1.0, 0.0:2.0:8.0)
            end;
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "3. grid functions (simulated 3x3x3 processes)" begin
        lx = 20;
        ly = 20;
        lz = 16;
        nx = 5;
        ny = 5;
        nz = 5;
        P  = zeros(nx,  ny,  nz  );
        A  = zeros(nx+1,ny-2,nz+2);
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, quiet=true, init_MPI=false);
        # (Set dims, nprocs and nxyz_g in GG.global_grid().)
        dims      = [3,3,3];
        nxyz      = GG.global_grid().nxyz;
        periods   = GG.global_grid().periods;
        overlaps  = GG.global_grid().overlaps;
        nprocs    = prod(dims);
        nxyz_g    = dims.*(nxyz.-overlaps) .+ overlaps.*(periods.==0);
        GG.global_grid().dims   .= dims;
        GG.global_grid().nxyz_g .= nxyz_g;
        @testset "nx_g / ny_g / nz_g" begin
            @test nx_g() == nxyz_g[1]
            @test ny_g() == nxyz_g[2]
            @test nz_g() == nxyz_g[3]
            @test nx_g(P) == nxyz_g[1]
            @test ny_g(P) == nxyz_g[2]
            @test nz_g(P) == nxyz_g[3]
            @test nx_g(A) == nxyz_g[1]+1
            @test ny_g(A) == nxyz_g[2]-2
            @test nz_g(A) == nxyz_g[3]+2
        end;
        @testset "ix_g / iy_g / iz_g" begin
            # (for P)
            @coords(1)=0;  @test [ix_g(ix, P) for ix = 1:size(P,1)] == [1, 2, 3, 4, 5]
            @coords(1)=1;  @test [ix_g(ix, P) for ix = 1:size(P,1)] == [4, 5, 6, 7, 8]
            @coords(1)=2;  @test [ix_g(ix, P) for ix = 1:size(P,1)] == [7, 8, 9, 10, 11]
            @coords(2)=0;  @test [iy_g(iy, P) for iy = 1:size(P,2)] == [1, 2, 3, 4, 5]
            @coords(2)=1;  @test [iy_g(iy, P) for iy = 1:size(P,2)] == [4, 5, 6, 7, 8] 
            @coords(2)=2;  @test [iy_g(iy, P) for iy = 1:size(P,2)] == [7, 8, 9, 10, 11]
            @coords(3)=0;  @test [iz_g(iz, P) for iz = 1:size(P,3)] == [9, 1, 2, 3, 4]
            @coords(3)=0;  @test [iz_g(iz, P; wrap_periodic=false) for iz = 1:size(P,3)] == [0, 1, 2, 3, 4]
            @coords(3)=1;  @test [iz_g(iz, P) for iz = 1:size(P,3)] == [3, 4, 5, 6, 7]
            @coords(3)=1;  @test [iz_g(iz, P; wrap_periodic=false) for iz = 1:size(P,3)] == [3, 4, 5, 6, 7]
            @coords(3)=2;  @test [iz_g(iz, P) for iz = 1:size(P,3)] == [6, 7, 8, 9, 1]
            @coords(3)=2;  @test [iz_g(iz, P; wrap_periodic=false) for iz = 1:size(P,3)] == [6, 7, 8, 9, 10]
            # (for A)
            @coords(1)=0;  @test [ix_g(ix, A) for ix = 1:size(A,1)] == [1, 2, 3, 4, 5, 6]
            @coords(1)=1;  @test [ix_g(ix, A) for ix = 1:size(A,1)] == [4, 5, 6, 7, 8, 9]
            @coords(1)=2;  @test [ix_g(ix, A) for ix = 1:size(A,1)] == [7, 8, 9, 10, 11, 12]
            @coords(2)=0;  @test [iy_g(iy, A) for iy = 1:size(A,2)] == [1, 2, 3]
            @coords(2)=1;  @test [iy_g(iy, A) for iy = 1:size(A,2)] == [6, 7, 8]
            @coords(2)=2;  @test [iy_g(iy, A) for iy = 1:size(A,2)] == [11, 12, 13]
            @coords(3)=0;  @test [iz_g(iz, A) for iz = 1:size(A,3)] == [8, 9, 1, 2, 3, 4, 5]
            @coords(3)=0;  @test [iz_g(iz, A; wrap_periodic=false) for iz = 1:size(A,3)] == [-1, 0, 1, 2, 3, 4, 5]
            @coords(3)=1;  @test [iz_g(iz, A) for iz = 1:size(A,3)] == [2, 3, 4, 5, 6, 7, 8]
            @coords(3)=1;  @test [iz_g(iz, A; wrap_periodic=false) for iz = 1:size(A,3)] == [2, 3, 4, 5, 6, 7, 8]
            @coords(3)=2;  @test [iz_g(iz, A) for iz = 1:size(A,3)] == [5, 6, 7, 8, 9, 1, 2]
            @coords(3)=2;  @test [iz_g(iz, A; wrap_periodic=false) for iz = 1:size(A,3)] == [5, 6, 7, 8, 9, 10, 11]
        end;
        @testset "x_g / y_g / z_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            # (for P)
            @coords(1)=0;  @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @coords(1)=1;  @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [6.0, 8.0, 10.0, 12.0, 14.0]
            @coords(1)=2;  @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [12.0, 14.0, 16.0, 18.0, 20.0]
            @coords(2)=0;  @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @coords(2)=1;  @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [6.0, 8.0, 10.0, 12.0, 14.0]
            @coords(2)=2;  @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [12.0, 14.0, 16.0, 18.0, 20.0]
            @coords(3)=0;  @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [16.0, 0.0, 2.0, 4.0, 6.0]
            @coords(3)=0;  @test [z_g(iz,dz,P; wrap_periodic=false) for iz = 1:size(P,3)] == [-2.0, 0.0, 2.0, 4.0, 6.0]
            @coords(3)=1;  @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [4.0, 6.0, 8.0, 10.0, 12.0]
            @coords(3)=1;  @test [z_g(iz,dz,P; wrap_periodic=false) for iz = 1:size(P,3)] == [4.0, 6.0, 8.0, 10.0, 12.0]
            @coords(3)=2;  @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [10.0, 12.0, 14.0, 16.0, 0.0]
            @coords(3)=2;  @test [z_g(iz,dz,P; wrap_periodic=false) for iz = 1:size(P,3)] == [10.0, 12.0, 14.0, 16.0, 18.0]
            # (for A)
            @coords(1)=0;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
            @coords(1)=1;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
            @coords(1)=2;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
            @coords(2)=0;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [2.0, 4.0, 6.0]
            @coords(2)=1;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [8.0, 10.0, 12.0]
            @coords(2)=2;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [14.0, 16.0, 18.0]
            @coords(3)=0;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [14.0, 16.0, 0.0, 2.0, 4.0, 6.0, 8.0]
            @coords(3)=0;  @test [z_g(iz,dz,A; wrap_periodic=false) for iz = 1:size(A,3)] == [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
            @coords(3)=1;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
            @coords(3)=1;  @test [z_g(iz,dz,A; wrap_periodic=false) for iz = 1:size(A,3)] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
            @coords(3)=2;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [8.0, 10.0, 12.0, 14.0, 16.0, 0.0, 2.0]
            @coords(3)=2;  @test [z_g(iz,dz,A; wrap_periodic=false) for iz = 1:size(A,3)] == [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        end;
        @testset "extents" begin
            @testset "default" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents() == (1:5, 1:5, 2:5)
                @test extents(P) == (1:5, 1:5, 2:5)
                @test extents(A) == (1:6, 1:3, 3:7)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents() == (1:5, 1:5, 1:5)
                @test extents(P) == (1:5, 1:5, 1:5)
                @test extents(A) == (1:6, 1:3, 1:7)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents() == (1:5, 1:5, 1:4)
                @test extents(P) == (1:5, 1:5, 1:4)
                @test extents(A) == (1:6, 1:3, 1:5)
            end;
            @testset "fix_global_boundaries" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents(; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(P; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(A; fix_global_boundaries=false) == (1:6, 1:3, 1:7)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents(; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(P; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(A; fix_global_boundaries=false) == (1:6, 1:3, 1:7)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents(; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(P; fix_global_boundaries=false) == (1:5, 1:5, 1:5)
                @test extents(A; fix_global_boundaries=false) == (1:6, 1:3, 1:7)
            end;
            @testset "overlap" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents(0) == (1:4, 1:4, 2:4)
                @test extents(1) == (1:4, 1:4, 2:4)
                @test extents(P, 0) == (1:4, 1:4, 2:4)
                @test extents(A, 0) == (1:4, 1:3, 3:5)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents(0) == (2:4, 2:4, 2:4)
                @test extents(1) == (1:4, 1:4, 1:4)
                @test extents(P, 0) == (2:4, 2:4, 2:4)
                @test extents(A, 0) == (2:4, 1:3, 3:5)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents(0) == (2:5, 2:5, 2:4)
                @test extents(1) == (1:5, 1:5, 1:4)
                @test extents(P, 0) == (2:5, 2:5, 2:4)
                @test extents(A, 0) == (2:6, 1:3, 3:5)
            end;
        end;
        @testset "extents_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            @testset "default" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents_g() == (1:5, 1:5, 1:4)
                @test extents_g(P) == (1:5, 1:5, 1:4)
                @test extents_g(A) == (1:6, 1:3, 1:5)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents_g() == (4:8, 4:8, 3:7)
                @test extents_g(P) == (4:8, 4:8, 3:7)
                @test extents_g(A) == (3:8, 6:8, 2:8)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents_g() == (7:11, 7:11, 6:9)
                @test extents_g(P) == (7:11, 7:11, 6:9)
                @test extents_g(A) == (5:10, 11:13, 5:9)
            end;
            @testset "fix_global_boundaries" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents_g(; fix_global_boundaries=false) == (1:5, 1:5, 0:4)
                @test extents_g(P; fix_global_boundaries=false) == (1:5, 1:5, 0:4)
                @test extents_g(A; fix_global_boundaries=false) == (1:6, 1:3, -1:5)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents_g(; fix_global_boundaries=false) == (4:8, 4:8, 3:7)
                @test extents_g(P; fix_global_boundaries=false) == (4:8, 4:8, 3:7)
                @test extents_g(A; fix_global_boundaries=false) == (3:8, 6:8, 2:8)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents_g(; fix_global_boundaries=false) == (7:11, 7:11, 6:10)
                @test extents_g(P; fix_global_boundaries=false) == (7:11, 7:11, 6:10)
                @test extents_g(A; fix_global_boundaries=false) == (5:10, 11:13, 5:11)
            end;
            @testset "overlap" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents_g(0) == (1:4, 1:4, 1:3)
                @test extents_g(1) == (1:4, 1:4, 1:3)
                @test extents_g(P, 0) == (1:4, 1:4, 1:3)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents_g(0) == (5:7, 5:7, 4:6)
                @test extents_g(1) == (4:7, 4:7, 3:6)
                @test extents_g(P, 0) == (5:7, 5:7, 4:6)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents_g(0) == (8:11, 8:11, 7:9)
                @test extents_g(1) == (7:11, 7:11, 6:9)
                @test extents_g(P, 0) == (8:11, 8:11, 7:9)
            end;
            @testset "coordinates" begin
                @coords(1)=0; @coords(2)=0; @coords(3)=0;
                @test extents_g(; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:2.0:6.0)
                @test extents_g(1; dxyz=(dx, dy, dz)) == (0.0:2.0:6.0, 0.0:2.0:6.0, 0.0:2.0:4.0)
                @test extents_g(P; dxyz=(dx, dy, dz)) == (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:2.0:6.0)
                @test extents_g(A; dxyz=(dx, dy, dz)) == (-1.0:2.0:9.0, 2.0:2.0:6.0, 0.0:2.0:8.0)
                
                @coords(1)=1; @coords(2)=1; @coords(3)=1;
                @test extents_g(; dxyz=(dx, dy, dz)) == (6.0:2.0:14.0, 6.0:2.0:14.0, 4.0:2.0:12.0)
                @test extents_g(1; dxyz=(dx, dy, dz)) == (6.0:2.0:12.0, 6.0:2.0:12.0, 4.0:2.0:10.0)
                @test extents_g(P; dxyz=(dx, dy, dz)) == (6.0:2.0:14.0, 6.0:2.0:14.0, 4.0:2.0:12.0)
                @test extents_g(A; dxyz=(dx, dy, dz)) == (5.0:2.0:15.0, 8.0:2.0:12.0, 2.0:2.0:14.0)
                
                @coords(1)=2; @coords(2)=2; @coords(3)=2;
                @test extents_g(; dxyz=(dx, dy, dz)) == (12.0:2.0:20.0, 12.0:2.0:20.0, 10.0:2.0:16.0)
                @test extents_g(1; dxyz=(dx, dy, dz)) == (12.0:2.0:20.0, 12.0:2.0:20.0, 10.0:2.0:16.0)
                @test extents_g(P; dxyz=(dx, dy, dz)) == (12.0:2.0:20.0, 12.0:2.0:20.0, 10.0:2.0:16.0)
                @test extents_g(A; dxyz=(dx, dy, dz)) == (11.0:2.0:21.0, 14.0:2.0:18.0, 8.0:2.0:16.0)
            end;
        end;
        finalize_global_grid(finalize_MPI=false);
    end;
end;

## Test tear down
MPI.Finalize()
