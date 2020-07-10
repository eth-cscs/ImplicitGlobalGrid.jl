push!(LOAD_PATH, "../src")
using Test
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import MPI
import ImplicitGlobalGrid: @require
macro coords(i) :(GG.global_grid().coords[$i]) end


## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD);
@require nprocs == 1  # This test requires nprocs to be 1.

@testset "$(basename(@__FILE__))" begin
    @testset "1. *_g functions" begin
        lx = 8;
        ly = 8;
        lz = 8;
        nx = 5;
        ny = 5;
        nz = 5;
        P   = zeros(nx,  ny,  nz  );
        Vx  = zeros(nx+1,ny,  nz  );
        Vz  = zeros(nx,  ny,  nz+1);
        A   = zeros(nx,  ny,  nz+2);
        Sxz = zeros(nx-2,ny-1,nz-2);
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, quiet=true, init_MPI=false);
        @testset "nx_g / ny_g / nz_g" begin
            @test nx_g() == nx
            @test ny_g() == ny
            @test nz_g() == nz-2
        end;
        @testset "x_g / y_g / z_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            # (for P)
            @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [8.0, 0.0, 4.0, 8.0, 0.0]
            # (for Vx)
            @test [x_g(ix,dx,Vx) for ix = 1:size(Vx,1)] == [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
            @test [y_g(iy,dy,Vx) for iy = 1:size(Vx,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Vx) for iz = 1:size(Vx,3)] == [8.0, 0.0, 4.0, 8.0, 0.0]
            # (for Vz)
            @test [x_g(ix,dx,Vz) for ix = 1:size(Vz,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Vz) for iy = 1:size(Vz,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Vz) for iz = 1:size(Vz,3)] == [ 6.0, 10.0,  2.0,  6.0, 10.0,  2.0]
            #                       base grid (z dim):        [ 8.0,  0.0,  4.0,  8.0,  0.0]
            #                       possible alternative:  [ 6.0, -2.0,  2.0,  6.0, -2.0,  2.0]  # This would be a possible alternative way to define {x,y,z}_g; however, we decided that the grid should start at 0.0 in this case and the overlap be at the end (we avoid completely any negative z_g).
            #                       wrong:                 [ 6.0, -2.0,  2.0,  6.0, 10.0,  2.0]  # The 2nd and the 2nd-last cell must be the same due to the overlap of 3.
            # (for A)
            @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [4.0, 8.0, 0.0, 4.0, 8.0, 0.0, 4.0]
            #                       base grid (z dim):       [ 8.0, 0.0, 4.0, 8.0, 0.0]
            # (for Sxz)
            @test [x_g(ix,dx,Sxz) for ix = 1:size(Sxz,1)] ==   [2.0, 4.0, 6.0]
            #                       base grid (x dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Sxz) for iy = 1:size(Sxz,2)] == [1.0, 3.0, 5.0, 7.0]
            #                       base grid (y dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Sxz) for iz = 1:size(Sxz,3)] ==   [0.0, 4.0, 8.0]
            #                       base grid (z dim):   [ 8.0, 0.0, 4.0, 8.0, 0.0]
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "2. *_g functions with non-default overlap" begin
        lx = 8;
        ly = 8;
        lz = 8;
        nx = 5;
        ny = 5;
        nz = 8;
        P   = zeros(nx,  ny,  nz  );
        Vx  = zeros(nx+1,ny,  nz  );
        Vz  = zeros(nx,  ny,  nz+1);
        A   = zeros(nx,  ny,  nz+2);
        Sxz = zeros(nx-2,ny-1,nz-2);
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, overlapx=3, overlapz=3, quiet=true, init_MPI=false);
        @testset "nx_g / ny_g / nz_g" begin
            @test nx_g() == nx
            @test ny_g() == ny
            @test nz_g() == nz-3
        end;
        @testset "x_g / y_g / z_g" begin
            dx  = lx/(nx_g()-1);
            dy  = ly/(ny_g()-1);
            dz  = lz/(nz_g()-1);
            # (for P)
            @test [x_g(ix,dx,P) for ix = 1:size(P,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]          # (same as in the first test)
            @test [y_g(iy,dy,P) for iy = 1:size(P,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]          # (same as in the first test)
            @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0]
            # (for Vz)
            @test [x_g(ix,dx,Vz) for ix = 1:size(Vz,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]        # (same as in the first test)
            @test [y_g(iy,dy,Vz) for iy = 1:size(Vz,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]        # (same as in the first test)
            @test [z_g(iz,dz,Vz) for iz = 1:size(Vz,3)] == [7.0, 9.0, 1.0, 3.0, 5.0, 7.0, 9.0, 1.0, 3.0]
            #                       base grid (z dim):       [8.0, 0.0, 2.0, 4.0, 6.0. 8.0, 0.0, 2.0]
            #                       possible alternative:  [7.0,-1.0, 1.0, 3.0, 5.0, 7.0,-1.0, 1.0, 3.0]
            # (for A)
            @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [0.0, 2.0, 4.0, 6.0, 8.0]          # (same as in the first test)
            @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [0.0, 2.0, 4.0, 6.0, 8.0]          # (same as in the first test)
            @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0, 4.0]
            #                       base grid (z dim):        [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0]
            # (for Sxz)
            @test [x_g(ix,dx,Sxz) for ix = 1:size(Sxz,1)] ==   [2.0, 4.0, 6.0]         # (same as in the first test)
            #                       base grid (x dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [y_g(iy,dy,Sxz) for iy = 1:size(Sxz,2)] == [1.0, 3.0, 5.0, 7.0]       # (same as in the first test)
            #                       base grid (y dim):    [0.0, 2.0, 4.0, 6.0, 8.0]
            @test [z_g(iz,dz,Sxz) for iz = 1:size(Sxz,3)] ==   [0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
            #                       base grid (z dim):    [8.0, 0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0]
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "3. *_g functions (simulated 3x3x3 processes)" begin
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
            @coords(3)=1;  @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [4.0, 6.0, 8.0, 10.0, 12.0]
            @coords(3)=2;  @test [z_g(iz,dz,P) for iz = 1:size(P,3)] == [10.0, 12.0, 14.0, 16.0, 0.0]
            # (for A)
            @coords(1)=0;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
            @coords(1)=1;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
            @coords(1)=2;  @test [x_g(ix,dx,A) for ix = 1:size(A,1)] == [11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
            @coords(2)=0;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [2.0, 4.0, 6.0]
            @coords(2)=1;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [8.0, 10.0, 12.0]
            @coords(2)=2;  @test [y_g(iy,dy,A) for iy = 1:size(A,2)] == [14.0, 16.0, 18.0]
            @coords(3)=0;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [14.0, 16.0, 0.0, 2.0, 4.0, 6.0, 8.0]
            @coords(3)=1;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
            @coords(3)=2;  @test [z_g(iz,dz,A) for iz = 1:size(A,3)] == [8.0, 10.0, 12.0, 14.0, 16.0, 0.0, 2.0]
        end;
        finalize_global_grid(finalize_MPI=false);
    end;
end;

## Test tear down
MPI.Finalize()
