push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require


## Test setup (NOTE: Testset "2. initialization including MPI" completes the test setup as it initializes MPI and must therefore mandatorily be at the 2nd position). NOTE: these tests require nprocs == 1.
p0 = MPI.PROC_NULL
nx = 4;
ny = 4;
nz = 1;

@testset "$(basename(@__FILE__))" begin
    @testset "1. pre-MPI_Init-exception" begin
        @require !GG.grid_is_initialized()
        @test_throws ErrorException init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);  # Error: init_MPI=false while MPI has not been initialized before.
        @test !GG.grid_is_initialized()
    end;

    @testset "2. initialization including MPI" begin
        me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, quiet=true);
        @testset "initialized" begin
            @test GG.grid_is_initialized()
            @test MPI.Initialized()
        end;
        @testset "return values" begin
            @test me     == 0
            @test dims   == [1, 1, 1]
            @test nprocs == 1
            @test coords == [0, 0, 0]
            @test typeof(comm_cart) == MPI.Comm
        end;
        @testset "values in global grid" begin
            @test GG.global_grid().nxyz_g    == [nx, ny, nz]
            @test GG.global_grid().nxyz      == [nx, ny, nz]
            @test GG.global_grid().dims      == dims
            @test GG.global_grid().overlaps  == [2, 2, 2]
            @test GG.global_grid().halowidths== [1, 1, 1]
            @test GG.global_grid().nprocs    == nprocs
            @test GG.global_grid().me        == me
            @test GG.global_grid().coords    == coords
            @test GG.global_grid().neighbors == [p0 p0 p0; p0 p0 p0]
            @test GG.global_grid().periods   == [0, 0, 0]
            @test GG.global_grid().disp      == 1
            @test GG.global_grid().reorder   == 1
            @test GG.global_grid().comm      == comm_cart
            @test GG.global_grid().quiet     == true
            @test GG.global_grid().origin    == [0.0, 0.0, 0.0]
            @test GG.global_grid().origin_on_vertex == false
            @test GG.global_grid().centerxyz == [false, false, false]
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "3. initialization with pre-initialized MPI" begin
        @require MPI.Initialized()
        @require !GG.grid_is_initialized()
        init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);
        @test GG.grid_is_initialized()
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "4. initialization with periodic boundaries" begin
        nz=4;
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodx=1, periodz=1, quiet=true, init_MPI=false);
        @testset "initialized" begin
            @test GG.grid_is_initialized()
        end;
        @testset "values in global grid" begin # (Checks only what is different than in the basic test.)
            @test GG.global_grid().nxyz_g    == [nx-2, ny, nz-2]
            @test GG.global_grid().nxyz      == [nx,   ny, nz  ]
            @test GG.global_grid().neighbors == [0 p0 0; 0 p0 0]
            @test GG.global_grid().periods   == [1, 0, 1]
        end
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "5. initialization with non-default overlaps and one periodic boundary" begin
        nz  = 10;
        olx = 3;
        oly = 0;
        olz = 4;
        init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, overlaps=(olx, oly, olz), quiet=true, init_MPI=false);
        @testset "initialized" begin
            @test GG.grid_is_initialized()
        end
        @testset "values in global grid" begin # (Checks only what is different than in the basic test.)
            @test GG.global_grid().nxyz_g    == [nx, ny, nz-olz]  # Note: olx has no effect as there is only 1 process and this boundary is not periodic.
            @test GG.global_grid().nxyz      == [nx, ny, nz    ]
            @test GG.global_grid().overlaps  == [olx, oly, olz]
            @test GG.global_grid().halowidths== [1, 1, 2]
            @test GG.global_grid().neighbors == [p0 p0 0; p0 p0 0]
            @test GG.global_grid().periods   == [0, 0, 1]
        end;
        finalize_global_grid(finalize_MPI=false);
    end;

    @testset "6. post-MPI_Init-exceptions" begin
        @require MPI.Initialized()
        @require !GG.grid_is_initialized()
        nx = 4;
        ny = 4;
        nz = 4;
        @test_throws ErrorException init_global_grid(1, ny, nz, quiet=true, init_MPI=false);                         # Error: nx==1.
        @test_throws ErrorException init_global_grid(nx, 1, nz, quiet=true, init_MPI=false);                         # Error: ny==1, while nz>1.
        @test_throws ErrorException init_global_grid(nx, ny, 1, dimz=3, quiet=true, init_MPI=false);                 # Error: dimz>1 while nz==1.
        @test_throws ErrorException init_global_grid(nx, ny, 1, periodz=1, quiet=true, init_MPI=false);              # Error: periodz==1 while nz==1.
        @test_throws ErrorException init_global_grid(nx, ny, nz, periody=1, overlaps=(2,3,2), quiet=true, init_MPI=false); # Error: periody==1 while ny<2*overlaps[2]-1 (4<5).
        @test_throws ErrorException init_global_grid(nx, ny, nz, halowidths=(1,0,1), quiet=true, init_MPI=false);    # Error: halowidths[2]<1.
        @test_throws ErrorException init_global_grid(nx, ny, nz, overlaps=(4,3,2), halowidths=(2,2,1), quiet=true, init_MPI=false); # Error: halowidths[2]==2 while overlaps[2]==3.
        @test_throws ErrorException init_global_grid(nx, ny, nz, origin=(0.0, 0.0, 0.0, 0.0), quiet=true, init_MPI=false); # Error: origin length > 3
        @test_throws ErrorException init_global_grid(5, ny, nz, centerx=true, origin_on_vertex=true, quiet=true, init_MPI=false); # Error: centerx && origin_on_vertex && nx odd
        @test_throws ErrorException init_global_grid(nx, 5, nz, centery=true, origin_on_vertex=true, quiet=true, init_MPI=false); # Error: centery && origin_on_vertex && ny odd
        @test_throws ErrorException init_global_grid(nx, ny, 5, centerz=true, origin_on_vertex=true, quiet=true, init_MPI=false); # Error: centerz && origin_on_vertex && nz odd
        @test_throws ErrorException init_global_grid(4, ny, nz, centerx=true, origin_on_vertex=false, quiet=true, init_MPI=false); # Error: centerx && !origin_on_vertex && nx even
        @test_throws ErrorException init_global_grid(nx, 4, nz, centery=true, origin_on_vertex=false, quiet=true, init_MPI=false); # Error: centery && !origin_on_vertex && ny even
        @test_throws ErrorException init_global_grid(nx, ny, 4, centerz=true, origin_on_vertex=false, quiet=true, init_MPI=false); # Error: centerz && !origin_on_vertex && nz even
        @test_throws ErrorException init_global_grid(nx, ny, nz, quiet=true);                                        # Error: MPI already initialized
        @testset "already initialized exception" begin
            init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);
            @require GG.grid_is_initialized()
            @test_throws ErrorException init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);  # Error: IGG already initialised
            finalize_global_grid(finalize_MPI=false);
        end;
    end;
end;

## Test tear down
MPI.Finalize()
