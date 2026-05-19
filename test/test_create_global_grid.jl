push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid;
GG = ImplicitGlobalGrid;
import ImplicitGlobalGrid: @require


## Test setup (NOTE: Testset "2. initialization including MPI" completes the test setup as it initializes MPI and must therefore mandatorily be at the 2nd position). NOTE: these tests require nprocs == 1.
p0 = MPI.PROC_NULL
nx = 4;
ny = 4;
nz = 1;

@testset "$(basename(@__FILE__))" begin
    @testset "1. grid creation including MPI" begin
        gg = nothing
        @testset "basic initialization and create" begin
            # No tests, just a more specific testset description in case of error
            init_global_grid(save_kwarg_defaults=true, quiet=true, init_MPI=true, device_type=GG.DEVICE_TYPE_NONE, select_device=false, dimx=1, dimy=1, dimz=1, periodx=1, periody=1, periodz=1, origin=(0.1, 0.1, 0.1), origin_on_vertex=false, centerx=true, centery=true, centerz=true, overlaps=(4, 4, 4), halowidths=(2, 2, 2), disp=0, reorder=0)
            gg = create_global_grid(31, 31, 31)

        end
        @testset "return values" begin
            @test GG.grid_is_initialized() == false

            @test typeof(gg) == GG.GlobalGrid
            @test gg.me == 0
            @test gg.dims == [1, 1, 1]
            @test gg.nprocs == 1
            @test gg.coords == [0, 0, 0]
            @test typeof(gg.comm) == MPI.Comm
            @test gg.nxyz_g == [27, 27, 27]
            @test gg.nxyz == [31, 31, 31]
            @test gg.overlaps == [4, 4, 4]
            @test gg.halowidths == [2, 2, 2]
            @test gg.neighbors == [0 0 0; 0 0 0]
            @test gg.periods == [1, 1, 1]
            @test gg.disp == 0
            @test gg.reorder == 0
            @test gg.quiet == true
            @test gg.origin == [0.1, 0.1, 0.1]
            @test gg.origin_on_vertex == false
            @test gg.centerxyz == [true, true, true]
            @test gg.cuda_enabled == false
            @test gg.amdgpu_enabled == false
        end

        finalize_global_grid(finalize_MPI=false)
    end

    @testset "2. initialization with non-default overlaps and one periodic boundary" begin

        init_global_grid(init_MPI=false)

        nz = 10
        olx = 3
        oly = 0
        olz = 4
        gg = create_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodz=1, overlaps=(olx, oly, olz), halowidths=(1, 1, 2), quiet=true)
        @testset "values in global grid" begin # (Checks only what is different than in the basic test.)
            @test gg.nxyz_g == [nx, ny, nz - olz]  # Note: olx has no effect as there is only 1 process and this boundary is not periodic.
            @test gg.nxyz == [nx, ny, nz]
            @test gg.overlaps == [olx, oly, olz]
            @test gg.halowidths == [1, 1, 2]
            @test gg.neighbors == [p0 p0 0; p0 p0 0]
            @test gg.periods == [0, 0, 1]
        end
        finalize_global_grid(finalize_MPI=false)
    end

    @testset "3. setting default parameters on initialization and grid creation" begin
        init_global_grid(save_kwarg_defaults=true, quiet=true, init_MPI=false, periodx=1, periody=1)

        gg = create_global_grid(nx, ny, nz, quiet=true)
        @testset "values in global grid" begin # Checks if new defaults spread
            @test gg.periods == [1, 1, 0]
        end

        finalize_global_grid(finalize_MPI=false)
    end

    @testset "4. post-MPI_Init-exceptions" begin

        nx = 4
        ny = 4
        nz = 4

        @test_throws ErrorException create_global_grid(nx, ny, nz)

        init_global_grid(save_kwarg_defaults=true, quiet=true, init_MPI=false, periodx=1, periody=1)
        @require MPI.Initialized()
        @require !GG.grid_is_initialized()

        @test_throws ErrorException create_global_grid(1, ny, nz, quiet=true)                         # Error: nx==1.
        @test_throws ErrorException create_global_grid(nx, 1, nz, quiet=true)                         # Error: ny==1, while nz>1.
        @test_throws ErrorException create_global_grid(nx, ny, 1, dimz=3, quiet=true)                 # Error: dimz>1 while nz==1.
        @test_throws ErrorException create_global_grid(nx, ny, 1, periodz=1, quiet=true)              # Error: periodz==1 while nz==1.
        @test_throws ErrorException create_global_grid(nx, ny, nz, periody=1, overlaps=(2, 3, 2), quiet=true) # Error: periody==1 while ny<2*overlaps[2]-1 (4<5).
        @test_throws ErrorException create_global_grid(nx, ny, nz, halowidths=(1, 0, 1), quiet=true)    # Error: halowidths[2]<1.
        @test_throws ErrorException create_global_grid(nx, ny, nz, overlaps=(4, 3, 2), halowidths=(2, 2, 1), quiet=true) # Error: halowidths[2]==2 while overlaps[2]==3.
        @test_throws ErrorException create_global_grid(nx, ny, nz, origin=(0.0, 0.0, 0.0, 0.0), quiet=true) # Error: origin length > 3
        @test_throws ErrorException create_global_grid(5, ny, nz, centerx=true, origin_on_vertex=true, quiet=true) # Error: centerx && origin_on_vertex && nx odd

        finalize_global_grid(finalize_MPI=false)
    end
end;

## Test tear down
MPI.Finalize()
