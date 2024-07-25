push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require


## Test setup (NOTE: Testset "2. initialization including MPI" completes the test setup as it initializes MPI and must therefore mandatorily be at the 2nd position). NOTE: these tests require nprocs == 1.
p0 = MPI.PROC_NULL
nx = 40;
ny = 40;
nz = 10;

@testset "$(basename(@__FILE__))" begin
    @testset "1. pre-MPI_Init-exception" begin
        @require !GG.grid_is_initialized()
        @test_throws ErrorException init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);  # Error: init_MPI=false while MPI has not been initialized before.
        @test !GG.grid_is_initialized()
    end;

    @testset "2. initialization including MPI" begin
	      # ATTENTION TO THE INIT_MPI PARAMETER THE DEFAULT IS FALSE IN THIS INIT
        global_grid_A = init_global_grid_instance(nx, ny, nz, dimx=1, dimy=1, dimz=1, quiet=true, init_MPI=true);

        @testset "NOT initialized" begin
            @test !GG.grid_is_initialized()
        end

        nullgrid = switch(global_grid_A)

        @testset "get switched grid (NULL GRID) from switch" begin
            @test nullgrid.nprocs <= 0
        end

        @testset "initialized" begin
            @test GG.grid_is_initialized()
            @test MPI.Initialized()
        end;

        @testset "values in global grid (A)" begin
            @test GG.global_grid().nxyz_g    == [nx, ny, nz]
            @test GG.global_grid().nxyz      == [nx, ny, nz]
            @test GG.global_grid().overlaps  == [2, 2, 2]
            @test GG.global_grid().halowidths== [1, 1, 1]
            @test GG.global_grid().neighbors == [p0 p0 p0; p0 p0 p0]
            @test GG.global_grid().periods   == [0, 0, 0]
            @test GG.global_grid().disp      == 1
            @test GG.global_grid().reorder   == 1
            @test GG.global_grid().quiet     == true
        end

        global_grid_B = init_global_grid_instance(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodx=1, periodz=1, quiet=true, init_MPI=false)
        switch(global_grid_B)

        @testset "values in global grid (B)" begin
            @test GG.global_grid().periods   == [1, 0, 1]
        end
        finalize_global_grid(finalize_MPI=false);

        @testset "single instance cannot be called if multiple instance was called before" begin
	          @test_throws ErrorException init_global_grid(nx, ny, nz, init_MPI = false)
        end
    end;

end;

## Test tear down
MPI.Finalize()
