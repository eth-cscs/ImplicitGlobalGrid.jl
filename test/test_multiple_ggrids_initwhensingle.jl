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
    init_global_grid(nx, ny, nz, dimx=1, dimy=1, dimz=1, quiet=true, init_MPI=true)

    @testset "initialized" begin
        @test GG.grid_is_initialized()
        @test MPI.Initialized()
    end

    @testset "multiple instance cannot be called if single instance was called before" begin
        @test_throws ErrorException init_global_grid_instance(nx, ny, nz, dimx=1, dimy=1, dimz=1, periodx=1, periodz=1, quiet=true, init_MPI=false)
    end

    finalize_global_grid(finalize_MPI=false)

end;

## Test tear down
MPI.Finalize()
