# NOTE: All tests of this file can be run with any number of processes.
push!(LOAD_PATH, "../src")
using Test
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import MPI
import ImplicitGlobalGrid: @require

test_gpu = GG.ENABLE_CUDA
if test_gpu
	using CUDA
	@assert CUDA.functional(true)
end

## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD);

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. select_device" begin
        me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false);
        @static if test_gpu
            gpu_id = select_device();
            @test gpu_id < length(CUDA.devices())
        else
            @test_throws ErrorException select_device()
        end
        finalize_global_grid(finalize_MPI=false);
    end;
end;

## Test tear down
MPI.Finalize()
