# NOTE: All tests of this file can be run with any number of processes.
push!(LOAD_PATH, "../src")
using Test
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import MPI
using CUDA
using AMDGPU
import ImplicitGlobalGrid: @require
import ImplicitGlobalGrid: AMDGPU_functional #NOTE: this is to be replaced with AMDGPU.functional() once available.

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU_functional()


## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. select_device" begin
        @static if test_cuda
            me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="CUDA");
            gpu_id = select_device();
            @test gpu_id < length(CUDA.devices())
            finalize_global_grid(finalize_MPI=false);
        end
        @static if test_amdgpu
            me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="AMDGPU");
            gpu_id = select_device();
            @test gpu_id < length(AMDGPU.get_agents(:gpu))
            finalize_global_grid(finalize_MPI=false);
        end
        @static if !(test_cuda || test_amdgpu)
            me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false);
            @test_throws ErrorException select_device()
            finalize_global_grid(finalize_MPI=false);
        end
    end;
end;

## Test tear down
MPI.Finalize()
