# NOTE: All tests of this file can be run with any number of processes.
push!(LOAD_PATH, "../src")
using Test
import MPI
using CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. select_device" begin
        @static if test_cuda && !test_amdgpu
            @testset "\"CUDA\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="CUDA");
                gpu_id = select_device();
                @test gpu_id < length(CUDA.devices())
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "\"auto\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="auto");
                gpu_id = select_device();
                @test gpu_id < length(CUDA.devices())
                finalize_global_grid(finalize_MPI=false);
            end;
        end
        @static if test_amdgpu && !test_cuda
            @testset "\"AMDGPU\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="AMDGPU");
                gpu_id = select_device();
                @test gpu_id <= length(AMDGPU.devices())
                finalize_global_grid(finalize_MPI=false);
            end;
            @testset "\"auto\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="auto");
                gpu_id = select_device();
                @test gpu_id <= length(AMDGPU.devices())
                finalize_global_grid(finalize_MPI=false);
            end;
        end
        @static if !(test_cuda || test_amdgpu) || (test_cuda && test_amdgpu)
            @testset "\"auto\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="auto");
                @test_throws ErrorException select_device()
                finalize_global_grid(finalize_MPI=false);
            end;
        end
        @static if !test_cuda
            @testset "\"CUDA\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="CUDA");
                @test_throws ErrorException select_device()
                finalize_global_grid(finalize_MPI=false);
            end;
        end
        @static if !test_amdgpu
            @testset "\"AMDGPU\"" begin
                me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="AMDGPU");
                @test_throws ErrorException select_device()
                finalize_global_grid(finalize_MPI=false);
            end;
        end
        @testset "\"none\"" begin
            me, = init_global_grid(3, 4, 5; quiet=true, init_MPI=false, device_type="none");
            @test_throws ErrorException select_device()
            finalize_global_grid(finalize_MPI=false);
        end
    end;
end;

## Test tear down
MPI.Finalize()
