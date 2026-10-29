# NOTE: All tests of this file can be run with any number of processes.
# For GPU arrays, update_halo! runs on streams of its own and does not synchronize with work enqueued before the call: the caller
# synchronizes that work before the call, and update_halo! returns once the halo update is complete (see the docstring of update_halo!
# and the section "Asynchronous halo updates and overlap with computation" of the documentation). These tests check that under this
# contract the halo update is correct on every exchange path (dimension 1: pack/unpack kernels; dimensions 2 and 3: 3-D memory copies;
# with and without GPU-aware MPI; local exchange (periodic dimension with one process) or MPI), i.e. that all operations of update_halo!
# are correctly ordered among each other. Unlike in test_update_halo.jl, the arrays and halos are large and the preceding kernel is long,
# so that a missing ordering would show.
# With one process, the halo update is local in all dimensions and needs no MPI: the GPU-aware variant is then always tested too. With
# multiple processes, the GPU-aware MPI setting of the environment is used (IGG_CUDAAWARE_MPI / IGG_ROCMAWARE_MPI).

using Test
import MPI, Polyester  # NOTE: Polyester must be loaded for the CPU halo update when IGG_USE_POLYESTER=1 (as in test_update_halo.jl).
using CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

# A deliberately slow kernel writing the value v to all elements of A (on the task-local stream).
function fill_slowly_cuda!(A, v, niter)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i <= length(A)
        x = Float64(i)
        for _ = 1:niter x = muladd(x, 0.999, 1e-3); end
        @inbounds A[i] = isnan(x) ? -one(v) : v
    end
    return nothing
end
function fill_slowly_amdgpu!(A, v, niter)
    i = (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
    if i <= length(A)
        x = Float64(i)
        for _ = 1:niter x = muladd(x, 0.999, 1e-3); end
        @inbounds A[i] = isnan(x) ? -one(v) : v
    end
    return nothing
end
const NITER = 4000
fill_slowly!(A::Array, v)    = fill!(A, v)
fill_slowly!(A::CuArray, v)  = (CUDA.@cuda threads=256 blocks=cld(length(A),256) fill_slowly_cuda!(A, v, NITER); nothing)
fill_slowly!(A::ROCArray, v) = (AMDGPU.@roc groupsize=256 gridsize=cld(length(A),256) fill_slowly_amdgpu!(A, v, NITER); nothing)
synchronize_all(A::Array)    = nothing
synchronize_all(A::CuArray)  = CUDA.synchronize()
synchronize_all(A::ROCArray) = AMDGPU.synchronize()

array_types  = ["CPU"]
device_types = ["auto"]
allocators   = Function[zeros]
sizes        = [32]              # Local array size in each dimension.
overlaps     = [8]               # Overlap in each dimension (the halowidth is half of it).
gpuaware     = Vector{Any}[[nothing]]  # GPU-aware MPI variants (nothing: as set in the environment).
if test_cuda
    push!(array_types, "CUDA"); push!(device_types, "CUDA"); push!(allocators, CUDA.zeros); push!(sizes, 256); push!(overlaps, 128)  # Halos of width 64: the local exchange copies 33 MB per neighbor.
end
if test_amdgpu
    push!(array_types, "AMDGPU"); push!(device_types, "AMDGPU"); push!(allocators, AMDGPU.zeros); push!(sizes, 256); push!(overlaps, 128)
end

## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.
for array_type in array_types[2:end] push!(gpuaware, (nprocs == 1) ? Any[false, true] : Any[nothing]); end
T    = Float64
nit  = 6

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. halo update after a kernel, synchronized before the call (arrays: $array_type)" for (array_type, device_type, zeros, n, ol, aware_variants) in zip(array_types, device_types, allocators, sizes, overlaps, gpuaware)
        for aware in aware_variants, d = 1:3
            env = isnothing(aware) ? () : ("IGG_CUDAAWARE_MPI" => string(Int(aware)), "IGG_ROCMAWARE_MPI" => string(Int(aware)))
            withenv(env...) do
                init_global_grid(n, n, n; periodx=(d==1), periody=(d==2), periodz=(d==3), overlaps=(ol, ol, ol), quiet=true, init_MPI=false, device_type=device_type);
            end
            gpuaware_d = (array_type == "CUDA") ? GG.cudaaware_MPI(d) : (array_type == "AMDGPU") ? GG.amdgpuaware_MPI(d) : false
            exchange   = all(GG.neighbors(d) .== GG.me()) ? "local" : "MPI"
            @testset "dimension $d periodic ($exchange exchange, GPU-aware MPI: $gpuaware_d)" begin
                A     = zeros(T, n, n, n);
                stale = 0
                for k = 1:nit
                    fill_slowly!(A, T(k));
                    synchronize_all(A);                  # The contract: the caller synchronizes before the call.
                    update_halo!(A);                     # NOTE: no synchronization after the call: update_halo! returns once the halo update is complete.
                    stale += count(!=(T(k)), A)          # All elements must hold k: the inner points from the kernel, the halos from the (periodic) neighbors (non-periodic boundaries keep the value of the kernel).
                end
                @test stale == 0
            end;
            finalize_global_grid(finalize_MPI=false);
        end
    end;
end;

## Test tear down
MPI.Finalize()
