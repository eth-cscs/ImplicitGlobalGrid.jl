# Streams probe (ImplicitGlobalGrid + GPU package): the check of test/test_update_halo_streams.jl, for one GPU package only, so that it can
# run in an environment without the other one (e.g. CUDA.jl 5.4 and AMDGPU.jl 2 cannot be installed together). A slow kernel writes to the
# array on the task-local stream, the caller synchronizes (the contract of update_halo!), then update_halo! runs; all elements must then hold
# the new value. Large arrays and halos and a long kernel, so that a missing ordering would show. In each dimension (periodic in that
# dimension); with one process the halo update is local and both GPU-aware MPI settings are checked, with several processes the setting of
# the environment is used.
# One line per case (rank 0; stale: number of wrong elements, summed over the processes and iterations):
#   STREAMS backend=.. version=.. IGG=.. nprocs=.. dim=.. exchange=local|MPI aware=0|1 stale=..
# Env: PROBE_BACKEND=CUDA|AMDGPU (default CUDA), PROBE_N (default 256), PROBE_NT (default 6).
const BACKEND = Symbol(get(ENV, "PROBE_BACKEND", "CUDA"))
using Printf
import MPI
# NOTE: the definitions are in @eval, as the macros of a toplevel `if` block are all expanded before `using` runs.
if BACKEND == :CUDA
    using CUDA
    @eval begin
        function fill_slowly_kernel!(A, v, niter)
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            if i <= length(A)
                x = Float64(i)
                for _ = 1:niter x = muladd(x, 0.999, 1e-3); end
                @inbounds A[i] = isnan(x) ? -one(v) : v
            end
            return nothing
        end
        fill_slowly!(A, v, niter) = (@cuda threads=256 blocks=cld(length(A),256) fill_slowly_kernel!(A, v, niter); nothing)
    end
else
    using AMDGPU
    @eval begin
        function fill_slowly_kernel!(A, v, niter)
            i = (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
            if i <= length(A)
                x = Float64(i)
                for _ = 1:niter x = muladd(x, 0.999, 1e-3); end
                @inbounds A[i] = isnan(x) ? -one(v) : v
            end
            return nothing
        end
        fill_slowly!(A, v, niter) = (@roc groupsize=256 gridsize=cld(length(A),256) fill_slowly_kernel!(A, v, niter); nothing)
    end
end
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
const GPU   = getfield(Main, BACKEND)
const NITER = 4000

gpuaware_MPI(dim) = (BACKEND == :CUDA) ? GG.cudaaware_MPI(dim) : GG.amdgpuaware_MPI(dim)

function main()
    MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    n      = parse(Int, get(ENV, "PROBE_N", "256"))
    nt     = parse(Int, get(ENV, "PROBE_NT", "6"))
    ol     = n ÷ 2  # Halos of width n/4: with n=256, the local exchange copies 33 MB per neighbor.
    T      = Float64
    for aware in ((nprocs == 1) ? (false, true) : (nothing,)), d = 1:3
        env = isnothing(aware) ? () : ("IGG_CUDAAWARE_MPI" => string(Int(aware)), "IGG_ROCMAWARE_MPI" => string(Int(aware)))
        me, = withenv(env...) do
            init_global_grid(n, n, n; periodx=(d==1), periody=(d==2), periodz=(d==3), overlaps=(ol, ol, ol), quiet=true, init_MPI=false, device_type=string(BACKEND))
        end
        exchange = all(GG.neighbors(d) .== GG.me()) ? "local" : "MPI"
        A        = GPU.zeros(T, n, n, n)
        stale    = 0
        for k = 1:nt
            fill_slowly!(A, T(k), NITER)
            GPU.synchronize()                    # The contract: the caller synchronizes before the call.
            update_halo!(A)                      # NOTE: no synchronization after the call: update_halo! returns once the halo update is complete.
            stale += count(!=(T(k)), Array(A))   # All elements must hold k (non-periodic boundaries keep the value of the kernel).
        end
        stale = MPI.Allreduce(stale, +, MPI.COMM_WORLD)
        me == 0 && @printf("STREAMS backend=%s version=%s IGG=%s nprocs=%d dim=%d exchange=%s aware=%d stale=%d\n",
                           BACKEND, pkgversion(GPU), pkgversion(ImplicitGlobalGrid), nprocs, d, exchange, gpuaware_MPI(d), stale)
        finalize_global_grid(finalize_MPI=false)
    end
    MPI.Finalize()
    return
end

main()
