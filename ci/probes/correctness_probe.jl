# Correctness probe (ParallelStencil + ImplicitGlobalGrid): nt steps of a 3-D diffusion with @hide_communication and with @parallel
# followed by update_halo!, from the same initial condition (fully periodic, n^3 per process). The results must be bitwise equal on every
# process, also where @hide_communication suspends the implicit synchronization of the arrays.
# One line (rank 0):
#   PSCORRECT backend=.. version=.. PS=.. nprocs=.. aware=.. n=.. nt=.. equal=true|false maxdiff=..
# Env: PROBE_BACKEND=CUDA|AMDGPU (default CUDA), PROBE_N (default 64), PROBE_NT (default 50); GPU-aware MPI as usual (IGG_*AWARE_MPI).
const BACKEND = Symbol(get(ENV, "PROBE_BACKEND", "CUDA"))
if BACKEND == :CUDA
    using CUDA
else
    using AMDGPU
end
using ParallelStencil, ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid, MPI, Printf
const GPU = getfield(Main, BACKEND)
@eval @init_parallel_stencil($BACKEND, Float64, 3)

@parallel function step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2))
    return
end

function run_seq!(T, T2, Ci, lam, dt, dx, dy, dz, nt)
    for _ in 1:nt
        @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
        update_halo!(T2)
        T, T2 = T2, T
    end
    return T
end

function run_hide!(T, T2, Ci, lam, dt, dx, dy, dz, nt)
    for _ in 1:nt
        @hide_communication (8, 4, 4) begin
            @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
            update_halo!(T2)
        end
        T, T2 = T2, T
    end
    return T
end

aware() = get(ENV, BACKEND == :CUDA ? "IGG_CUDAAWARE_MPI" : "IGG_ROCMAWARE_MPI", "0")

function main()
    MPI.Init()
    n  = parse(Int, get(ENV, "PROBE_N", "64"))
    nt = parse(Int, get(ENV, "PROBE_NT", "50"))
    me, dims, nprocs = init_global_grid(n, n, n; periodx=1, periody=1, periodz=1, init_MPI=false, quiet=true)
    lam = 1.0; dx = dy = dz = 1.0; dt = dx^2/lam/8.1
    T0 = @rand(n, n, n)
    update_halo!(T0)
    Ci = @ones(n, n, n)
    Ts = Array(run_seq!(copy(T0), copy(T0), Ci, lam, dt, dx, dy, dz, nt))
    Th = Array(run_hide!(copy(T0), copy(T0), Ci, lam, dt, dx, dy, dz, nt))
    equal   = MPI.Allreduce(Ts == Th, &, MPI.COMM_WORLD)
    maxdiff = MPI.Allreduce(maximum(abs.(Ts .- Th)), max, MPI.COMM_WORLD)
    me == 0 && @printf("PSCORRECT backend=%s version=%s PS=%s nprocs=%d aware=%s n=%d nt=%d equal=%s maxdiff=%.3e\n",
                       BACKEND, pkgversion(GPU), pkgversion(ParallelStencil), nprocs, aware(), n, nt, equal, maxdiff)
    finalize_global_grid(finalize_MPI=false)
    MPI.Finalize()
    return
end

main()
