# Overlap probe (ParallelStencil + ImplicitGlobalGrid): does @hide_communication of ParallelStencil overlap update_halo! with the
# computation? 3-D diffusion step on n^3 per process. Exchange in one dimension (PROBE_DIMS entries 1, 2, 3: periodic in that dimension, all
# processes along it; with one process, the local exchange) or in all (entry 0: fully periodic, default process topology).
# @hide_communication is timed as is (optout=0) and, if the GPU package provides `enable_synchronization!`, with the implicit
# synchronization of the arrays disabled by the caller (optout=1).
# One line per case (rank 0; times: maximum over the processes of the minimum over PROBE_REPS repetitions of the mean per iteration):
#   PSOVERLAP backend=.. version=.. PS=.. IGG=.. nprocs=.. aware=.. dim=.. n=.. bkopt=.. psopt=.. optout=.. comp_ms=.. comm_ms=.. seq_ms=.. split_ms=.. hide_ms=.. hidden=.. hidden_split=..
#   comp: computation alone; comm: update_halo! alone; seq: computation followed by update_halo!; split: @hide_communication without
#   communication (the cost of the split computation); hide: @hide_communication.
#   hidden = (seq - hide) / min(comp, comm); hidden_split = (split + comm - hide) / min(split, comm), which does not count the cost of the
#   split computation: about 1 = communication fully hidden; about 0 or below = serialized.
#   bkopt: the GPU package provides enable_synchronization!; psopt: ParallelStencil suspends the implicit synchronization in @hide_communication.
# Env: PROBE_BACKEND=CUDA|AMDGPU (default CUDA), PROBE_N (default 256; a list, e.g. "256 512", runs each), PROBE_NT (default 20), PROBE_REPS
# (default 3), PROBE_DIMS (default "1 2 3 0"). GPU-aware MPI: IGG_CUDAAWARE_MPI / IGG_ROCMAWARE_MPI as usual. Everything runs inside functions (ParallelStencil 0.17
# emits swap assignments to the kernel arguments after @parallel calls, omlins/ParallelStencil.jl#243).
const BACKEND = Symbol(get(ENV, "PROBE_BACKEND", "CUDA"))
if BACKEND == :CUDA
    using CUDA
else
    using AMDGPU
end
using ParallelStencil, ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid, MPI, Printf
const GPU   = getfield(Main, BACKEND)
const BKOPT = isdefined(GPU, :enable_synchronization!)
const PSOPT = isdefined(ParallelStencil.ParallelKernel, :suspend_implicit_sync)
const NREPS = parse(Int, get(ENV, "PROBE_REPS", "3"))
@eval @init_parallel_stencil($BACKEND, Float64, 3)

@parallel function step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2))
    return
end

function comp!(T2, T, Ci, lam, dt, dx, dy, dz)
    @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
    return
end
function comm!(T2, T, Ci, lam, dt, dx, dy, dz)
    update_halo!(T2)
    return
end
function seq!(T2, T, Ci, lam, dt, dx, dy, dz)
    @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
    update_halo!(T2)
    return
end
function hide!(T2, T, Ci, lam, dt, dx, dy, dz)
    @hide_communication (16, 2, 2) begin
        @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
        update_halo!(T2)
    end
    return
end
nocomm!(A) = nothing
function split!(T2, T, Ci, lam, dt, dx, dy, dz)
    @hide_communication (16, 2, 2) begin
        @parallel step!(T2, T, Ci, lam, dt, dx, dy, dz)
        nocomm!(T2)
    end
    return
end

function timeit(f, nt, args...)
    f(args...); GPU.synchronize()  # Warm-up (compilation).
    t = Inf
    for _ in 1:NREPS  # NOTE: the minimum over repetitions, as single slow iterations (e.g. a garbage collection) inflate a mean.
        MPI.Barrier(MPI.COMM_WORLD)
        t0 = time_ns()
        for _ in 1:nt; f(args...); end
        GPU.synchronize()
        t = min(t, (time_ns() - t0) / 1e9 / nt)
    end
    return MPI.Allreduce(t, max, MPI.COMM_WORLD)
end

aware() = get(ENV, BACKEND == :CUDA ? "IGG_CUDAAWARE_MPI" : "IGG_ROCMAWARE_MPI", "0")

function probe(dim, n, nt)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    dxyz = dim == 0 ? (0, 0, 0) : ntuple(i -> i == dim ? nprocs : 1, 3)
    per  = dim == 0 ? (1, 1, 1) : ntuple(i -> Int(i == dim), 3)
    me, = init_global_grid(n, n, n; dimx=dxyz[1], dimy=dxyz[2], dimz=dxyz[3], periodx=per[1], periody=per[2], periodz=per[3], init_MPI=false, quiet=true)
    lam = 1.0; dx = dy = dz = 1.0; dt = dx^2/lam/8.1
    T  = @rand(n, n, n)
    T2 = copy(T)
    Ci = @ones(n, n, n)
    args = (T2, T, Ci, lam, dt, dx, dy, dz)
    t_comp = timeit(comp!, nt, args...)
    t_comm = timeit(comm!, nt, args...)
    t_seq  = timeit(seq!,  nt, args...)
    for optout in (BKOPT ? (false, true) : (false,))
        optout && foreach(A -> GPU.enable_synchronization!(A, false), (T, T2, Ci))
        t_split = timeit(split!, nt, args...)
        t_hide  = timeit(hide!,  nt, args...)
        optout && foreach(A -> GPU.enable_synchronization!(A, true), (T, T2, Ci))
        hidden       = (t_seq - t_hide) / min(t_comp, t_comm)
        hidden_split = (t_split + t_comm - t_hide) / min(t_split, t_comm)
        me == 0 && @printf("PSOVERLAP backend=%s version=%s PS=%s IGG=%s nprocs=%d aware=%s dim=%d n=%d bkopt=%d psopt=%d optout=%d comp_ms=%.3f comm_ms=%.3f seq_ms=%.3f split_ms=%.3f hide_ms=%.3f hidden=%.2f hidden_split=%.2f\n",
                           BACKEND, pkgversion(GPU), pkgversion(ParallelStencil), pkgversion(ImplicitGlobalGrid), nprocs, aware(), dim, n, BKOPT, PSOPT, optout,
                           1e3t_comp, 1e3t_comm, 1e3t_seq, 1e3t_split, 1e3t_hide, hidden, hidden_split)
    end
    finalize_global_grid(finalize_MPI=false)
    return
end

function main()
    MPI.Init()
    nt = parse(Int, get(ENV, "PROBE_NT", "20"))
    for n in parse.(Int, split(get(ENV, "PROBE_N", "256"))), dim in parse.(Int, split(get(ENV, "PROBE_DIMS", "1 2 3 0")))
        probe(dim, n, nt)
    end
    MPI.Finalize()
    return
end

main()
