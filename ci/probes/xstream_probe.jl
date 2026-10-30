# Cross-stream probe (GPU package only): a long kernel uses the array A on a stream s1; while it runs, A (or another array) is used on a
# stream s2. Does that block the host (host-side synchronization), make s2 wait on the GPU (device-side wait) or run concurrently? Measured
# with the implicit synchronization of the GPU package and, if the package provides `enable_synchronization!`, with it disabled on A
# (optout=1). Cases:
#   kernel_other:           kernel launch on s2 using another array.
#   kernel_same:            kernel launch on s2 using A (as the pack kernels of update_halo!, dimension 1).
#   pointer_same:           pointer(A) (as the 3-D memory copies of update_halo!, dimensions 2 and 3); s1 is not the task-local stream.
#   pointer_same_tasklocal: pointer(A) as above, but s1 is the task-local stream (the usual case: the caller computes on the task-local stream).
# One line per case:
#   XSTREAM backend=.. version=.. case=.. optout=0|1 op_ms=<host time of the operation> s2_done_ms=<time until s2 is done> T1_ms=<long kernel> verdict=HOST_BLOCKED|GPU_SERIALIZED|CONCURRENT
# Env: PROBE_BACKEND=CUDA|AMDGPU (default CUDA). See docs/src/stream_semantics.md; expected verdicts: expectations.toml.
const BACKEND = Symbol(get(ENV, "PROBE_BACKEND", "CUDA"))
using Printf
# NOTE: the definitions are in @eval, as the macros of a toplevel `if` block are all expanded before `using` runs.
if BACKEND == :CUDA
    using CUDA
    @eval begin
        nap!(a, n) = (for _ in 1:n; CUDA.nanosleep(UInt32(1000)); end; @inbounds a[1] = 1; nothing)
        poke!(a) = (@inbounds a[2] = 2; nothing)
        launch_nap(s, a, n) = (@cuda threads=1 blocks=1 stream=s nap!(a, n); nothing)
        launch_poke(s, a) = (@cuda threads=1 blocks=1 stream=s poke!(a); nothing)
        newstream() = CuStream()
    end
else
    using AMDGPU
    @eval begin
        nap!(a, n) = (for _ in 1:n; AMDGPU.Device.device_sleep(Int32(127)); end; @inbounds a[1] = 1; nothing)  # NOTE: s_sleep instead of a clock, as gfx11+ lacks s_memrealtime.
        poke!(a) = (@inbounds a[2] = 2; nothing)
        launch_nap(s, a, n) = (@roc groupsize=1 gridsize=1 stream=s nap!(a, n); nothing)
        launch_poke(s, a) = (@roc groupsize=1 gridsize=1 stream=s poke!(a); nothing)
        newstream() = HIPStream()
    end
end
const GPU = getfield(Main, BACKEND)
const HAS_OPTOUT = isdefined(GPU, :enable_synchronization!)

syncall(s1, s2) = (GPU.synchronize(s1); GPU.synchronize(s2); GPU.synchronize(); nothing)

function main()
    s1, s2 = newstream(), newstream()
    A = GPU.zeros(Int, 4)
    B = GPU.zeros(Int, 4)
    # Compile, then calibrate the long kernel to at least 0.2 s.
    launch_nap(s1, A, 1); launch_poke(s2, B); syncall(s1, s2)
    n = 1000
    T1 = 0.0
    while true
        syncall(s1, s2)
        T1 = @elapsed (launch_nap(s1, A, n); GPU.synchronize(s1))
        T1 > 0.2 && break
        n *= 2
    end
    cases = (("kernel_other",           s1,           () -> launch_poke(s2, B)),
             ("kernel_same",            s1,           () -> launch_poke(s2, A)),
             ("pointer_same",           s1,           () -> (pointer(A); nothing)),
             ("pointer_same_tasklocal", GPU.stream(), () -> (pointer(A); nothing)))
    for optout in (HAS_OPTOUT ? (false, true) : (false,)), (case, s_nap, op) in cases
        for rep in 1:2  # Report the second repetition.
            syncall(s_nap, s2)
            optout && GPU.enable_synchronization!(A, false)
            t0 = time()
            launch_nap(s_nap, A, n)
            top = @elapsed op()
            GPU.synchronize(s2)
            ts2 = time() - t0  # NOTE: the work on s2 is tiny: if s2 is done only after about T1, it waited for the long kernel on the GPU.
            syncall(s_nap, s2)
            optout && GPU.enable_synchronization!(A, true)
            verdict = top > 0.5T1 ? "HOST_BLOCKED" : (ts2 > 0.5T1 ? "GPU_SERIALIZED" : "CONCURRENT")
            rep == 2 && @printf("XSTREAM backend=%s version=%s case=%s optout=%d op_ms=%.2f s2_done_ms=%.1f T1_ms=%.1f verdict=%s\n",
                                BACKEND, pkgversion(GPU), case, optout, 1e3top, 1e3ts2, 1e3T1, verdict)
        end
    end
end

main()
