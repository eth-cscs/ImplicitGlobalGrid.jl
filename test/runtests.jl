# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
#
# Each test file is run in its own Julia process; all test files are run unless some are selected by name in ARGS (test_args). The following environment variables control the test run:
#   IGG_TEST_NPROCS:  number of MPI processes to run the test files in `multirankfiles` with (default: 1). All other test files require one single process.
#   IGG_TEST_MPIEXEC: MPI launcher command overriding `MPI.mpiexec()`; if set, all test files are launched with it (e.g. "srun --overlap" from within a Slurm job step).
#   IGG_TEST_BACKEND: GPU backend that must be functional ("CUDA" or "AMDGPU"); if it is not, the test run fails instead of skipping the GPU tests.
import ImplicitGlobalGrid # Precompile it.
import ImplicitGlobalGrid: SUPPORTED_DEVICE_TYPES, DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU
import MPI
@static if (DEVICE_TYPE_CUDA in SUPPORTED_DEVICE_TYPES) import CUDA end
@static if (DEVICE_TYPE_AMDGPU in SUPPORTED_DEVICE_TYPES) import AMDGPU end

excludedfiles  = ["test_excluded.jl"];
multirankfiles = ["test_finalize_global_grid.jl", "test_gather.jl", "test_select_device.jl", "test_update_halo.jl"];

function runtests()
    juliacmd  = Base.julia_cmd() # NOTE: julia_cmd() forwards the options of this process (e.g. --code-coverage and --check-bounds set by Pkg.test) to the test processes.
    testdir   = @__DIR__
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))
    if (!isempty(ARGS)) testfiles = filter(f -> basename(f) ∈ ARGS, testfiles) end # NOTE: test files can be selected by name, e.g. Pkg.test(test_args=["test_gather.jl"]).
    nprocs    = parse(Int, get(ENV, "IGG_TEST_NPROCS", "1"))
    mpiexec   = haskey(ENV, "IGG_TEST_MPIEXEC") ? Cmd(Base.shell_split(ENV["IGG_TEST_MPIEXEC"])) : MPI.mpiexec()
    backend   = get(ENV, "IGG_TEST_BACKEND", "")

    failedfiles = String[]
    printstyled("Testing package ImplicitGlobalGrid.jl (processes for multi-rank tests: $nprocs)\n"; bold=true, color=:white)

    if !(backend in ["", DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU])
        error("IGG_TEST_BACKEND: invalid value ($backend). Valid values are: $DEVICE_TYPE_CUDA, $DEVICE_TYPE_AMDGPU or empty.")
    end

    if (DEVICE_TYPE_CUDA in SUPPORTED_DEVICE_TYPES && !CUDA.functional())
        if (backend == DEVICE_TYPE_CUDA) error("IGG_TEST_BACKEND=$backend, but CUDA is not functional (type `import CUDA; CUDA.functional(true)` to debug your CUDA installation).") end
        @warn "Test Skip: All CUDA tests will be skipped because CUDA is not functional (if this is unexpected type `import CUDA; CUDA.functional(true)` to debug your CUDA installation)."
    end

    if (DEVICE_TYPE_AMDGPU in SUPPORTED_DEVICE_TYPES && !AMDGPU.functional())
        if (backend == DEVICE_TYPE_AMDGPU) error("IGG_TEST_BACKEND=$backend, but AMDGPU is not functional (type `import AMDGPU; AMDGPU.functional()` to debug your AMDGPU installation).") end
        @warn "Test Skip: All AMDGPU tests will be skipped because AMDGPU is not functional (if this is unexpected type `import AMDGPU; AMDGPU.functional()` to debug your AMDGPU installation)."
    end

    for f in testfiles
        println("")
        if basename(f) ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        np = (basename(f) ∈ multirankfiles) ? nprocs : 1
        cmd = `$juliacmd -O3 --startup-file=no $f`
        if (np > 1 || haskey(ENV, "IGG_TEST_MPIEXEC")) cmd = `$mpiexec -n $np $cmd` end
        try
            run(cmd)
        catch ex
            push!(failedfiles, basename(f))
        end
    end

    if !isempty(failedfiles)
        printstyled("\nTest files with failures or errors: $(join(failedfiles, ", "))\n"; bold=true, color=:red)
    end
    return length(failedfiles)
end

exit(runtests())
