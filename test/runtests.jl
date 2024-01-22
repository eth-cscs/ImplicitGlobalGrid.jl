# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src") # FIXME: to be removed everywhere?

import ImplicitGlobalGrid # Precompile it.
import ImplicitGlobalGrid: SUPPORTED_DEVICE_TYPES, DEVICE_TYPE_CUDA, DEVICE_TYPE_AMDGPU
@static if (DEVICE_TYPE_CUDA in SUPPORTED_DEVICE_TYPES) import CUDA end
@static if (DEVICE_TYPE_AMDGPU in SUPPORTED_DEVICE_TYPES) import AMDGPU end

excludedfiles = ["test_excluded.jl"];

function runtests()
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    nfail = 0
    printstyled("Testing package ImplicitGlobalGrid.jl\n"; bold=true, color=:white)

    if (DEVICE_TYPE_CUDA in SUPPORTED_DEVICE_TYPES && !CUDA.functional())
        @warn "Test Skip: All CUDA tests will be skipped because CUDA is not functional (if this is unexpected type `import CUDA; CUDA.functional(true)` to debug your CUDA installation)."
    end

    if (DEVICE_TYPE_AMDGPU in SUPPORTED_DEVICE_TYPES && !AMDGPU.functional())
        @warn "Test Skip: All AMDGPU tests will be skipped because AMDGPU is not functional (if this is unexpected type `import AMDGPU; AMDGPU.functional()` to debug your AMDGPU installation)."
    end

    for f in testfiles
        println("")
        if f ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end
    return nfail
end

exit(runtests())
