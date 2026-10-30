# Create the environment of the stack check (run_stack_check.sh, active project): ImplicitGlobalGrid of this repository, the GPU package
# STACK_BACKEND in the version STACK_BACKEND_VERSION (a Pkg version specifier, e.g. "5.4" for the latest 5.4.x; empty: the latest release;
# "<url>#<rev>": a branch or commit, e.g. of a pull request), ParallelStencil (STACK_PS: empty for the latest release, else a path to
# develop or "<url>#<rev>") and MPI.
using Pkg

backend = ENV["STACK_BACKEND"]
version = get(ENV, "STACK_BACKEND_VERSION", "")
ps      = get(ENV, "STACK_PS", "")

Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
specs = [PackageSpec(name="MPI")]
if isempty(version)
    push!(specs, PackageSpec(name=backend))
elseif occursin('#', version)
    url, rev = split(version, "#")
    push!(specs, PackageSpec(url=url, rev=rev))
else
    push!(specs, PackageSpec(name=backend, version=version))
end
if isempty(ps)
    push!(specs, PackageSpec(name="ParallelStencil"))
elseif isdir(ps)
    Pkg.develop(path=ps)
else
    url, rev = split(ps, "#")
    push!(specs, PackageSpec(url=url, rev=rev))
end
Pkg.add(specs)
Pkg.precompile()

deps = Pkg.dependencies()
v(name) = only(d.version for d in values(deps) if d.name == name)
println("STACKENV julia=$VERSION backend=$backend version=$(v(backend)) source=$(occursin('#', version) ? version : "registry") PS=$(v("ParallelStencil")) PS_source=$(isempty(ps) ? "registry" : ps) IGG=$(v("ImplicitGlobalGrid")) MPI=$(v("MPI"))")
