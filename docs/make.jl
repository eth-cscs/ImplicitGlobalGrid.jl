using Documenter
using DocumenterCitations
using DocumenterVitepress
using Markdown
using TOML
using ImplicitGlobalGrid

const EXAMPLEROOT = joinpath(@__DIR__, "..", "examples")

# Return the example script `file` as a Julia code block (used in the `@eval` blocks of the example pages).
mdinclude(file) = Markdown.parse("```julia\n" * read(joinpath(EXAMPLEROOT, file), String) * "\n```")

# Return the table "Behaviour per version" of stream_semantics.md, generated from the expectations of the stack check (used in an `@eval` block).
function stream_semantics_table()
    entries = TOML.parsefile(joinpath(@__DIR__, "..", "ci", "probes", "expectations.toml"))["xstream"]
    cases   = ["Kernel launch using `A`"                                   => "kernel_same",
               "`pointer(A)` (e.g. for a memory copy)"                     => "pointer_same",
               "`pointer(A)`, if `s1` is the task-local stream"            => "pointer_same_tasklocal",
               "Kernel launch using another array"                         => "kernel_other",
               "Any of the above after `enable_synchronization!(A, false)`" => "optout"]
    verdict = Dict("HOST_BLOCKED" => "host waits", "GPU_SERIALIZED" => "GPU waits", "CONCURRENT" => "concurrent")
    rows    = ["| Use on `s2` | " * join((e["label"] for e in entries), " | ") * " |",
               "|:--" * " |:--"^length(entries) * " |"]
    for (description, case) in cases
        push!(rows, "| $description | " * join((verdict[e[case]] for e in entries), " | ") * " |")
    end
    return Markdown.parse(join(rows, "\n"))
end

const bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)


@info "Building documentation website using DocumenterVitepress.jl..."
makedocs(;
    modules   = [ImplicitGlobalGrid],
    authors   = "Samuel Omlin, Ludovic Räss, Ivan Utkin",
    sitename  = "ImplicitGlobalGrid.jl",
    format    = DocumenterVitepress.MarkdownVitepress(
        repo      = "github.com/eth-cscs/ImplicitGlobalGrid.jl",
        devbranch = "main",
        devurl    = "dev",
    ),
    checkdocs = :exports,
    plugins   = [bib],
    pages     = [
        "Home"          => "index.md",
        "Overview"      => "overview.md",
        "Usage"         => "usage.md",
        "GPU streams"   => "stream_semantics.md",
        "Examples"      => [
            "examples/diffusion3D_multigpu_CuArrays_novis.md",
            "examples/diffusion3D_multigpu_CuArrays_onlyvis.md",
        ],
        "Testing"       => "testing.md",
        "API reference" => "api.md",
        "References"    => "references.md",
    ],
)


@info "Deploying docs..."
DocumenterVitepress.deploydocs(;
    repo         = "github.com/eth-cscs/ImplicitGlobalGrid.jl",
    devbranch    = "main",
    push_preview = true,
)
