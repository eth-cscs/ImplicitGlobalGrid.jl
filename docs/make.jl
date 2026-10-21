using Documenter
using DocumenterCitations
using DocumenterVitepress
using Markdown
using ImplicitGlobalGrid

const EXAMPLEROOT = joinpath(@__DIR__, "..", "examples")

# Return the example script `file` as a Julia code block (used in the `@eval` blocks of the example pages).
mdinclude(file) = Markdown.parse("```julia\n" * read(joinpath(EXAMPLEROOT, file), String) * "\n```")

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
