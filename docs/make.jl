using ImplicitGlobalGrid
using Documenter
using DocExtensions
using DocExtensions.DocumenterExtensions

const DOCSRC      = joinpath(@__DIR__, "src")
const EXAMPLEROOT = joinpath(@__DIR__, "..", "examples")

DocMeta.setdocmeta!(ImplicitGlobalGrid, :DocTestSetup, :(using ImplicitGlobalGrid); recursive=true)


@info "Preprocessing .MD-files..."
include("reflinks.jl")
MarkdownExtensions.expand_reflinks(reflinks; rootdir=DOCSRC)


@info "Building documentation website using Documenter.jl..."
makedocs(;
    modules  = [ImplicitGlobalGrid],
    authors  = "Samuel Omlin, Ludovic Räss, Ivan Utkin",
    repo     = "https://github.com/eth-cscs/ImplicitGlobalGrid.jl/blob/{commit}{path}#{line}",
    sitename = "ImplicitGlobalGrid.jl",
    format   = Documenter.HTML(;
        prettyurls       = get(ENV, "CI", "false") == "true",
        canonical        = "https://omlins.github.io/ImplicitGlobalGrid.jl",
        collapselevel    = 1,
        sidebar_sitename = true,
        edit_link        = "master",
        #assets           = [asset("https://img.shields.io/github/stars/eth-cscs/ImplicitGlobalGrid.jl.svg", class = :ico)],
        #warn_outdated    = true,
    ),
    pages   = [
        "Introduction"  => "index.md",
        "Usage"         => "usage.md",
        "Examples"      => [hide("..." => "examples.md"),
                            "examples/diffusion3D_multigpu_CuArrays_novis.md",
                            "examples/diffusion3D_multigpu_CuArrays_onlyvis.md",
                           ],
        "API reference" => "api.md",
    ],
)


@info "Deploying docs..."
deploydocs(;
    repo         = "github.com/eth-cscs/ImplicitGlobalGrid.jl",
    push_preview = true,
    devbranch    = "master",
)
