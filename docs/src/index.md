```@raw html
---
layout: home

hero:
  name: ImplicitGlobalGrid.jl
  text: Distributed parallelization of stencil-based GPU and CPU applications
  tagline: Close to ideal weak scaling on thousands of GPUs.
  actions:
    - theme: brand
      text: Usage 🚀
      link: /usage
    - theme: alt
      text: API reference 📚
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/eth-cscs/ImplicitGlobalGrid.jl
  image:
    src: /logo.png
    alt: ImplicitGlobalGrid.jl

features:
  - icon: 🧩
    title: Three functions
    details: "<code>init_global_grid</code>, <code>update_halo!</code> and <code>finalize_global_grid</code> can be enough to turn a single-GPU/CPU code into a massively scaling Multi-GPU/CPU application."
    link: /usage#usage-three-functions
  - icon: 🌐
    title: Implicit global grid
    details: The global grid is created implicitly from the number of processes and the process topology, defined automatically or chosen by the user; 1-D, 2-D and 3-D grids are supported.
    link: /overview
  - icon: 🖥️
    title: GPUs and CPUs
    details: Nvidia and AMD GPUs through CUDA.jl and AMDGPU.jl, with automatic device selection, and CPUs with multi-threaded halo copies through Polyester.jl.
    link: /usage#usage-gpu
  - icon: ⚡
    title: GPU-aware MPI
    details: Halo updates close to hardware limit with MPI.jl; CUDA-aware or ROCm-aware MPI is activated with an environment variable, and communication can be hidden behind computation.
    link: /usage#usage-gpu-aware-mpi
  - icon: 🔗
    title: Interoperable with MPI.jl
    details: The Cartesian communicator is exposed, the initialization and finalization of MPI can be left to the application, and every argument passed to MPI.jl is customizable.
    link: /usage#usage-mpi
  - icon: 📐
    title: Global coordinates and extents
    details: Global coordinates, sizes and indices of the local grid, and local and global array extents, e.g. for parallel VTK output.
    link: /usage#usage-grid-queries
  - icon: 📊
    title: In-situ visualization
    details: "<code>gather!</code> assembles an array from all processes on a single process according to the global grid, for in-situ visualization or monitoring."
    link: /usage#in-situ-usage
  - icon: 🗂️
    title: Multiple grids
    details: "Several global grids can be created and activated; <code>update_halo!</code> operates on the active grid or on the grid passed to it."
    link: /usage#usage-multiple-grids
  - icon: 🧊
    title: Arrays of small arrays / structs
    details: "<code>update_halo!</code> supports the logical arrays of small arrays or structs of CellArrays.jl."
    link: /usage#usage-cellarrays
---
```

## What is ImplicitGlobalGrid.jl?

ImplicitGlobalGrid renders the distributed parallelization of stencil-based GPU and CPU applications on a regular staggered grid almost trivial and enables close to ideal weak scaling of real-world applications on thousands of GPUs.

The global computational grid is created *implicitly* from the number of processes the application is run with: the user writes a code for one GPU/CPU (*local grid*); then, **as little as three functions can be enough to transform it into a massively scaling Multi-GPU/CPU application**. Halo updates rely on [MPI.jl](https://github.com/JuliaParallel/MPI.jl) and can leverage CUDA-aware or ROCm-aware MPI. 1-D, 2-D and 3-D grids are supported. See the [Overview](overview.md) for more details.

## Installation
ImplicitGlobalGrid can be installed directly with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/) from the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/):
```julia
import Pkg; Pkg.add("ImplicitGlobalGrid")
```

## Dependencies
ImplicitGlobalGrid relies on the Julia MPI wrapper ([MPI.jl](https://github.com/JuliaParallel/MPI.jl)) and [CellArrays.jl](https://github.com/omlins/CellArrays.jl); GPU support is provided through package extensions for the Julia CUDA package ([CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) [Besard2019TPDS, Besard2019AES](@cite)) and the Julia AMDGPU package ([AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)), and multi-threaded CPU halo copies through [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).

## Questions, comments and discussions
To discuss technical issues, please post on Julia Discourse in the [GPU topic](https://discourse.julialang.org/c/domain/gpu) or the [Julia at Scale topic](https://discourse.julialang.org/c/domain/parallel) or in the `#gpu` or `#hpc` channels on the [Julia Slack](https://julialang.org/slack/).
To discuss numerical/domain-science issues, please post on Julia Discourse in the [Numerics topic](https://discourse.julialang.org/c/domain/numerics) or the [Modelling & Simulations topic](https://discourse.julialang.org/c/domain/models) or whichever other topic fits best your issue.

## Contributing
Contributions are welcome, be it new examples, convenience features or core improvements! Please open an issue to discuss your idea beforehand. A pull request should address an issue in its completeness and blend nicely into the existing project (community guidelines such as [ColPrac](https://github.com/SciML/ColPrac) can serve as inspiration).

## Authors
ImplicitGlobalGrid originated from a collaboration of the Swiss National Supercomputing Centre (CSCS), ETH Zurich with Stanford University and the Geocomputing Centre, University of Lausanne (Prof. Yuri Podladchikov). It is developed by Samuel Omlin ([@omlins](https://github.com/omlins), CSCS, ETH Zurich), Ludovic Räss ([@luraess](https://github.com/luraess), University of Lausanne) and Ivan Utkin ([@utkinis](https://github.com/utkinis), ETH Zurich).

## Citing
If you use ImplicitGlobalGrid in your work, please cite [OmlinRaessUtkin2024](@cite). The BibTeX entry and further publications are found in the [References](references.md).

## Funding
The development of this package is supported by the [GPU4GEO](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html) and [∂GPU4GEO](https://pasc-ch.org/projects/2025-2028/gpu4geo-differentiable-multi-physics-solvers-for-extreme-scale-geophysics-simulations-on-gpus/index.html) PASC projects. More information about the GPU4GEO project can be found on the [GPU4GEO website](https://GPU4GEO.org/).
