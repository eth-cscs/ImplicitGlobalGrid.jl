<h1> <img src="docs/src/assets/logo/logo_ImplicitGlobalGrid.png" alt="ImplicitGlobalGrid.jl" width="50"> ImplicitGlobalGrid.jl </h1>

# [ImplicitGlobalGrid.jl] [![Star on GitHub](https://img.shields.io/github/stars/eth-cscs/ImplicitGlobalGrid.jl.svg)](https://github.com/eth-cscs/ImplicitGlobalGrid.jl/stargazers)

ImplicitGlobalGrid renders the distributed parallelization of stencil-based GPU and CPU applications on a regular staggered grid almost trivial and enables close to ideal weak scaling of real-world applications on thousands of GPUs \[[1][JuliaCon19], [2][PASC19], [3][JuliaCon20a]\]:

![Weak scaling Piz Daint](docs/src/assets/images/fig_parEff_HM3D_Julia_CUDA_all_Daint_extrapol.png)

ImplicitGlobalGrid relies on the Julia MPI wrapper ([MPI.jl]) to perform halo updates close to hardware limit and leverages CUDA-aware or ROCm-aware MPI for GPU-applications. The communication can straightforwardly be hidden behind computation \[[1][JuliaCon19], [3][JuliaCon20a]\] (how this can be done automatically when using ParallelStencil.jl is shown in \[[3][JuliaCon20a]\]; a general approach particularly suited for CUDA C applications is explained in \[[4][GTC19]\]).

A particularity of ImplicitGlobalGrid is the automatic *implicit creation of the global computational grid* based on the number of processes the application is run with (and based on the process topology, which can be explicitly chosen by the user or automatically defined). As a consequence, the user only needs to write a code to solve his problem on one GPU/CPU (*local grid*); then, **as little as three functions can be enough to transform a single GPU/CPU application into a massively scaling Multi-GPU/CPU application**. See the [example](#multi-gpu-with-three-functions) below. 1-D, 2-D and 3-D grids are supported. Here is a sketch of the global grid that results from running a 2-D solver with 4 processes (P1-P4) (a 2x2 process topology is created by default in this case):

![Implicit global grid](docs/src/assets/images/implicit_global_grid.png)


## Dependencies
ImplicitGlobalGrid relies on the Julia MPI wrapper ([MPI.jl]), the Julia CUDA package ([CUDA.jl] \[[5][Julia CUDA paper 1], [6][Julia CUDA paper 2]\]) and the Julia AMDGPU package ([AMDGPU.jl]).


## References
\[1\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). Porting a Massively Parallel Multi-GPU Application to Julia: a 3-D Nonlinear Multi-Physics Flow Solver. JuliaCon Conference, Baltimore, USA.][JuliaCon19]

\[2\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). A Nonlinear Multi-Physics 3-D Solver: From CUDA C + MPI to Julia. PASC19 Conference, Zurich, Switzerland.][PASC19]

\[3\] [Omlin, S., Räss, L., Kwasniewski, G., Malvoisin, B., & Podladchikov, Y. Y. (2020). Solving Nonlinear Multi-Physics on GPU Supercomputers with Julia. JuliaCon Conference, virtual.][JuliaCon20a]

\[4\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). Resolving Spontaneous Nonlinear Multi-Physics Flow Localisation in 3-D: Tackling Hardware Limit. GPU Technology Conference 2019, San Jose, Silicon Valley, CA, USA.][GTC19]

\[5\] [Besard, T., Foket, C., & De Sutter, B. (2018). Effective Extensible Programming: Unleashing Julia on GPUs. IEEE Transactions on Parallel and Distributed Systems, 30(4), 827-841. doi: 10.1109/TPDS.2018.2872064][Julia CUDA paper 1]

\[6\] [Besard, T., Churavy, V., Edelman, A., & De Sutter B. (2019). Rapid software prototyping for heterogeneous and distributed platforms. Advances in Engineering Software, 132, 29-46. doi: 10.1016/j.advengsoft.2019.02.002][Julia CUDA paper 2]
