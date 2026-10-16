# Overview

ImplicitGlobalGrid renders the distributed parallelization of stencil-based GPU and CPU applications on a regular staggered grid almost trivial and enables close to ideal weak scaling of real-world applications on thousands of GPUs \[[1](https://pretalx.com/juliacon2019/talk/LGHLC3/), [2](https://pasc19.pasc-conference.org/program/schedule/presentation/?id=msa218&sess=sess144), [3](https://www.youtube.com/watch?v=vPsfZUqI4_0)\]:

```@raw html
<img src="./assets/images/fig_parEff_HM3D_Julia_CUDA_all_Daint_extrapol.png" alt="Weak scaling Piz Daint" style="display: block; margin: auto; width: 100%; max-width: 560px;">
```

ImplicitGlobalGrid relies on the Julia MPI wrapper ([MPI.jl](https://github.com/JuliaParallel/MPI.jl)) to perform halo updates close to hardware limit and leverages CUDA-aware or ROCm-aware MPI for GPU-applications. The communication can straightforwardly be hidden behind computation \[[1](https://pretalx.com/juliacon2019/talk/LGHLC3/), [3](https://www.youtube.com/watch?v=vPsfZUqI4_0)\] (how this can be done automatically when using ParallelStencil.jl is shown in \[[3](https://www.youtube.com/watch?v=vPsfZUqI4_0)\]; a general approach particularly suited for CUDA C applications is explained in \[[4](https://on-demand.gputechconf.com/gtc/2019/video/_/S9368/)\]).

A particularity of ImplicitGlobalGrid is the automatic *implicit creation of the global computational grid* based on the number of processes the application is run with (and based on the process topology, which can be explicitly chosen by the user or automatically defined). As a consequence, the user only needs to write a code to solve his problem on one GPU/CPU (*local grid*); then, **as little as three functions can be enough to transform a single GPU/CPU application into a massively scaling Multi-GPU/CPU application**. See the [50-lines Multi-GPU example](@ref example-50-lines). 1-D, 2-D and 3-D grids are supported. Here is a sketch of the global grid that results from running a 2-D solver with 4 processes (P1-P4) (a 2x2 process topology is created by default in this case):

```@raw html
<img src="./assets/images/implicit_global_grid.png" alt="Implicit global grid" style="display: block; margin: auto; width: 100%; max-width: 420px;">
```
