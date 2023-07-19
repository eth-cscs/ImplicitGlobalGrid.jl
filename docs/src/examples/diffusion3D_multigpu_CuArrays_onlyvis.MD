# [Straightforward in-situ visualization / monitoring](@id in-situ-example)

Thanks to the function [`gather!`](@ref), ImplicitGlobalGrid enables straightforward in-situ visualization or monitoring of Multi-GPU/CPU applications using e.g. the [Julia Plots package] as shown in the following (the GR backend is used as it is particularly fast according to the [Julia Plots documentation]). It is enough to add a couple of lines to the [previous example](@ref example-50-lines) (omitted unmodified lines are represented with `#(...)`):

```@eval
Main.mdinclude(joinpath(Main.EXAMPLEROOT, "diffusion3D_multigpu_CuArrays_onlyvis.jl"))
```

Here is the resulting movie when running the application on 8 GPUs, solving 3-D heat diffusion with heterogeneous heat capacity (two Gaussian anomalies) on a global computational grid of size 510x510x510 grid points. It shows the x-z-dimension plane in the middle of the dimension y:

![Implicit global grid](../../../assets/videos/diffusion3D_8gpus.gif)

The simulation producing this movie - *including the in-situ visualization* - took 29 minutes on 8 NVIDIA® Tesla® P100 GPUs on Piz Daint (an optimized solution using [CUDA.jl]'s native kernel programming capabilities can be more than 10 times faster).
The complete example can be found [here](../../../assets/examples/diffusion3D_multigpu_CuArrays.jl). A corresponding basic cpu-only example is available [here](../../../assets/examples/diffusion3D_multicpu.jl) (no usage of multi-threading) and a movie of a simulation with 254x254x254 grid points which it produced within 34 minutes using 8 Intel® Xeon® E5-2690 v3 is found [here](../../../assets/videos/diffusion3D_8cpus.gif) (with 8 processes, no multi-threading).
