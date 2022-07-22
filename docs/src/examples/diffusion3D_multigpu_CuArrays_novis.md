# 50-lines Multi-GPU example

This simple Multi-GPU 3-D heat diffusion solver uses ImplicitGlobalGrid. It relies fully on the broadcasting capabilities of [CUDA.jl]'s `CuArray` type to perform the stencil-computations with maximal simplicity ([CUDA.jl] enables also writing explicit GPU kernels which can lead to significantly better performance for these computations).

```@eval
Main.mdinclude(joinpath(Main.EXAMPLEROOT, "diffusion3D_multigpu_CuArrays_novis.jl"))
```

The corresponding file can be found [here](docs/examples/diffusion3D_multigpu_CuArrays_novis.jl). A basic CPU-only example is available [here](docs/examples/diffusion3D_multicpu_novis.jl) (no usage of multi-threading).
