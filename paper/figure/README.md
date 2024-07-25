# Parallel efficiency on a weak scaling benchmark

This folder contains raw data of a weak scaling benchmark from 1 to 2197
Nvidia P100 GPUs on the Cray XC 50 Piz Daint at CSCS. The raw data are
contained in the `out_diff3D_pareff.txt` script and the plotting script
is `julia_scale.jl`. To reproduce the figure, execute
```bash
julia julia_scale.jl
```

The script was created for Julia v1.8.0.
