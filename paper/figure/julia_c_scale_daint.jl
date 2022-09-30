# Activate the current environment and load all packages
using Pkg
Pkg.activate(@__DIR__)

ax_log = false # choose between log or linear x-axis scale

using Plots, Plots.Measures

# Weak scaling parallel efficiency data on Piz Daint

# CUDA C
nprocs_C   = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,5120]
Teff_C     = [0.9500,0.9439,0.9433,0.9424,0.9356,0.9351,0.9346,0.9344,0.9342,0.9307,0.9320,0.9316,0.9313,0.9307]
Teff_lo_C  = [0.9499,0.9438,0.9432,0.9423,0.9354,0.9350,0.9345,0.9343,0.9342,0.9307,0.9261,0.9274,0.9277,0.9299]
Teff_hi_C  = [0.9501,0.9440,0.9433,0.9425,0.9357,0.9352,0.9347,0.9345,0.9344,0.9308,0.9322,0.9319,0.9316,0.9313]
σs_C       = Teff_hi_C .- Teff_lo_C

# Julia
nprocs_jl  = [1,2,4,8,16,32,64,128,256,512,1024]
Teff_jl    = [0.9870,0.9718,0.9714,0.9711,0.9564,0.9559,0.9559,0.9549,0.9537,0.9528,0.9521]
Teff_lo_jl = [0.9868,0.9718,0.9714,0.9710,0.9563,0.9559,0.9559,0.9547,0.9532,0.9525,0.9519]
Teff_hi_jl = [0.9870,0.9718,0.9715,0.9711,0.9564,0.9560,0.9560,0.9549,0.9538,0.9531,0.9523]
σs_jl      = Teff_hi_jl .- Teff_lo_jl

default(fontfamily="Computer Modern", linewidth=3,  markershape=:circle, markersize=4,
        framestyle=:box, fillalpha=0.4,margin=5mm)
scalefontsizes(); scalefontsizes(1.3)

# xtick_lin = (1,512,1024,2048,4096,5120)
xtick_lin = (1,64,128,256,512,1024)
plot(xlabel="Number of GPUs", ylabel="Parallel efficiency",
     xticks=(xtick_lin, string.(xtick_lin)))

plot!(nprocs_C[1:end-3], Teff_C[1:end-3], ribbon=σs_C, label="CUDA C")
plot!(nprocs_jl, Teff_jl, ribbon=σs_jl, label="Julia",foreground_color_legend = nothing,
     dpi=150,size=(600, 380))

png("julia_c_gpu_par_eff_lin.png")
