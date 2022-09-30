# Activate the current environment and load all packages
using Pkg
Pkg.activate(@__DIR__)

using DelimitedFiles, Statistics

nexp = 20

data_r = readdlm("out_diff3D_pareff_noMPI_1.txt")
Teff_ref = mean(data_r[:,end])

data   = readdlm("out_diff3D_pareff.txt")
# data   = readdlm("out_diff3D_pareff_nordma.txt")
nprocs = convert(Vector{Int64},data[1:nexp:end,1])
Teff   = zeros(Float64,length(nprocs),3)
for i ∈ 1:length(nprocs)
     local range = nexp*(i-1)+1 : nexp*i
     Teff[i,1] =  median(data[range,end]./Teff_ref)
     # 95% of confidence interval
     tmp       = sort(data[range,end]./Teff_ref; rev=true)
     Teff[i,2] = tmp[5]  # 5th rank  <=   (n-1.96*n^(1/2))/2 =  5.617306764100412
     Teff[i,3] = tmp[16] # 16th rank <= 1+(n+1.96*n^(1/2))/2 = 15.382693235899588
     # previous naive approach
     # Teff[i,2] = minimum(data[range,end]./Teff_ref)
     # Teff[i,3] = maximum(data[range,end]./Teff_ref)
end
σs = Teff[:,3] .- Teff[:,2]

# Weak scaling parallel efficiency data on Piz Daint
using Plots

ax_log = false # choose between log or linear x-axis scale

default(fontfamily="Computer Modern", linewidth=3,  markershape=:circle, markersize=4,
        framestyle=:box, fillalpha=0.4)
scalefontsizes(); scalefontsizes(1.4)

if ax_log
xtick_powers = 0:1:12
plot(xlabel="Number of GPUs", ylabel="Parallel efficiency", xscale=:log,
     xticks=(2 .^ xtick_powers, "\$2^{" .* string.(xtick_powers) .* "}\$"), legend=false)
plot!(nprocs, Teff[:,1], ribbon=σs,dpi=150)
else
# xtick_lin = (1,512,1024,2048,4096,5120)
xtick_lin = (1,64,216,512,1000,2197)
plot(xlabel="Number of GPUs", ylabel="Parallel efficiency",
     xticks=(xtick_lin, string.(xtick_lin)), legend=false)
plot!(nprocs[[1,4,6,7,8,9]], Teff[[1,4,6,7,8,9],1], ribbon=σs,dpi=150)
end

ax_log ? png("julia_gpu_par_eff2.png") : png("julia_gpu_par_eff.png")
# ax_log ? png("julia_gpu_par_eff2_nordma.png") : png("julia_gpu_par_eff_nordma.png")
