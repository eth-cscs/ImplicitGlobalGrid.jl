using ImplicitGlobalGrid, CUDA, Plots
#(...)

@views function diffusion3D()
    # Physics
    #(...)

    # Numerics
    #(...)
    me, dims   = init_global_grid(nx, ny, nz);              # Initialize the implicit global grid
    #(...)

    # Array initializations
    #(...)

    # Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
    #(...)

    # Preparation of visualisation
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    nx_v = (nx-2)*dims[1];
    ny_v = (ny-2)*dims[2];
    nz_v = (nz-2)*dims[3];
    T_v  = zeros(nx_v, ny_v, nz_v);
    T_nohalo = zeros(nx-2, ny-2, nz-2);

    # Time loop
    #(...)
    for it = 1:nt
        if mod(it, 1000) == 1                                                                 # Visualize only every 1000th time step
            T_nohalo .= T[2:end-1,2:end-1,2:end-1];                                           # Copy data to CPU removing the halo.
            gather!(T_nohalo, T_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
            if (me==0) heatmap(transpose(T_v[:,ny_v÷2,:]), aspect_ratio=1); frame(anim); end  # Visualize it on process 0.
        end
        #(...)
    end

    # Postprocessing
    if (me==0) gif(anim, "diffusion3D.gif", fps = 15) end                                     # Create a gif movie on process 0.
    if (me==0) mp4(anim, "diffusion3D.mp4", fps = 15) end                                     # Create a mp4 movie on process 0.
    finalize_global_grid();                                                                   # Finalize the implicit global grid
end

diffusion3D()
