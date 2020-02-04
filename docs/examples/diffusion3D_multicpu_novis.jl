using ImplicitGlobalGrid

@views d_xa(A) = A[2:end  , :     , :     ] .- A[1:end-1, :     , :     ];
@views d_xi(A) = A[2:end  ,2:end-1,2:end-1] .- A[1:end-1,2:end-1,2:end-1];
@views d_ya(A) = A[ :     ,2:end  , :     ] .- A[ :     ,1:end-1, :     ];
@views d_yi(A) = A[2:end-1,2:end  ,2:end-1] .- A[2:end-1,1:end-1,2:end-1];
@views d_za(A) = A[ :     , :     ,2:end  ] .- A[ :     , :     ,1:end-1];
@views d_zi(A) = A[2:end-1,2:end-1,2:end  ] .- A[2:end-1,2:end-1,1:end-1];
@views  inn(A) = A[2:end-1,2:end-1,2:end-1]

@views function diffusion3D()
    # Physics
    lam        = 1.0;                                       # Thermal conductivity
    cp_min     = 1.0;                                       # Minimal heat capacity
    lx, ly, lz = 10.0, 10.0, 10.0;                          # Length of domain in dimensions x, y an z.

    # Numerics
    nx, ny, nz = 128, 128, 128;                             # Number of gridpoints dimensions x, y an z.
    nt         = 10000;                                     # Number of time steps
    me, dims   = init_global_grid(nx, ny, nz);              # Initialize the implicit global grid
    dx         = lx/(nx_g()-1);                             # Space step in dimension x
    dy         = ly/(ny_g()-1);                             # ...        in dimension y
    dz         = lz/(nz_g()-1);                             # ...        in dimension z

    # Array initializations
    T     = zeros(nx,   ny,   nz  );
    Cp    = zeros(nx,   ny,   nz  );
    dTedt = zeros(nx-2, ny-2, nz-2);
    qx    = zeros(nx-1, ny-2, nz-2);
    qy    = zeros(nx-2, ny-1, nz-2);
    qz    = zeros(nx-2, ny-2, nz-1);

    # Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
    Cp .= cp_min .+ [5*exp(-((x_g(ix,dx,Cp)-lx/1.5))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) +
                     5*exp(-((x_g(ix,dx,Cp)-lx/3.0))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]
    T  .= [100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
            50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]

    # Time loop
    dt = min(dx*dx,dy*dy,dz*dz)*cp_min/lam/8.1;                                               # Time step for the 3D Heat diffusion
    for it = 1:nt
        qx    .= -lam.*d_xi(T)./dx;                                                           # Fourier's law of heat conduction: q_x   = -λ δT/δx
        qy    .= -lam.*d_yi(T)./dy;                                                           # ...                               q_y   = -λ δT/δy
        qz    .= -lam.*d_zi(T)./dz;                                                           # ...                               q_z   = -λ δT/δz
        dTedt .= 1.0./inn(Cp).*(-d_xa(qx)./dx .- d_ya(qy)./dy .- d_za(qz)./dz);               # Conservation of energy:           δT/δt = 1/cₚ (-δq_x/δx - δq_y/dy - δq_z/dz)
        T[2:end-1,2:end-1,2:end-1] .= inn(T) .+ dt.*dTedt;                                    # Update of temperature             T_new = T_old + δT/δt
        update_halo!(T);                                                                      # Update the halo of T
    end

    finalize_global_grid();                                                                   # Finalize the implicit global grid
end

diffusion3D()
