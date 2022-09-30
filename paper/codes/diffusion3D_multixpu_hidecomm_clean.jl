using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(CUDA, Float64, 3)

@parallel function step!(T2,T,Ci,lam,dt,dx,dy,dz)
    @inn(T2) = @inn(T) + dt*(
        lam*@inn(Ci)*(@d2_xi(T)/dx^2 + 
                      @d2_yi(T)/dy^2 + 
                      @d2_zi(T)/dz^2 ) )
    return
end

function diffusion3D()
    # Physics
    lam      = 1.0           #Thermal conductivity
    c0       = 2.0           #Heat capacity
    lx=ly=lz = 1.0           #Domain length x|y|z

    # Numerics
    nx=ny=nz = 512           #Nb gridpoints x|y|z
    nt       = 100           #Nb time steps
    me,      = init_global_grid(nx, ny, nz)
    dx       = lx/(nx_g()-1) #Space step in x
    dy       = ly/(ny_g()-1) #Space step in y
    dz       = lz/(nz_g()-1) #Space step in z

    # Initial conditions
    T  = @ones(nx,ny,nz).*1.7 #Temperature
    T2 = copy(T)              #Temperature (2nd)
    Ci = @ones(nx,ny,nz)./c0  #1/Heat capacity

    # Time loop
    dt = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/6.1
    for it = 1:nt
        @hide_communication (16, 2, 2) begin
            @parallel step!(T2,T,Ci,lam,dt,dx,dy,dz)
            update_halo!(T2)
        end
        T, T2 = T2, T
    end

    finalize_global_grid()
end

diffusion3D()