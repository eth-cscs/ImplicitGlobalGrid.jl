export nx_g, ny_g, nz_g, x_g, y_g, z_g, tic, toc

import MPI
@static if ENABLE_CUDA
    using CuArrays
end


macro nx_g()    esc(:( global_grid().nxyz_g[1] )) end
macro ny_g()    esc(:( global_grid().nxyz_g[2] )) end
macro nz_g()    esc(:( global_grid().nxyz_g[3] )) end
macro nx()      esc(:( global_grid().nxyz[1] )) end
macro ny()      esc(:( global_grid().nxyz[2] )) end
macro nz()      esc(:( global_grid().nxyz[3] )) end
macro coordx()  esc(:( global_grid().coords[1] )) end
macro coordy()  esc(:( global_grid().coords[2] )) end
macro coordz()  esc(:( global_grid().coords[3] )) end
macro olx()     esc(:( global_grid().overlaps[1] )) end
macro oly()     esc(:( global_grid().overlaps[2] )) end
macro olz()     esc(:( global_grid().overlaps[3] )) end
macro periodx() esc(:( convert(Bool, global_grid().periods[1]) )) end
macro periody() esc(:( convert(Bool, global_grid().periods[2]) )) end
macro periodz() esc(:( convert(Bool, global_grid().periods[3]) )) end

"Return the size of the global grid in dimension x."
nx_g()::GGInt = @nx_g()

"Return the size of the global grid in dimension y."
ny_g()::GGInt = @ny_g()

"Return the size of the global grid in dimension z."
nz_g()::GGInt = @nz_g()

"Return the size of A in the global grid in dimension x."
nx_g(A::GGArray)::GGInt = @nx_g() + (size(A,1)-@nx())

"Return the size of A in the global grid in dimension y."
ny_g(A::GGArray)::GGInt = @ny_g() + (size(A,2)-@ny())

"Return the size of A in the global grid in dimension x."
nz_g(A::GGArray)::GGInt = @nz_g() + (size(A,3)-@nz())

"Return the global x-coordinate for the element 'ix' in the local array 'A'."
function x_g(ix::Integer, dx::GGNumber, A::GGArray)::GGNumber
    x0 = 0.5*(@nx()-size(A,1))*dx;
    x  = (@coordx()*(@nx()-@olx()) + ix-1)*dx + x0;
    if @periodx()
        x = x - dx;                                     # The first cell of the global problem is a ghost cell; so, all must be shifted by dx to the left.
        if (x > (@nx_g()-1)*dx) x = x - @nx_g()*dx; end # It must not be (nx_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (x < 0)              x = x + @nx_g()*dx; end # ...
    end
    return x
end

"Return the global y-coordinate for the element 'iy' in the local array 'A'."
function y_g(iy::Integer, dy::GGNumber, A::GGArray)::GGNumber
    y0 = 0.5*(@ny()-size(A,2))*dy;
    y  = (@coordy()*(@ny()-@oly()) + iy-1)*dy + y0;
    if @periody()
        y = y - dy;
        if (y > (@ny_g()-1)*dy) y = y - @ny_g()*dy; end
        if (y < 0)              y = y + @ny_g()*dy; end
    end
    return y
end

"Return the global z-coordinate for the element 'iz' in the local array 'A'."
function z_g(iz::Integer, dz::GGNumber, A::GGArray)::GGNumber
    z0 = 0.5*(@nz()-size(A,3))*dz;
    z  = (@coordz()*(@nz()-@olz()) + iz-1)*dz + z0;
    if @periodz()
        z = z - dz;
        if (z > (@nz_g()-1)*dz) z = z - @nz_g()*dz; end
        if (z < 0)              z = z + @nz_g()*dz; end
    end
    return z
end

# Timing tools.
let
    global tic, toc
    t0 = nothing

    #TODO: see if global statement here needed!
    "Start chronometer once all processes have reached this point."
    tic() = ( check_initialized(); MPI.Barrier(comm()); t0 = time() )

    "Return the elapsed time from chronometer (since the last call to tic()) when all processes have reached this point."
    toc() = ( check_initialized(); MPI.Barrier(comm()); time() - t0 )
end
