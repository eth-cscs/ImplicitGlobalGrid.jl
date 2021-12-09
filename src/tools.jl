export nx_g, ny_g, nz_g, x_g, y_g, z_g, tic, toc

import MPI
using CUDA


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

"""
    nx_g()

Return the size of the global grid in dimension x.
"""
nx_g()::GGInt = @nx_g()

"""
    ny_g()

Return the size of the global grid in dimension y.
"""
ny_g()::GGInt = @ny_g()

"""
    nz_g()

Return the size of the global grid in dimension z.
"""
nz_g()::GGInt = @nz_g()

"""
    nx_g(A)

Return the size of array `A` in the global grid in dimension x.
"""
nx_g(A::GGArray)::GGInt = @nx_g() + (size(A,1)-@nx())

"""
    ny_g(A)

Return the size of array `A` in the global grid in dimension y.
"""
ny_g(A::GGArray)::GGInt = @ny_g() + (size(A,2)-@ny())

"""
    nz_g(A)

Return the size of array `A` in the global grid in dimension z.
"""
nz_g(A::GGArray)::GGInt = @nz_g() + (size(A,3)-@nz())

"""
    x_g(ix, dx, A)

Return the global x-coordinate for the element `ix` in the local array `A` (`dx` is the space step between the elements).

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> lx=4; nx=3; ny=3; nz=3;

julia> init_global_grid(nx, ny, nz);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> dx = lx/(nx_g()-1)
2.0

julia> A  = zeros(nx,ny,nz);

julia> Vx = zeros(nx+1,ny,nz);

julia> [x_g(ix, dx, A) for ix=1:size(A, 1)]
3-element Array{Float64,1}:
 0.0
 2.0
 4.0

julia> [x_g(ix, dx, Vx) for ix=1:size(Vx, 1)]
4-element Array{Float64,1}:
 -1.0
  1.0
  3.0
  5.0
```
"""
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

"""
    y_g(iy, dy, A)

Return the global y-coordinate for the element `iy` in the local array `A` (`dy` is the space step between the elements).

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> ly=4; nx=3; ny=3; nz=3;

julia> init_global_grid(nx, ny, nz);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> dy = ly/(ny_g()-1)
2.0

julia> A  = zeros(nx,ny,nz);

julia> Vy = zeros(nx,ny+1,nz);

julia> [y_g(iy, dy, A) for iy=1:size(A, 1)]
3-element Array{Float64,1}:
 0.0
 2.0
 4.0

julia> [y_g(iy, dy, Vy) for iy=1:size(Vy, 2)]
4-element Array{Float64,1}:
 -1.0
  1.0
  3.0
  5.0
```
"""
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

"""
    z_g(iz, dz, A)

Return the global z-coordinate for the element `iz` in the local array `A` (`dz` is the space step between the elements).

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> lz=4; nx=3; ny=3; nz=3;

julia> init_global_grid(nx, ny, nz);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> dz = lz/(nz_g()-1)
2.0

julia> A  = zeros(nx,ny,nz);

julia> Vz = zeros(nx,ny,nz+1);

julia> [z_g(iz, dz, A) for iz=1:size(A, 1)]
3-element Array{Float64,1}:
 0.0
 2.0
 4.0

julia> [z_g(iz, dz, Vz) for iz=1:size(Vz, 3)]
4-element Array{Float64,1}:
 -1.0
  1.0
  3.0
  5.0
```
"""
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
@doc """
    tic()

Start chronometer once all processes have reached this point.

!!! warning
    The chronometer may currently add an overhead of multiple 10th of miliseconds at the first usage.

See also: [`toc`](@ref)
"""
tic

@doc """
    toc()

Return the elapsed time from chronometer (since the last call to `tic`) when all processes have reached this point.

!!! warning
    The chronometer may currently add an overhead of multiple 10th of miliseconds at the first usage.

See also: [`tic`](@ref)
"""
toc

let
    global tic, toc
    t0 = nothing

    tic() = ( check_initialized(); MPI.Barrier(comm()); t0 = time() )
    toc() = ( check_initialized(); MPI.Barrier(comm()); time() - t0 )
end
