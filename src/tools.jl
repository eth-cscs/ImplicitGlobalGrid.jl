export nx_g, ny_g, nz_g, x_g, y_g, z_g, ix_g, iy_g, iz_g, extents, extents_g, metagrid, tic, toc

macro nx_g()    esc(:( global_grid().nxyz_g[1] )) end
macro ny_g()    esc(:( global_grid().nxyz_g[2] )) end
macro nz_g()    esc(:( global_grid().nxyz_g[3] )) end
macro nx()      esc(:( global_grid().nxyz[1] )) end
macro ny()      esc(:( global_grid().nxyz[2] )) end
macro nz()      esc(:( global_grid().nxyz[3] )) end
macro nxyz()    esc(:( (Int.(global_grid().nxyz)...,) )) end
macro coordx()  esc(:( global_grid().coords[1] )) end
macro coordy()  esc(:( global_grid().coords[2] )) end
macro coordz()  esc(:( global_grid().coords[3] )) end
macro coords()  esc(:( (Int.(global_grid().coords)...,) )) end
macro dimx()   esc(:( global_grid().dims[1] )) end
macro dimy()   esc(:( global_grid().dims[2] )) end
macro dimz()   esc(:( global_grid().dims[3] )) end
macro dims()    esc(:( (Int.(global_grid().dims)...,) )) end
macro olx()     esc(:( global_grid().overlaps[1] )) end
macro oly()     esc(:( global_grid().overlaps[2] )) end
macro olz()     esc(:( global_grid().overlaps[3] )) end
macro ols()     esc(:( (Int.(global_grid().overlaps)...,) )) end
macro periodx() esc(:( convert(Bool, global_grid().periods[1]) )) end
macro periody() esc(:( convert(Bool, global_grid().periods[2]) )) end
macro periodz() esc(:( convert(Bool, global_grid().periods[3]) )) end
macro periods() esc(:( (Bool.(global_grid().periods)...,) )) end
macro halowidthx() esc(:( global_grid().halowidths[1] )) end
macro halowidthy() esc(:( global_grid().halowidths[2] )) end
macro halowidthz() esc(:( global_grid().halowidths[3] )) end
macro halowidths() esc(:( global_grid().halowidths )) end


"""
    nx_g()

Return the size of the global grid in dimension x.
"""
nx_g() = @nx_g()


"""
    ny_g()

Return the size of the global grid in dimension y.
"""
ny_g() = @ny_g()


"""
    nz_g()

Return the size of the global grid in dimension z.
"""
nz_g() = @nz_g()


"""
    nx_g(A)

Return the size of array `A` in the global grid in dimension x.
"""
nx_g(A::AbstractArray) = @nx_g() + (size(A,1)-@nx())


"""
    ny_g(A)

Return the size of array `A` in the global grid in dimension y.
"""
ny_g(A::AbstractArray) = @ny_g() + (size(A,2)-@ny())


"""
    nz_g(A)

Return the size of array `A` in the global grid in dimension z.
"""
nz_g(A::AbstractArray) = @nz_g() + (size(A,3)-@nz())


"""
    x_g(ix, dx, A)
    x_g(ix, dx, A; wrap_periodic)

Return the global x-coordinate for the index `ix` in the local array `A` (`dx` is the space step between the elements).

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the coordinate at the periodic boundaries (default: `true`).

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
3-element Vector{Float64}:
 0.0
 2.0
 4.0

julia> [x_g(ix, dx, Vx) for ix=1:size(Vx, 1)]
4-element Vector{Float64}:
 -1.0
  1.0
  3.0
  5.0

julia> finalize_global_grid()
```
"""
x_g(ix::Integer, dx::Real, A::AbstractArray; wrap_periodic::Bool=true) =_x_g(ix, dx, size(A,1), wrap_periodic)

function _x_g(ix::Integer, dx::Real, nx_A::Integer, wrap_periodic::Bool, coordx::Integer=@coordx(), dimx::Integer=@dimx())
    nx_g  = dimx*(@nx()-@olx()) + @olx()*(@periodx()==0)
    x0_g  = @periodx() ? -dx*@halowidthx() : dx*0 # The first cells of the global problem are halo cells; so, all must be shifted by dx to the left.
    x0    = 0.5*(@nx()-nx_A)*dx
    x     = (coordx*(@nx()-@olx()) + ix-1)*dx + x0 + x0_g
    if @periodx() && wrap_periodic
        if (x > (nx_g-1)*dx) x = x - nx_g*dx; end # It must not be (nx_g()-1)*dx as the distance between the local problems (1*dx) must also be taken into account!
        if (x < 0)           x = x + nx_g*dx; end # ...
    end
    return x
end


"""
    y_g(iy, dy, A)
    y_g(iy, dy, A; wrap_periodic)

Return the global y-coordinate for the index `iy` in the local array `A` (`dy` is the space step between the elements).

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the coordinate at the periodic boundaries (default: `true`).

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
3-element Vector{Float64}:
 0.0
 2.0
 4.0

julia> [y_g(iy, dy, Vy) for iy=1:size(Vy, 2)]
4-element Vector{Float64}:
 -1.0
  1.0
  3.0
  5.0

julia> finalize_global_grid()
```
"""
y_g(iy::Integer, dy::Real, A::AbstractArray; wrap_periodic::Bool=true) =_y_g(iy, dy, size(A,2), wrap_periodic)

function _y_g(iy::Integer, dy::Real, ny_A::Integer, wrap_periodic::Bool, coordy::Integer=@coordy(), dimy::Integer=@dimy())
    ny_g  = dimy*(@ny()-@oly()) + @oly()*(@periody()==0)
    y0_g  = @periody() ? -dy*@halowidthy() : dy*0 # The first cells of the global problem are halo cells; so, all must be shifted by dy to the left.
    y0    = 0.5*(@ny()-ny_A)*dy
    y     = (coordy*(@ny()-@oly()) + iy-1)*dy + y0 + y0_g
    if @periody() && wrap_periodic
        if (y > (ny_g-1)*dy) y = y - ny_g*dy; end # It must not be (ny_g()-1)*dy as the distance between the local problems (1*dy) must also be taken into account!
        if (y < 0)           y = y + ny_g*dy; end # ...
    end
    return y
end


"""
    z_g(iz, dz, A)
    z_g(iz, dz, A; wrap_periodic)

Return the global z-coordinate for the index `iz` in the local array `A` (`dz` is the space step between the elements).

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the coordinate at the periodic boundaries (default: `true`).

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
3-element Vector{Float64}:
 0.0
 2.0
 4.0

julia> [z_g(iz, dz, Vz) for iz=1:size(Vz, 3)]
4-element Vector{Float64}:
 -1.0
  1.0
  3.0
  5.0

julia> finalize_global_grid()
```
"""
z_g(iz::Integer, dz::Real, A::AbstractArray; wrap_periodic::Bool=true) =_z_g(iz, dz, size(A,3), wrap_periodic)

function _z_g(iz::Integer, dz::Real, nz_A::Integer, wrap_periodic::Bool, coordz::Integer=@coordz(), dimz::Integer=@dimz())
    nz_g  = dimz*(@nz()-@olz()) + @olz()*(@periodz()==0)
    z0_g  = @periodz() ? -dz*@halowidthz() : dz*0 # The first cells of the global problem are halo cells; so, all must be shifted by dz to the left.
    z0    = 0.5*(@nz()-nz_A)*dz
    z     = (coordz*(@nz()-@olz()) + iz-1)*dz + z0 + z0_g
    if @periodz() && wrap_periodic
        if (z > (nz_g-1)*dz) z = z - nz_g*dz; end # It must not be (nz_g()-1)*dz as the distance between the local problems (1*dz) must also be taken into account!
        if (z < 0)           z = z + nz_g*dz; end # ...
    end
    return z
end


"""
    ix_g(ix, A)
    ix_g(ix, A; wrap_periodic)

Return the global x-index for the local index `ix` in the local array `A`.

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the index at the periodic boundaries (default: `true`). If `wrap_periodic=false`, the global index can be negative or bigger than the size of the global grid.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> nx=5; ny=3; nz=3;

julia> init_global_grid(nx, ny, nz, periodx=1);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> A  = zeros(nx,ny,nz);

julia> Vx = zeros(nx+1,ny,nz);

julia> [ix_g(ix, A) for ix=1:size(A, 1)]
5-element Vector{Int64}:
 3
 1
 2
 3
 1

julia> [ix_g(ix, Vx) for ix=1:size(Vx, 1)]
6-element Vector{Int64}:
 3
 1
 2
 3
 1
 2

julia> finalize_global_grid()
```
"""
ix_g(ix::Integer, A::AbstractArray; wrap_periodic::Bool=true) = _ix_g(ix, size(A,1), wrap_periodic)


function _ix_g(ix::Integer, nx_A::Integer, wrap_periodic::Bool, coordx::Integer=@coordx(), dimx::Integer=@dimx())
    nx_g  = dimx*(@nx()-@olx()) + @olx()*(@periodx()==0)
    olx_A = @olx() + (nx_A-@nx())
    ix0_g = @periodx() ? 0 : olx_A÷2
    ix0   = -olx_A÷2
    ix    = coordx*(@nx()-olx_A) + ix + ix0 + ix0_g
    if wrap_periodic && @periodx()
        if (ix > nx_g) ix = ix - nx_g; end
        if (ix < 1)    ix = ix + nx_g; end
    end
    return ix
end


"""
    iy_g(iy, A)
    iy_g(iy, A; wrap_periodic)

Return the global y-index for the local index `iy` in the local array `A`.

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the index at the periodic boundaries (default: `true`). If `wrap_periodic=false`, the global index can be negative or bigger than the size of the global grid.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> nx=3; ny=5; nz=3;

julia> init_global_grid(nx, ny, nz, periody=1);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> A  = zeros(nx,ny,nz);

julia> Vy = zeros(nx,ny+1,nz);

julia> [iy_g(iy, A) for iy=1:size(A, 2)]
5-element Vector{Int64}:
 3
 1
 2
 3
 1

julia> [iy_g(iy, Vy) for iy=1:size(Vy, 2)]
6-element Vector{Int64}:
 3
 1
 2
 3
 1
 2

julia> finalize_global_grid()
```
"""
iy_g(iy::Integer, A::AbstractArray; wrap_periodic::Bool=true) = _iy_g(iy, size(A,2), wrap_periodic)

function _iy_g(iy::Integer, ny_A::Integer, wrap_periodic::Bool, coordy::Integer=@coordy(), dimy::Integer=@dimy())
    ny_g  = dimy*(@ny()-@oly()) + @oly()*(@periody()==0)
    oly_A = @oly() + (ny_A-@ny())
    iy0_g = @periody() ? 0 : oly_A÷2
    iy0   = -oly_A÷2
    iy    = coordy*(@ny()-oly_A) + iy + iy0 + iy0_g
    if wrap_periodic && @periody()
        if (iy > ny_g) iy = iy - ny_g; end
        if (iy < 1)    iy = iy + ny_g; end
    end
    return iy
end


"""
    iz_g(iz, A)
    iz_g(iz, A; wrap_periodic)

Return the global z-index for the local index `iz` in the local array `A`.

# Keyword arguments
- `wrap_periodic::Bool=true`: whether to wrap the index at the periodic boundaries (default: `true`). If `wrap_periodic=false`, the global index can be negative or bigger than the size of the global grid.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> nx=3; ny=3; nz=5;

julia> init_global_grid(nx, ny, nz, periodz=1);
Global grid: 3x3x3 (nprocs: 1, dims: 1x1x1)

julia> A  = zeros(nx,ny,nz);

julia> Vz = zeros(nx,ny,nz+1);

julia> [iz_g(iz, A) for iz=1:size(A, 3)]
5-element Vector{Int64}:
 3
 1
 2
 3
 1

julia> [iz_g(iz, A; wrap_periodic=false) for iz=1:size(A, 3)]
5-element Vector{Int64}:
 0
 1
 2
 3
 4

julia> [iz_g(iz, Vz) for iz=1:size(Vz, 3)]
6-element Vector{Int64}:
 3
 1
 2
 3
 1
 2

julia> [iz_g(iz, Vz; wrap_periodic=false) for iz=1:size(Vz, 3)]
6-element Vector{Int64}:
 0
 1
 2
 3
 4
 5

julia> finalize_global_grid()
```
"""
iz_g(iz::Integer, A::AbstractArray; wrap_periodic::Bool=true) = _iz_g(iz, size(A,3), wrap_periodic)

function _iz_g(iz::Integer, nz_A::Integer, wrap_periodic::Bool, coordz::Integer=@coordz(), dimz::Integer=@dimz())
    nz_g  = dimz*(@nz()-@olz()) + @olz()*(@periodz()==0)
    olz_A = @olz() + (nz_A-@nz())
    iz0_g = @periodz() ? 0 : olz_A÷2
    iz0   = -olz_A÷2
    iz    = coordz*(@nz()-olz_A) + iz + iz0 + iz0_g
    if wrap_periodic && @periodz()
        if (iz > nz_g) iz = iz - nz_g; end
        if (iz < 1)    iz = iz + nz_g; end
    end
    return iz
end

"""
    extents(; fix_global_boundaries, coords)
    extents(A; fix_global_boundaries, coords)
    extents(overlaps; fix_global_boundaries, coords)
    extents(A, overlaps; fix_global_boundaries, coords)

Return the local extents in each dimension of the array `A` or the local extents of the base grid if `A` is not provided (return type: tuple of ranges).

# Arguments
- `overlaps::Integer|Tuple{Int,Int,Int}`: the overlap of the "extent" with the neighboring processes' extents in each dimension; the overlaps chosen cannot be bigger than the actual overlaps on the global grid of `A` or of the base grid, respectively. To obtain the extents as required by VTK, set `overlaps=1`. The default is the actual full overlaps on the global grid (i.e., the extents are simply the full ranges of the array or of the base grid, respectively).

# Keyword arguments
- `fix_global_boundaries::Bool=true`: by default, the extents are fixed at the global boundaries to include them on all sides (attention, the extents are not of equal size for all processes in this case). If `fix_global_boundaries=false`, the extents are not fixed at the global boundaries and the size of the extents is equal for all processes.
- `coords::Tuple{Int,Int,Int}`: the coordinates of the process for which the local extents is requested. The default is the coordinates of the current process.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> nx=5; ny=5; nz=7;

julia> init_global_grid(nx, ny, nz; periodz=1);
Global grid: 5x5x5 (nprocs: 1, dims: 1x1x1)

julia> extents()
(1:5, 1:5, 2:6)

julia> extents(; fix_global_boundaries=false)
(1:5, 1:5, 1:7)

julia> extents(0)
(1:5, 1:5, 2:6)

julia> extents(0; fix_global_boundaries=false)
(2:4, 2:4, 2:6)

julia> extents(1) # The extents as required by VTK
(1:5, 1:5, 2:6)

julia> Vx = zeros(nx+1,ny,nz);

julia> extents(Vx, 1) # The extents as required by VTK
(1:6, 1:5, 2:6)

julia> Vx_IO = view(Vx, extents(Vx, 1)...);

julia> summary(Vx_IO)
"6×5×5 view(::Array{Float64, 3}, 1:6, 1:5, 2:6) with eltype Float64"

julia> finalize_global_grid()
```
"""
function extents(; fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    extents = (1:@nx(), 1:@ny(), 1:@nz())
    return _adjust_extents(extents, @nxyz(), @ols(), coords, dims, fix_global_boundaries)
end

function extents(A::AbstractArray; fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    # ol_A = @olx() + (size(A,1)-@nx()), @oly() + (size(A,2)-@ny()), @olz() + (size(A,3)-@nz())
    ol_A = @ols() .+ (size(A) .- @nxyz())
    extents = (1:size(A,1), 1:size(A,2), 1:size(A,3))
    return _adjust_extents(extents, size(A), ol_A, coords, dims, fix_global_boundaries)
end

function extents(overlaps::Union{Integer,Tuple{Int,Int,Int}}; fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    overlaps = isa(overlaps, Integer) ? (Int(overlaps), Int(overlaps), Int(overlaps)) : overlaps
    # if (overlaps[1] > @olx()) || (overlaps[2] > @oly()) || (overlaps[3] > @olz()) @ArgumentError("The overlaps chosen cannot be bigger than the actual overlaps on the global grid.") end
    if any(overlaps .> @ols()) @ArgumentError("The overlaps chosen cannot be bigger than the actual overlaps on the global grid.") end
    # bx, by, bz = (@olx()-overlaps[1]) ÷ 2, (@oly()-overlaps[2]) ÷ 2, (@olz()-overlaps[3]) ÷ 2
    bx, by, bz = (@ols() .- overlaps) .÷ 2
    # ex, ey, ez = cld(@olx()-overlaps[1], 2), cld(@oly()-overlaps[2], 2), cld(@olz()-overlaps[3], 2)
    ex, ey, ez = cld.(@ols() .- overlaps, 2)
    extents = (1+bx:@nx()-ex, 1+by:@ny()-ey, 1+bz:@nz()-ez)
    return _adjust_extents(extents, @nxyz(), @ols(), coords, dims, fix_global_boundaries)
end

function extents(A::AbstractArray, overlaps::Union{Integer,Tuple{Int,Int,Int}}; fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    overlaps = isa(overlaps, Integer) ? (Int(overlaps), Int(overlaps), Int(overlaps)) : overlaps
    # ol_A = @olx() + (size(A,1)-@nx()), @oly() + (size(A,2)-@ny()), @olz() + (size(A,3)-@nz())
    ol_A = @ols() .+ (size(A) .- @nxyz())
    # if (overlaps[1] > ol_A[1]) || (overlaps[2] > ol_A[2]) || (overlaps[3] > ol_A[3]) @ArgumentError("The overlaps chosen cannot be bigger than the actual overlaps on the global grid.") end
    if any(overlaps .> ol_A) @ArgumentError("The overlaps chosen cannot be bigger than the actual overlaps on the global grid.") end
    # bx, by, bz = (ol_A[1]-overlaps[1]) ÷ 2, (ol_A[2]-overlaps[2]) ÷ 2, (ol_A[3]-overlaps[3]) ÷ 2
    bx, by, bz = (ol_A .- overlaps) .÷ 2
    # ex, ey, ez = cld(ol_A[1]-overlaps[1], 2), cld(ol_A[2]-overlaps[2], 2), cld(ol_A[3]-overlaps[3], 2)
    ex, ey, ez = cld.(ol_A .- overlaps, 2)
    extents = (1+bx:size(A,1)-ex, 1+by:size(A,2)-ey, 1+bz:size(A,3)-ez)
    return _adjust_extents(extents, size(A), ol_A, coords, dims, fix_global_boundaries)
end

# function _adjust_extents(extents::Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}, nxyz_A::Tuple{Int,Int,Int}, coords::Tuple{Int,Int,Int}, dims::Tuple{Int,Int,Int}, fix_global_boundaries::Bool)
#     @show extents
#     if fix_global_boundaries
#         extents_new = ( if !(@periods()[i])
#                             if coords[i] == 0
#                                 1:extents[i].last
#                             end
#                             if coords[i] == @dims()[i]-1
#                                 extents[i].first:nxyz_A[i]
#                             end
#                         else
#                             extents[i]
#                         end
#                         for i in 1:3 )
#     else
#         extents_new = extents
#     end
#     return extents_new
# end

function _adjust_extents(extents::Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}, nxyz_A::Tuple{Int,Int,Int}, ol_A::Tuple{Int,Int,Int}, coords::Tuple{Int,Int,Int}, dims::Tuple{Int,Int,Int}, fix_global_boundaries::Bool)
    extents = [extents...]
    if fix_global_boundaries
        # b_g = (ol_A[1] ÷ 2, ol_A[2] ÷ 2, ol_A[3] ÷ 2)
        b_g = ol_A .÷ 2
        # e_g = (cld(ol_A[1], 2), cld(ol_A[2], 2), cld(ol_A[3], 2))
        e_g = cld.(ol_A, 2)
        for i in 1:3
            if @periods()[i]
                if coords[i] == 0
                    extents[i] = 1+b_g[i]:last(extents[i])
                end
                if coords[i] == dims[i]-1
                    extents[i] = first(extents[i]):nxyz_A[i]-e_g[i]
                end
            else
                if coords[i] == 0
                    extents[i] = 1:last(extents[i])
                end
                if coords[i] == dims[i]-1
                    extents[i] = first(extents[i]):nxyz_A[i]
                end
            end
        end
    end
    return (extents...,)
end

"""
    extents_g(; dxyz, fix_global_boundaries, coords)
    extents_g(A; dxyz, fix_global_boundaries, coords)
    extents_g(overlaps; dxyz, fix_global_boundaries, coords)
    extents_g(A, overlaps; dxyz, fix_global_boundaries, coords)

Return the global extents in each dimension of the array `A` or the global extents of the base grid if `A` is not provided (return type: tuple of ranges); if `dxyz` is set, global Cartesian coordinates extents are returned, else global index extents are returned.

# Arguments
- `overlaps::Integer|Tuple{Int,Int,Int}`: the overlap of the "extent" with the neighboring processes' extents in each dimension; the overlaps chosen cannot be bigger than the actual overlaps on the global grid of `A` or of the base grid, respectively. To obtain the extents as required by VTK, set `overlaps=1`. The default is the actual full overlaps on the global grid.

# Keyword arguments
- `dxyz::Tuple{Real,Real,Real}`: the space step between the elements in each dimension if global Cartesian coordinates are desired.
- `fix_global_boundaries::Bool=true`: by default, the extents are fixed at the global boundaries to include them on all sides (attention, the extents are not of equal size for all processes in this case). If `fix_global_boundaries=false`, the extents are not fixed at the global boundaries and the size of the extents is equal for all processes.
- `coords::Tuple{Int,Int,Int}`: the coordinates of the process for which the global extents is requested. The default is the coordinates of the current process.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> lx=8; ly=8; lz=8; nx=5; ny=5; nz=7;

julia> init_global_grid(nx, ny, nz; periodz=1);
Global grid: 5x5x5 (nprocs: 1, dims: 1x1x1)

julia> extents_g()
(1:5, 1:5, 1:5)

julia> extents_g(; fix_global_boundaries=false)
(1:5, 1:5, 0:6)

julia> extents_g(0)
(1:5, 1:5, 1:5)

julia> extents_g(0; fix_global_boundaries=false)
(2:4, 2:4, 1:5)

julia> extents_g(1) # The extents as required by VTK
(1:5, 1:5, 1:5)

julia> dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)
(2.0, 2.0, 2.0)

julia> extents_g(; dxyz=(dx, dy, dz))
(0.0:2.0:8.0, 0.0:2.0:8.0, 1.0:2.0:7.0)

julia> extents_g(; dxyz=(dx, dy, dz), fix_global_boundaries=false)
(0.0:2.0:8.0, 0.0:2.0:8.0, -2.0:2.0:10.0)

julia> extents_g(0; dxyz=(dx, dy, dz), fix_global_boundaries=false)
(2.0:2.0:6.0, 2.0:2.0:6.0, 0.0:2.0:8.0)

julia> extents_g(1; dxyz=(dx, dy, dz)) # The Cartesian coordinates corresponding to the extents required by VTK
(0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:2.0:8.0)

julia> finalize_global_grid()
```
"""
function extents_g(; dxyz::Union{Nothing,Tuple{Real,Real,Real}}=nothing, fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    extents_l = extents(; fix_global_boundaries=fix_global_boundaries, coords=coords, dims=dims)
    return extents_g(extents_l, @nxyz(), dxyz, coords, dims)
end

function extents_g(A::AbstractArray; dxyz::Union{Nothing,Tuple{Real,Real,Real}}=nothing, fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    extents_l = extents(A; fix_global_boundaries=fix_global_boundaries, coords=coords, dims=dims)
    return extents_g(extents_l, size(A), dxyz, coords, dims)
end

function extents_g(overlaps::Union{Integer,Tuple{Int,Int,Int}}; dxyz::Union{Nothing,Tuple{Real,Real,Real}}=nothing, fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    extents_l = extents(overlaps; fix_global_boundaries=fix_global_boundaries, coords=coords, dims=dims)
    return extents_g(extents_l, @nxyz(), dxyz, coords, dims)
end

function extents_g(A::AbstractArray, overlaps::Union{Integer,Tuple{Int,Int,Int}}; dxyz::Union{Nothing,Tuple{Real,Real,Real}}=nothing, fix_global_boundaries::Bool=true, coords::Tuple{Int,Int,Int}=@coords(), dims::Tuple{Int,Int,Int}=@dims())
    extents_l = extents(A, overlaps; fix_global_boundaries=fix_global_boundaries, coords=coords, dims=dims)
    return extents_g(extents_l, size(A), dxyz, coords, dims)
end

function extents_g(extents::Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}, nxyz_A::Tuple{Int,Int,Int}, dxyz::Nothing, coords::Tuple{Int,Int,Int}, dims::Tuple{Int,Int,Int})
    return (_ix_g(first(extents[1]), nxyz_A[1], false, coords[1], dims[1]) : _ix_g(last(extents[1]), nxyz_A[1], false, coords[1], dims[1]),
            _iy_g(first(extents[2]), nxyz_A[2], false, coords[2], dims[2]) : _iy_g(last(extents[2]), nxyz_A[2], false, coords[2], dims[2]),
            _iz_g(first(extents[3]), nxyz_A[3], false, coords[3], dims[3]) : _iz_g(last(extents[3]), nxyz_A[3], false, coords[3], dims[3]))
end

function extents_g(extents::Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}, nxyz_A::Tuple{Int,Int,Int}, dxyz::Tuple{Real,Real,Real}, coords::Tuple{Int,Int,Int}, dims::Tuple{Int,Int,Int})
    return (_x_g(first(extents[1]), dxyz[1], nxyz_A[1], false, coords[1], dims[1]) : dxyz[1] : _x_g(last(extents[1]), dxyz[1], nxyz_A[1], false, coords[1], dims[1]),
            _y_g(first(extents[2]), dxyz[2], nxyz_A[2], false, coords[2], dims[2]) : dxyz[2] : _y_g(last(extents[2]), dxyz[2], nxyz_A[2], false, coords[2], dims[2]),
            _z_g(first(extents[3]), dxyz[3], nxyz_A[3], false, coords[3], dims[3]) : dxyz[3] : _z_g(last(extents[3]), dxyz[3], nxyz_A[3], false, coords[3], dims[3]))
end


"""
    metagrid(f::Function, args...; kwargs...)
    metagrid(dims, f::Function, args...; kwargs...)

Return a metadata grid containing the results of the function `f` applied to each process coordinates in the global grid; if `dims` is provided, the function is applied to each process coordinates in an imaginary grid with the dimensions `dims` (the other grid parameters like periodicity and overlaps are inherited from the actual global grid). The return type is an array with elements of the same type as the return type of `f`.

# Arguments
- `f::Function`: the function to apply to each process coordinates; the function must accept the keyword argument `coords` (a tuple of the coordinates of the process). Furthermore, for imaginary grids, the function can optionally also accept the keyword arguments `dims` (the dimensions of the imaginary grid).
- `args...`: the arguments to pass to the function `f`.
- `kwargs...`: the keyword arguments to pass to the function `f`; the keyword argument `coords` (and optionally `dims` for imaginary grids) is automatically added.
- `dims::Tuple{Int,Int,Int}`: the dimensions of the imaginary grid to use instead of the global grid.

# Examples
```jldoctest
julia> using ImplicitGlobalGrid

julia> lx=8; ly=8; lz=8; nx=5; ny=5; nz=7;

julia> init_global_grid(nx, ny, nz; periodz=1);
Global grid: 5x5x5 (nprocs: 1, dims: 1x1x1)

julia> dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)
(2.0, 2.0, 2.0)

julia> metagrid((; coords)->coords)
1×1×1 Array{Tuple{Int64, Int64, Int64}, 3}:
[:, :, 1] =
 (0, 0, 0)

julia> metagrid((2, 2, 2), (; coords)->coords)
2×2×2 Array{Tuple{Int64, Int64, Int64}, 3}:
[:, :, 1] =
 (0, 0, 0)  (0, 1, 0)
 (1, 0, 0)  (1, 1, 0)

[:, :, 2] =
 (0, 0, 1)  (0, 1, 1)
 (1, 0, 1)  (1, 1, 1)

julia> metagrid(extents_g, 0)
1×1×1 Array{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}, 3}:
[:, :, 1] =
 (1:5, 1:5, 1:5)

julia> metagrid(extents_g, 0; fix_global_boundaries=false)
1×1×1 Array{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}, 3}:
[:, :, 1] =
 (2:4, 2:4, 1:5)

julia> metagrid((2, 2, 2), extents_g, 0)
2×2×2 Array{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}, 3}:
[:, :, 1] =
 (1:4, 1:4, 1:5)  (1:4, 5:8, 1:5)
 (5:8, 1:4, 1:5)  (5:8, 5:8, 1:5)

[:, :, 2] =
 (1:4, 1:4, 6:10)  (1:4, 5:8, 6:10)
 (5:8, 1:4, 6:10)  (5:8, 5:8, 6:10)

julia> metagrid(extents_g, 1) # The extents grid as required by VTK
1×1×1 Array{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}, 3}:
[:, :, 1] =
 (1:5, 1:5, 1:5)

julia> metagrid(extents_g, 1; dxyz=(dx, dy, dz)) # The Cartesian coordinates corresponding to the extents required by VTK
1×1×1 Array{Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, 3}:
[:, :, 1] =
 (0.0:2.0:8.0, 0.0:2.0:8.0, 0.0:2.0:8.0)


julia> metagrid((2, 2, 2), extents_g, 1) # The extents grid as required by VTK for an imaginary grid with dimensions (2, 2, 2) and periodic in z (inherited from the global grid)
2×2×2 Array{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}, 3}:
[:, :, 1] =
 (1:4, 1:4, 1:5)  (1:4, 4:8, 1:5)
 (4:8, 1:4, 1:5)  (4:8, 4:8, 1:5)

[:, :, 2] =
 (1:4, 1:4, 5:10)  (1:4, 4:8, 5:10)
 (4:8, 1:4, 5:10)  (4:8, 4:8, 5:10)

julia> metagrid((2, 2, 2), extents_g, 1; dxyz=(dx, dy, dz)) # The Cartesian coordinates corresponding to the extents required by VTK for an imaginary grid with dimensions (2, 2, 2) and periodic in z (inherited from the global grid)
2×2×2 Array{Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, 3}:
[:, :, 1] =
 (0.0:2.0:6.0, 0.0:2.0:6.0, 0.0:2.0:8.0)   (0.0:2.0:6.0, 6.0:2.0:14.0, 0.0:2.0:8.0)
 (6.0:2.0:14.0, 0.0:2.0:6.0, 0.0:2.0:8.0)  (6.0:2.0:14.0, 6.0:2.0:14.0, 0.0:2.0:8.0)

[:, :, 2] =
 (0.0:2.0:6.0, 0.0:2.0:6.0, 8.0:2.0:18.0)   (0.0:2.0:6.0, 6.0:2.0:14.0, 8.0:2.0:18.0)
 (6.0:2.0:14.0, 0.0:2.0:6.0, 8.0:2.0:18.0)  (6.0:2.0:14.0, 6.0:2.0:14.0, 8.0:2.0:18.0)

julia> finalize_global_grid()
```
"""
function metagrid(f::Function, args...; kwargs...)
    return [f(args...; coords=(x,y,z), kwargs...) for x=0:@dimx()-1, y=0:@dimy()-1, z=0:@dimz()-1]
end

function metagrid(dims::Tuple{Int,Int,Int}, f::Function, args...; kwargs...)
    if haskwarg(f, :dims) && haskwarg(f, :coords)
        return [f(args...; coords=(x,y,z), dims=dims, kwargs...) for x=0:dims[1]-1, y=0:dims[2]-1, z=0:dims[3]-1]
    elseif haskwarg(f, :coords)
        return [f(args...; coords=(x,y,z), kwargs...) for x=0:dims[1]-1, y=0:dims[2]-1, z=0:dims[3]-1]
    else
        @ArgumentError("The function `f` must accept the keyword argument `coords` (and, for imaginary grids, optionally also `dims`).")
    end
end

haskwarg(f::Function, kwarg::Symbol) = any(kwarg ∈ Base.kwarg_decl(m) for m in methods(f))



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
