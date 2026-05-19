push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid;
GG = ImplicitGlobalGrid;
import ImplicitGlobalGrid: @require


## Test setup (NOTE: Testset "2. initialization including MPI" completes the test setup as it initializes MPI and must therefore mandatorily be at the 2nd position). NOTE: these tests require nprocs == 1.
p0 = MPI.PROC_NULL
nx = 4;
ny = 4;
nz = 1;

@testset "$(basename(@__FILE__))" begin

    @testset "1. one grid activation" begin
        init_global_grid(save_kwarg_defaults=true, quiet=true, init_MPI=true)

        gg = create_global_grid(nx, ny, nz)

        dims = gg.dims
        nprocs = gg.nprocs
        me = gg.me
        coords = gg.coords
        comm_cart = gg.comm

        @test !GG.grid_is_initialized()
        activate_global_grid(gg)
        @test GG.grid_is_initialized()
        @testset "values in global grid" begin
            @test GG.global_grid().nxyz_g == [nx, ny, nz]
            @test GG.global_grid().nxyz == [nx, ny, nz]
            @test GG.global_grid().dims == dims
            @test GG.global_grid().overlaps == [2, 2, 2]
            @test GG.global_grid().halowidths == [1, 1, 1]
            @test GG.global_grid().nprocs == nprocs
            @test GG.global_grid().me == me
            @test GG.global_grid().coords == coords
            @test GG.global_grid().neighbors == [p0 p0 p0; p0 p0 p0]
            @test GG.global_grid().periods == [0, 0, 0]
            @test GG.global_grid().disp == 1
            @test GG.global_grid().reorder == 1
            @test GG.global_grid().comm == comm_cart
            @test GG.global_grid().quiet == true
            @test GG.global_grid().origin == [0.0, 0.0, 0.0]
            @test GG.global_grid().origin_on_vertex == false
            @test GG.global_grid().centerxyz == [false, false, false]
        end

        finalize_global_grid(finalize_MPI=false)
    end

    @testset "2. alternating two grid activations" begin
        init_global_grid(save_kwarg_defaults=true, quiet=true, init_MPI=false)

        gg1 = create_global_grid(nx, ny, nz)
        gg2 = create_global_grid(nx * 2, ny * 2, nz * 2)

        activate_global_grid(gg1)
        @test GG.grid_is_initialized()
        @test GG.global_grid().nxyz == [nx, ny, nz]
        gg_old = GG.get_global_grid()
        activate_global_grid(gg2)
        @test GG.grid_is_initialized()
        @test GG.global_grid().nxyz == [nx * 2, ny * 2, nz * 2]
        @test gg_old.nxyz == gg1.nxyz

        finalize_global_grid(finalize_MPI=false)
    end

    @testset "3. activation inside and outside update_halo!" begin
        init_global_grid(save_kwarg_defaults=true, quiet=true, periodx=0, init_MPI=false)

        P = zeros(Float64, nx)
        P[[1, 2, end - 1, end]] = [2.0, 1.0, 2.0, 1.0]
        P = Array(P)
        P_ref = copy(P)
        P[[1, end]] .= (eltype(P)(0.0),)

        gg1 = create_global_grid(nx)
        gg2 = create_global_grid(nx, periodx=1)

        # Updating with gg2 with no default active
        @require !(P == P_ref)
        update_halo!(P; active_global_grid=gg2)
        @test (P == P_ref)

        # Trying to update with active grid with no active grid
        @test_throws ErrorException update_halo!(P)

        activate_global_grid(gg1)
        P[[1, end]] .= (eltype(P)(0.0),)

        # Updating with gg2 even though gg1 is active
        @require !(P == P_ref)
        update_halo!(P; active_global_grid=gg2)
        @test (P == P_ref)

        @test GG.grid_is_initialized()
        @test GG.global_grid().periods == [0, 0, 0]
        # Mutation checks
        @test gg1.periods == [0, 0, 0]
        @test gg2.periods == [1, 0, 0]

        # Updating with gg1 as active GG
        P = copy(P_ref)
        update_halo!(P)
        @test (P == P_ref)

        # Updating with gg2 as active GG
        activate_global_grid(gg2)
        P[[1, end]] .= (eltype(P)(0.0),)
        @require (P != P_ref)
        update_halo!(P)
        @test (P == P_ref)

        finalize_global_grid(finalize_MPI=false)
    end

    @testset "4. activation when package is not initialized" begin
        @test_throws ErrorException activate_global_grid(GG.GLOBAL_GRID_NULL)
        @test_throws ErrorException GG.check_grid_is_initialized()

        init_global_grid(save_kwarg_defaults=true, quiet=true, periodx=0, init_MPI=false)

        @test_throws ErrorException GG.global_grid()
        
        gg = GG.get_global_grid()
        @test all([gg.:($field) == GG.GLOBAL_GRID_NULL.:($field) for field in fieldnames(GG.GlobalGrid)])

        finalize_global_grid(finalize_MPI=false)
    end
end;

## Test tear down
MPI.Finalize()
