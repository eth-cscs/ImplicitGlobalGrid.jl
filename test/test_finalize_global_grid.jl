push!(LOAD_PATH, "../src")
using Test
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import MPI
import ImplicitGlobalGrid: @require


@testset "$(basename(@__FILE__))" begin
    @testset "1. finalization of global grid and MPI" begin
        init_global_grid(4, 4, 4, quiet=true);
        @require GG.grid_is_initialized()
        @require !MPI.Finalized()
        finalize_global_grid()
        @test !GG.grid_is_initialized()
    end;

    @testset "2. exceptions" begin
        @test_throws ErrorException finalize_global_grid();  # Finalize can never be before initialize.
    end;
end;
