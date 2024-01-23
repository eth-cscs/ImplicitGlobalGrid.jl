push!(LOAD_PATH, "../src")
using Test
import MPI, CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require


## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.
nx = 7;
ny = 5;
nz = 6;
dx = 1.0
dy = 1.0
dz = 1.0

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin
    @testset "1. argument check" begin
        @testset "sizes" begin
        	me, dims = init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);
        	A = zeros(nx);
            B = zeros(nx, ny);
            C = zeros(nx, ny, nz);
            A_g = zeros(nx*dims[1]+1);
            B_g = zeros(nx*dims[1], ny*dims[2]-1);
            C_g = zeros(nx*dims[1], ny*dims[2], nz*dims[3]+2);
            if (me == 0) @test_throws ErrorException gather!(A, A_g) end # Error: A_g is not product of size(A) and dims (1D)
            if (me == 0) @test_throws ErrorException gather!(B, B_g) end # Error: B_g is not product of size(A) and dims (2D)
            if (me == 0) @test_throws ErrorException gather!(C, C_g) end # Error: C_g is not product of size(A) and dims (3D)
            if (me == 0) @test_throws ErrorException gather!(C, nothing) end # Error: global is nothing
        	finalize_global_grid(finalize_MPI=false);
        end;
    end;

    @testset "2. gather!" begin
        @testset "1D" begin
            me, dims = init_global_grid(nx, 1, 1, overlaps=(0,0,0), quiet=true, init_MPI=false);
            P   = zeros(nx);
            P_g = zeros(nx*dims[1]);
            P  .= [x_g(ix,dx,P) for ix=1:size(P,1)];
            P_g_ref = [x_g(ix,dx,P_g) for ix=1:size(P_g,1)];
            P_g_ref .= -P_g_ref[1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            finalize_global_grid(finalize_MPI=false);
        end;
        @testset "2D" begin
            me, dims = init_global_grid(nx, ny, 1, overlaps=(0,0,0), quiet=true, init_MPI=false);
            P   = zeros(nx, ny);
            P_g = zeros(nx*dims[1], ny*dims[2]);
            P  .= [y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2)];
            P_g_ref = [y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2)];
            P_g_ref .= -P_g_ref[1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            finalize_global_grid(finalize_MPI=false);
        end;
        @testset "3D" begin
            me, dims = init_global_grid(nx, ny, nz, overlaps=(0,0,0), quiet=true, init_MPI=false);
            P   = zeros(nx, ny, nz);
            P_g = zeros(nx*dims[1], ny*dims[2], nz*dims[3]);
            P  .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
            P_g_ref = [z_g(iz,dz,P_g)*1e2 + y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P_g,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            finalize_global_grid(finalize_MPI=false);
        end;
        @testset "1D, then larger 3D, then smaller 2D" begin
            me, dims = init_global_grid(nx, ny, nz, overlaps=(0,0,0), quiet=true, init_MPI=false);
            # (1D)
            P   = zeros(nx);
            P_g = zeros(nx*dims[1], dims[2], dims[3]);
            P  .= [x_g(ix,dx,P) for ix=1:size(P,1)];
            P_g_ref = [x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P_g,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            # (3D)
            P   = zeros(nx, ny, nz);
            P_g = zeros(nx*dims[1], ny*dims[2], nz*dims[3]);
            P  .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
            P_g_ref = [z_g(iz,dz,P_g)*1e2 + y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P_g,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            # (2D)
            P   = zeros(nx, ny);
            P_g = zeros(nx*dims[1], ny*dims[2], dims[3]);
            P  .= [y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2)];
            P_g_ref = [y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            finalize_global_grid(finalize_MPI=false);
        end;
        @testset "Float32, then Float64, then Int16" begin
            me, dims = init_global_grid(nx, ny, nz, overlaps=(0,0,0), quiet=true, init_MPI=false);
            # Float32 (1D)
            P   = zeros(Float32, nx);
            P_g = zeros(Float32, nx*dims[1], dims[2], dims[3]);
            P  .= [x_g(ix,dx,P) for ix=1:size(P,1)];
            P_g_ref = [x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P_g,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== Float32.(P_g_ref)) end
            # Float64 (3D)
            P   = zeros(Float64, nx, ny, nz);
            P_g = zeros(Float64, nx*dims[1], ny*dims[2], nz*dims[3]);
            P  .= [z_g(iz,dz,P)*1e2 + y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
            P_g_ref = [z_g(iz,dz,P_g)*1e2 + y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P_g,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== Float64.(P_g_ref)) end
            # Int16 (2D)
            P   = zeros(Int16, nx, ny);
            P_g = zeros(Int16, nx*dims[1], ny*dims[2], dims[3]);
            P  .= [y_g(iy,dy,P)*1e1 + x_g(ix,dx,P) for ix=1:size(P,1), iy=1:size(P,2)];
            P_g_ref = [y_g(iy,dy,P_g)*1e1 + x_g(ix,dx,P_g) for ix=1:size(P_g,1), iy=1:size(P_g,2), iz=1:size(P,3)];
            P_g_ref .= -P_g_ref[1,1,1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== Int16.(P_g_ref)) end
            finalize_global_grid(finalize_MPI=false);
        end;
        if (nprocs>1)
            @testset "non-default root" begin
                me, dims = init_global_grid(nx, 1, 1, quiet=true, init_MPI=false);
            	A = zeros(nx);
                A_g = zeros(nx*dims[1]);
                A .= 1.0;
                root = 1;
                gather!(A, A_g; root=root);
                if (me == root) @test all(A_g .== 1.0) end
            	finalize_global_grid(finalize_MPI=false);
            end;
        end
        @testset "nothing on non-root" begin
            me, dims = init_global_grid(nx, 1, 1, overlaps=(0,0,0), quiet=true, init_MPI=false);
            P   = zeros(nx);
            P_g = (me == 0) ? zeros(nx*dims[1]) : nothing
            P  .= [x_g(ix,dx,P) for ix=1:size(P,1)];
            if (me == 0)
                P_g_ref = [x_g(ix,dx,P_g) for ix=1:size(P_g,1)];
                P_g_ref .= -P_g_ref[1] .+ P_g_ref;  # NOTE: We add the first value of P_g_ref to have it start at 0.0.
            end
            gather!(P, P_g);
            if (me == 0) @test all(P_g .== P_g_ref) end
            finalize_global_grid(finalize_MPI=false);
        end;
    end;
end;

## Test tear down
MPI.Finalize()
