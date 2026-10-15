# Testing

The test suite is run with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/):
```julia-repl
julia>]
  pkg> test ImplicitGlobalGrid
```
Each test file runs in its own Julia process. Test files can be selected by name:
```julia
using Pkg; Pkg.test("ImplicitGlobalGrid"; test_args=["test_update_halo.jl"])
```

The GPU tests run if CUDA or AMDGPU is functional, and are skipped otherwise. The following environment variables control the test run:

| Variable           | Description                                                                                                                                   |
|:------------------ |:--------------------------------------------------------------------------------------------------------------------------------------------- |
| `IGG_TEST_NPROCS`  | Number of MPI processes for the multi-rank test files (default: `1`); all other test files run with a single process.                        |
| `IGG_TEST_MPIEXEC` | MPI launcher overriding `MPI.mpiexec()` (e.g. `"srun --overlap"` within a Slurm allocation); if set, all test files are launched with it.  |
| `IGG_TEST_BACKEND` | GPU backend that must be functional (`CUDA` or `AMDGPU`); if it is not, the test run fails instead of skipping the GPU tests.                 |

For example, to run the multi-rank tests with 4 processes on a GPU node within a Slurm allocation:
```bash
export IGG_TEST_NPROCS=4 IGG_TEST_BACKEND=CUDA IGG_TEST_MPIEXEC="srun --overlap"
julia --project -e 'using Pkg; Pkg.test()'
```
As the tests select the GPU by node-local MPI rank, every rank must see all GPUs of its node (no per-task GPU binding).
