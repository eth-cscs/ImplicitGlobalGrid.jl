# Testing

The test suite is run with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/):
```julia
import Pkg; Pkg.test("ImplicitGlobalGrid")
```
Each test file runs in its own Julia process. Test files can be selected by name:
```julia
import Pkg; Pkg.test("ImplicitGlobalGrid"; test_args=["test_update_halo.jl"])
```

The GPU tests run if CUDA or AMDGPU is functional, and are skipped otherwise. The following environment variables control the test run:

| Variable           | Description                                                                                                                                   |
|:------------------ |:--------------------------------------------------------------------------------------------------------------------------------------------- |
| `IGG_TEST_NPROCS`  | Number of MPI processes for the multi-rank test files (default: `1`); all other test files run with a single process.                        |
| `IGG_TEST_MPIEXEC` | MPI launcher overriding `MPI.mpiexec()` (e.g. `"srun --overlap"` within a Slurm allocation); if set, all test files are launched with it.  |
| `IGG_TEST_BACKEND` | GPU backend that must be functional (`CUDA` or `AMDGPU`); if it is not, the test run fails instead of skipping the GPU tests.                 |

For example, to run the multi-rank tests with 4 processes (on GPUs: on a GPU node within a Slurm allocation):

:::code-group

```bash [CPUs]
export IGG_TEST_NPROCS=4
julia --project -e 'import Pkg; Pkg.test()'
```

```bash [Nvidia GPUs]
export IGG_TEST_NPROCS=4 IGG_TEST_BACKEND=CUDA IGG_TEST_MPIEXEC="srun --overlap"
julia --project -e 'import Pkg; Pkg.test()'
```

```bash [AMD GPUs]
export IGG_TEST_NPROCS=4 IGG_TEST_BACKEND=AMDGPU IGG_TEST_MPIEXEC="srun --overlap"
julia --project -e 'import Pkg; Pkg.test()'
```

:::

As the tests select the GPU by node-local MPI rank, every rank must see all GPUs of its node (no per-task GPU binding).

## Continuous integration

- **GitHub Actions** (CPU): Julia `min`, `1` and `pre` on one process, the multi-rank tests on 4 and 8 processes, and the Polyester code path with multiple threads (`.github/workflows/ci.yml`).
- **CSCS CI** on [Alps](https://docs.cscs.ch/alps/), with one pipeline per vCluster:
  - [Daint](https://docs.cscs.ch/clusters/daint/) (Nvidia GH200 GPUs): the full test suite on one process, the multi-rank tests on 1 node × 4 processes (with and without CUDA-aware MPI) and on 2 nodes × 4 processes, and the Polyester code path (`ci/cscs-gh200.yml`; [pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/1939427232800723/3544630643253668/-/pipelines)). The multi-rank jobs only run if the single-process job passes.
  - [Beverin](https://docs.cscs.ch/clusters/beverin/) (AMD MI250X and MI300A GPUs): the full test suite on one process, on each GPU type (`ci/cscs-mi250.yml`, `ci/cscs-mi300.yml`; [MI250X pipelines](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/1939427232800723/3544630643253669/-/pipelines)).

Coverage of all jobs is uploaded to [Codecov](https://app.codecov.io/gh/eth-cscs/ImplicitGlobalGrid.jl) (flags `cpu`, `gh200`, `mi250` and `mi300`).
