#!/bin/bash
# Stack check: run the probes of this directory with one version of a GPU package and compare their output with expectations.toml (see
# docs/src/stream_semantics.md). Run on a node with STACK_NPROCS GPUs, within a Slurm allocation, with MPI.jl configured for the MPI of the
# system (MPIPreferences); used by ci/cscs-stack.yml. Variables:
#   STACK_BACKEND          CUDA or AMDGPU.
#   STACK_BACKEND_VERSION  Version of the GPU package (Pkg version specifier, e.g. 5.4 for the latest 5.4.x; empty: the latest release;
#                          <url>#<rev>: a branch or commit).
#   STACK_PS               ParallelStencil: empty for the latest release, else a path (developed) or <url>#<rev>.
#   STACK_JULIA            Julia command (default: julia).
#   STACK_NPROCS           Number of processes of the multi-process runs (default: 4).
#   STACK_MPIEXEC          MPI launcher (default: srun).
#   STACK_DIR              Directory for the environment and the logs (default: .stack); the environment is recreated at each run.
# Exit code: 0 if all probe results match the expectations, else 1.
set -u
PROBES=$(cd "$(dirname "$0")" && pwd)
: "${STACK_BACKEND:?STACK_BACKEND must be CUDA or AMDGPU}" "${STACK_BACKEND_VERSION:=}" "${STACK_PS:=}" "${STACK_JULIA:=julia}" "${STACK_NPROCS:=4}" "${STACK_MPIEXEC:=srun}" "${STACK_DIR:=.stack}"
export STACK_BACKEND STACK_BACKEND_VERSION STACK_PS PROBE_BACKEND=$STACK_BACKEND
mkdir -p "$STACK_DIR"; STACK_DIR=$(cd "$STACK_DIR" && pwd)
ENVDIR=$STACK_DIR/env; LOGS=$STACK_DIR/logs
rm -rf "$ENVDIR" "$LOGS"; mkdir -p "$ENVDIR" "$LOGS"
JULIA="$STACK_JULIA --startup-file=no --project=$ENVDIR"
AWARE_VAR=$([ "$STACK_BACKEND" = CUDA ] && echo IGG_CUDAAWARE_MPI || echo IGG_ROCMAWARE_MPI)

echo "=== stack check: $STACK_BACKEND ${STACK_BACKEND_VERSION:-latest}, ParallelStencil ${STACK_PS:-latest}, $STACK_NPROCS processes, $(hostname), $(date -Is)"
$JULIA "$PROBES/setup_env.jl" > "$LOGS/setup.log" 2>&1 || { tail -30 "$LOGS/setup.log"; echo "=== environment setup failed"; exit 1; }
grep '^STACKENV' "$LOGS/setup.log"

# run <log name> <number of processes (0: without MPI launcher)> <GPU-aware MPI: 0|1> <probe>
run() {
    local log=$LOGS/$1.log np=$2 aware=$3 probe=$PROBES/$4 launcher=""
    local env=("$AWARE_VAR=$aware")
    if [ "$STACK_BACKEND" = AMDGPU ] && [ "$aware" = 1 ]; then env+=(MPICH_GPU_SUPPORT_ENABLED=1 HSA_ENABLE_IPC_MODE_LEGACY=1); fi  # See docs/src/stream_semantics.md.
    [ "$np" -gt 0 ] && launcher="$STACK_MPIEXEC -n $np"
    env "${env[@]}" $launcher $JULIA -O3 "$probe" > "$log" 2>&1
    local status=$?
    grep -E '^(XSTREAM|STREAMS|PSOVERLAP|PSCORRECT) ' "$log"
    if [ $status -ne 0 ]; then echo "--- $1 failed (exit code $status):"; tail -20 "$log"; fi
}
run xstream               0              0 xstream_probe.jl
run streams-np1           1              0 streams_probe.jl       # Local exchange, both GPU-aware MPI settings.
for aware in 0 1; do
    run streams-aware$aware     "$STACK_NPROCS" $aware streams_probe.jl
    run correctness-aware$aware "$STACK_NPROCS" $aware correctness_probe.jl
done
run overlap-aware0        "$STACK_NPROCS" 0 overlap_probe.jl      # NOTE: host-staged only: with GPU-aware MPI, the overlap also depends on how the hardware copies between device buffers (on MI250X, the copies wait for the compute units held by the computation).

echo "=== comparison with expectations.toml"
$STACK_JULIA --startup-file=no "$PROBES/check.jl" "$LOGS"/*.log
