# Implementation Plan: `auto_task_sync` as the third init-only flag

**Goal:** Implement `auto_task_sync` as a third init-only keyword (alongside `device_type`/`select_device`), which automatically synchronizes the task-local stream before every
`update_halo!` on GPU arrays when set to `true` (default `false`). This addresses the stream-ordering race described in issue #141
(`update_halo!` packs/unpacks halos on its own non-blocking IGG streams, which are not ordered after the task-local stream where user kernels run).

**Storage decision:** module state dict `_default_args` in `src/shared.jl` — **not** a new field in `struct GlobalGrid`, and **no** entry in `GLOBAL_GRID_NULL`,
following the established pattern of `device_type`/`select_device` exactly.

The lifecycle to replicate for `device_type`/`select_device` is:

1. declared in `_default_args` (`src/shared.jl`) — the single source of truth;
2. passed through `normalize_input`;
3. accepted by both `init_global_grid` signatures (literal defaults);
4. written via the unconditional `set_default_args(...)` call — **exempt** from the `save_kwarg_defaults`/`differ_default_args` gating;
5. read at the use site via `default(:auto_task_sync)`;
6. reset automatically by `reset_default_args()` in `finalize_global_grid` (no manual reset code anywhere).

---

## Phase 1 — Module state & plumbing (`src/shared.jl`)

- Add `:auto_task_sync => false` to the `_default_args` dict, in the `device_type`/`select_device` cluster, with the same "changing this between grids is disabled" comment style.
- `DEFAULT_OF_DEFAULT_ARGS` is built via `copy()` and `reset_default_args()` is generic — **no edits needed** there (this is where the "free" reset comes from).
- Extend **both** `normalize_input` methods (lines 156 and 179): new `auto_task_sync` parameter after `select_device`, passed through and returned. A `Bool` needs no extra validation.
- **Explicitly out of scope here:** no new field in `struct GlobalGrid`, no entry in `GLOBAL_GRID_NULL`, no change to the `GlobalGrid(...)` constructor call.

## Phase 2 — Initialization API (`src/init_global_grid.jl`)

- **Method A** (`init_global_grid(; ...)`): add `auto_task_sync::Bool=false` to the signature, in the init-only cluster next to `device_type`/`select_device`.
- **Method B** (`init_global_grid(nx, ny, nz; ...)`): add `auto_task_sync::Bool=false` after `select_device` (before `quiet`), and capture it in the destructuring assignment from `normalize_input`.
- In **both** methods, extend the unconditional call to `set_default_args(device_type=..., select_device=..., auto_task_sync=...)`.
- **Do not** add it to the `differ_default_args(dimx=..., quiet=quiet)` check — the flag must be exempt from `save_kwarg_defaults` gating, exactly like `device_type`/`select_device`.
- Docstring: add the `auto_task_sync` bullet under "Advanced keyword arguments" with the "can only be set up on initialization" sentence.

## Phase 3 — Grid creation passthrough (`src/create_global_grid.jl`)

- Add `auto_task_sync = default(:auto_task_sync)` next to the two lines

  ```julia
  device_type     = default(:device_type)
  select_device   = default(:select_device)
  ```

  i.e. in the "moved out of the argument list" block with the "not expected to change in different grids" comment.
- Pass it through the `normalize_input` call.
- The flag is **not** stored in the `GlobalGrid` struct — it is read again at `update_halo!` time (see Phase 4), which is what makes it apply to *every* halo update on *every* grid, matching the init-only semantics.

## Phase 4 — Runtime hook (stubs + `update_halo!` + both extensions)

This follows the `allocate_custreams`/`allocate_rocstreams` precedent exactly:

- **`src/CUDAExt/defaults.jl`**: declare the stub `function cuda_task_local_sync end` in the `# update_halo.jl` section.
- **`src/AMDGPUExt/defaults.jl`**: declare the stub `function rocm_task_local_sync end` in the `# update_halo.jl` section.
- **`src/update_halo.jl`** (`_update_halo!`): right after the existing error check, **before** `allocate_bufs` (so it precedes any IGG stream work enqueued in the `dims` loop), add the two guarded lines mirroring the `allocate_custreams`/`allocate_rocstreams` lines:

  ```julia
  if (default(:auto_task_sync) && any_cuarray(fields...))  cuda_task_local_sync(); end
  if (default(:auto_task_sync) && any_rocarray(fields...)) rocm_task_local_sync(); end
  ```

- **`src/CUDAExt/update_halo.jl`**: define `ImplicitGlobalGrid.cuda_task_local_sync() = CUDA.synchronize()` — no-arg task-local stream sync, matching the exact call from the #141 MWE and the maintainer's proposal (verify the precise no-arg semantics against CUDA.jl docs at implementation time).
- **`src/AMDGPUExt/update_halo.jl`**: define `ImplicitGlobalGrid.rocm_task_local_sync() = AMDGPU.synchronize()` — no-arg form documented as task-local stream sync by the existing comment in `AMDGPUExt/alloc_rocbuf`.
- Docstring of `update_halo!`: note the stream-ordering contract (caller must synchronize the task-local stream unless `auto_task_sync=true`).

**Why the guards make this safe:** `any_cuarray`/`any_rocarray` are only true when the corresponding extension is loaded, and the flag defaults to `false` — so no `MethodError` is possible, and existing users see zero behavior change.

## Phase 5 — Tests

- **`test/test_init_global_grid.jl`** (plumbing, CPU-only):
  - `GG.default(:auto_task_sync) == false` on a fresh init; `== true` after `init_global_grid(..., auto_task_sync=true)`.
  - Exemption test: passing `auto_task_sync=true` **without** `save_kwarg_defaults` must **not** throw (mirrors the `device_type`/`select_device` exemption).
  - After `finalize_global_grid()`, `default(:auto_task_sync) == false` again (the `reset_default_args` path).
- **`test/test_update_halo.jl`** (functional, gated on CUDA/AMDGPU availability like existing GPU tests):
  - Adapt the #141 MWE (slow kernel writing the array, then `update_halo!(A)` with `auto_task_sync=true`): assert halos match exactly — 0 stale ghost cells.
  - Flag off: assert no error only (the race is not deterministically testable).
  - CPU-array path with the flag on: no `MethodError`, no needless sync.

## Phase 6 — Documentation

- **`docs/src/usage.md`**: new paragraph after the GPU-selection paragraph in the "device selection" section explaining the stream-ordering contract and the `auto_task_sync` keyword (mirroring the `select_device` paragraph's treatment of init-only keywords).
- **`docs/src/api.md`**: no edit needed — the `init_global_grid` `@docs` block renders the docstring automatically, which already contains the `auto_task_sync` bullet (from Phase 2).

---

## Files touched (full list for review)

| # | File | Phase |
|---|---|---|
| 1 | `src/shared.jl` | 1 |
| 2 | `src/init_global_grid.jl` | 2 |
| 3 | `src/create_global_grid.jl` | 3 |
| 4 | `src/CUDAExt/defaults.jl` | 4 |
| 5 | `src/AMDGPUExt/defaults.jl` | 4 |
| 6 | `src/update_halo.jl` | 4 |
| 7 | `src/CUDAExt/update_halo.jl` | 4 |
| 8 | `src/AMDGPUExt/update_halo.jl` | 4 |
| 9 | `test/test_init_global_grid.jl` | 5 |
| 10 | `test/test_update_halo.jl` | 5 |
| 11 | `docs/src/usage.md`, `docs/src/api.md` | 6 |

## Open naming decision

The hook names `cuda_task_local_sync`/`rocm_task_local_sync` follow the `allocate_custreams`/`allocate_rocstreams` precedent (device-prefixed stubs in the per-device defaults files, implementations in the extension files). If a different convention is preferred — e.g. `synchronize_cuda`/`synchronize_rocm` — it's a pure rename within Phase 4 and affects nothing else.
