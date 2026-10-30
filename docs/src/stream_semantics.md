# [GPU streams and synchronization](@id stream-semantics)

[`update_halo!`](@ref) runs on streams of its own: it does not wait for work enqueued before the call, which must therefore be synchronized before, and it returns once the halo update is complete (see [Asynchronous halo updates and overlap with computation](@ref usage-stream-ordering)).

CUDA.jl (from v6.3) and AMDGPU.jl moreover record the stream that last used an array and make a use on another stream wait for it (*implicit synchronization*). For `update_halo!`, this means:

- It does not replace the synchronization before the call: it orders the kernels that `update_halo!` uses for dimension 1 after preceding work on the array, but not the memory copies it uses for dimensions 2 and 3.
- It prevents overlapping the computation of an array with the update of its halo. To overlap them, disable it for the arrays concerned and order the work explicitly:

```julia
CUDA.enable_synchronization!(A, false)   # AMDGPU.enable_synchronization!(A, false) with AMDGPU.jl ≥ FILL
# ... computation of A on another stream, update_halo!(A), synchronization of that stream ...
CUDA.enable_synchronization!(A, true)
```

`@hide_communication` and `@overlap` of [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) do this automatically (from ParallelStencil FILL on).


## [Behaviour per version](@id stream-semantics-versions)
A kernel uses the array `A` on stream `s1`; while it runs, `A` or another array is used on stream `s2`. *Host waits*: the call returns only once `s1` is done. *GPU waits*: the call returns, but the work on `s2` starts only once `s1` is done. *Concurrent*: no wait.

```@eval
Main.stream_semantics_table()
```

This table is checked weekly on Nvidia GH200 and AMD MI250X GPUs with an old and the latest version of each package (see [Continuous integration](@ref)).
