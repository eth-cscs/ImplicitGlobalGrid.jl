# shared.jl

is_loaded(arg) = false #TODO: this would not work as it should be the caller module...: (Base.get_extension(@__MODULE__, ext) !== nothing)
is_functional(arg) = false


# update_halo.jl

function gpusendbuf end
function gpurecvbuf end
function gpusendbuf_flat end
function gpurecvbuf_flat end

function write_d2x! end
function read_x2d! end
function write_d2h_async! end
function read_h2d_async! end

function gpumemcopy! end
