# shared.jl

is_cuarray(A::GGArray) = false


# update_halo.jl

function free_update_halo_cubuffers end
function init_cubufs_arrays end
function init_cubufs end
function reinterpret_cubufs end
function reallocate_undersized_cubufs end
function reregister_cubufs end
function get_cusendbufs_raw end
function get_curecvbufs_raw end
function allocate_custreams end