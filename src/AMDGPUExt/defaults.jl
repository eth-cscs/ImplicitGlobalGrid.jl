# shared.jl

is_rocarray(A::GGArray) = false


# update_halo.jl

function free_update_halo_rocbuffers end
function init_rocbufs_arrays end
function init_rocbufs end
function reinterpret_rocbufs end
function reallocate_undersized_rocbufs end
function reregister_rocbufs end
function get_rocsendbufs_raw end
function get_rocrecvbufs_raw end
function allocate_rocstreams end