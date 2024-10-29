# shared.jl

is_onearray(A::GGArray) = false


# select_device.jl

function nb_inteldevices end
function inteldevice! end


# update_halo.jl

function free_update_halo_intelbuffers end
function init_intelbufs_arrays end
function init_intelbufs end
function reinterpret_intelbufs end
function reallocate_undersized_intelbufs end
function reregister_itnelbufs end
function get_intelsendbufs_raw end
function get_intelrecvbufs_raw end
function allocate_intelstreams end
