# shared.jl

is_onearray(A::GGArray) = false


# select_device.jl

function nb_oneapidevices end
function oneapidevice! end


# update_halo.jl

function free_update_halo_intelbuffers end
function init_onebufs_arrays end
function init_onebufs end
function reinterpret_onebufs end
function reallocate_undersized_onebufs end
function reregister_onebufs end
function get_onesendbufs_raw end
function get_onerecvbufs_raw end
function allocate_onestreams end
