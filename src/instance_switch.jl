export switch, update_halo!

# Once multi instance is enabled in an execution it wont be disabled
# Same applies for single instance
let
	  global is_instance_number_defined, is_multi_instance, set_multi_instance, set_single_instance

    undefined                        :: Bool = true
    multi_instance                   :: Bool = false
    is_instance_number_defined()     :: Bool = !undefined
    is_multi_instance()              :: Bool = (if undefined error("Undefined instance policy,"
                                                              * "has yet to be set to single or multi")
                                           else multi_instance end)

    set_multi_instance()                     = (undefined = false; multi_instance = true; nothing)
    set_single_instance()                    = (undefined = false; multi_instance = false; nothing)


    ### Below is an alternative way of interacting with the grids
    # Its pretty much a globally accesible table of global grids to save the instantiated grids
    # Using it is completely optional

    global add_gg_to_table, gg_table_get, gg_table_erase

    gg_table      :: Vector{GlobalGrid} = []
    # Adds a global grid, returns its ID (position) in the table
    add_gg_to_table(gg :: GlobalGrid) :: Int = (push!(gg_table, gg); length(gg_table))
    # Returns a global grid at position ID
    gg_table_get(id :: Int)    :: GlobalGrid = (if (length(gg_table) >= id > 0)
                                                    gg_table[id]
                                                else error("Bad global grid ID") end)
    gg_table_erase(id::Int)    :: GlobalGrid = (if (length(gg_table) >= id > 0)
                                                    gg_table[id] = GLOBAL_GRID_NULL
                                                else error("Bad global grid ID") end)
end

"""
  Changes the focused global grid and returns the previously focused one.
"""
function switch(global_grid::GlobalGrid)::GlobalGrid

    if !is_multi_instance()
        error("Illegal switch: this execution environment has been initialised in a single global instance regime")
    end
    gg = get_global_grid()
    set_global_grid(global_grid)
    if gg.nprocs <= 0
        init_timing_functions()
    end
    return gg
end

"""
  Additional dispatch that adds the posibility to specify a global grid to use.
"""
function update_halo!(global_grid_instance :: GlobalGrid, A::Union{GGArray, GGField, GGFieldConvertible}...)

    old = switch(global_grid_instance)
    update_halo!(A...)
    switch(old)
    return nothing
end
