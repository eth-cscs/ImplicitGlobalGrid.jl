const ERRMSG_EXTENSION_NOT_LOADED = "AD: the LoopVectorization extension was not loaded. Make sure to import LoopVectorization before ImplicitGlobalGrid."

memcopy_loopvect!(args...) = @NotLoadedError(ERRMSG_EXTENSION_NOT_LOADED)