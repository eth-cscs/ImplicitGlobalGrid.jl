const ERRMSG_EXTENSION_NOT_LOADED = "PolyesterExt: the Polyester extension was not loaded. Make sure to import Polyester before ImplicitGlobalGrid."

memcopy_polyester!(args...) = @NotLoadedError(ERRMSG_EXTENSION_NOT_LOADED)