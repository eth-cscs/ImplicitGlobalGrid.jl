module Exceptions
export @ModuleInternalError, @IncoherentCallError, @NotInitializedError, @NotLoadedError, @IncoherentArgumentError, @KeywordArgumentError, @ArgumentEvaluationError, @ArgumentError
export ModuleInternalError, IncoherentCallError, NotInitializedError, NotLoadedError, IncoherentArgumentError, KeywordArgumentError, ArgumentEvaluationError

macro ModuleInternalError(msg) esc(:(throw(ModuleInternalError($msg)))) end
macro IncoherentCallError(msg) esc(:(throw(IncoherentCallError($msg)))) end
macro NotInitializedError(msg) esc(:(throw(NotInitializedError($msg)))) end
macro NotLoadedError(msg)  esc(:(throw(NotLoadedError($msg)))) end
macro IncoherentArgumentError(msg) esc(:(throw(IncoherentArgumentError($msg)))) end
macro KeywordArgumentError(msg) esc(:(throw(KeywordArgumentError($msg)))) end
macro ArgumentEvaluationError(msg) esc(:(throw(ArgumentEvaluationError($msg)))) end
macro ArgumentError(msg) esc(:(throw(ArgumentError($msg)))) end

struct ModuleInternalError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ModuleInternalError) = print(io, "ModuleInternalError: ", e.msg)

struct IncoherentCallError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IncoherentCallError) = print(io, "IncoherentCallError: ", e.msg)

struct NotInitializedError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NotInitializedError) = print(io, "NotInitializedError: ", e.msg)

struct NotLoadedError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NotLoadedError) = print(io, "NotLoadedError: ", e.msg)

struct IncoherentArgumentError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IncoherentArgumentError) = print(io, "IncoherentArgumentError: ", e.msg)

struct KeywordArgumentError <: Exception
    msg::String
end
Base.showerror(io::IO, e::KeywordArgumentError) = print(io, "KeywordArgumentError: ", e.msg)

struct ArgumentEvaluationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ArgumentEvaluationError) = print(io, "ArgumentEvaluationError: ", e.msg)

end # Module Exceptions
