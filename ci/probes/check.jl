# Compare the output of the probes with expectations.toml: julia check.jl <files with probe output>...
# Prints one line per probe line (OK, MISMATCH or UNJUDGED) and a summary; exits with 1 if there is a mismatch or if no line of a probe in
# CHECK_REQUIRE (default: all probes) was found (e.g. because the probe failed). Needs only the standard library.
using TOML, Printf

const EXPECTATIONS = TOML.parsefile(joinpath(@__DIR__, "expectations.toml"))
const KINDS        = ("XSTREAM", "STREAMS", "PSOVERLAP", "PSCORRECT")

# Parse a line `KIND key=value key=value ...` (values without spaces).
function parse_line(line)
    words = split(strip(line))
    return words[1], Dict(String(k) => String(v) for (k, v) in (split(w, "=", limit=2) for w in words[2:end] if occursin('=', w)))
end

function xstream_entry(backend, version)
    for e in EXPECTATIONS["xstream"]
        if e["backend"] == backend && VersionNumber(e["from"]) <= version && (!haskey(e, "below") || version < VersionNumber(e["below"]))
            return e
        end
    end
    return nothing
end

# Returns (status, expected).
function check(kind, f)
    if kind == "XSTREAM"
        e = xstream_entry(f["backend"], VersionNumber(f["version"]))
        isnothing(e) && return ("MISMATCH", "no entry for $(f["backend"]) $(f["version"]) in expectations.toml")
        expected = (f["optout"] == "1") ? e["optout"] : e[f["case"]]
        return (f["verdict"] == expected ? "OK" : "MISMATCH", expected)
    elseif kind == "STREAMS"
        return (f["stale"] == "0" ? "OK" : "MISMATCH", "stale=0")
    elseif kind == "PSCORRECT"
        return (f["equal"] == "true" ? "OK" : "MISMATCH", "equal=true")
    elseif kind == "PSOVERLAP"
        o = EXPECTATIONS["overlap"]
        e = xstream_entry(f["backend"], VersionNumber(f["version"]))
        isnothing(e) && return ("MISMATCH", "no entry for $(f["backend"]) $(f["version"]) in expectations.toml")
        overlaps = f["optout"] == "1" || (f["psopt"] == "1" && f["bkopt"] == "1") || e["kernel_same"] == "CONCURRENT"
        expected = overlaps ? "OVERLAP" : "SERIALIZED"
        split_ms, comm_ms, hidden = parse(Float64, f["split_ms"]), parse(Float64, f["comm_ms"]), parse(Float64, f["hidden_split"])
        if min(split_ms, comm_ms) < max(o["judge_min_ms"], o["judge_min_ratio"] * max(split_ms, comm_ms))
            return ("UNJUDGED", "$expected (split_ms and comm_ms too different)")
        end
        verdict = hidden >= o["overlap_min"] ? "OVERLAP" : hidden <= o["serialized_max"] ? "SERIALIZED" : "UNCLEAR"
        return (verdict == expected ? "OK" : verdict == "UNCLEAR" ? "UNJUDGED" : "MISMATCH", "$expected (measured: $verdict)")
    end
end

function main(files)
    required = split(get(ENV, "CHECK_REQUIRE", join(KINDS, " ")))
    counts   = Dict("OK" => 0, "MISMATCH" => 0, "UNJUDGED" => 0)
    found    = Set{String}()
    for file in files, line in eachline(file)
        any(k -> startswith(line, k * " "), KINDS) || continue
        kind, f = parse_line(line)
        status, expected = try
            check(kind, f)
        catch err
            ("MISMATCH", "line not understood: $(sprint(showerror, err))")
        end
        push!(found, kind)
        counts[status] += 1
        println(rpad(status, 9), " expected: ", expected, " | ", line)
    end
    missing_kinds = setdiff(required, found)
    for kind in missing_kinds println("MISSING   no $kind line in the probe output") end
    @printf("CHECK SUMMARY ok=%d mismatch=%d unjudged=%d missing=%d\n", counts["OK"], counts["MISMATCH"], counts["UNJUDGED"], length(missing_kinds))
    return (counts["MISMATCH"] == 0 && isempty(missing_kinds)) ? 0 : 1
end

exit(main(ARGS))
