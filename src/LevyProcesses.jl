module LevyProcesses

import QuadGK: quadgk
import Random: AbstractRNG, default_rng
using SpecialFunctions: SpecialFunctions
import StatsBase: sample
using StaticArrays

export LevyProcess, TruncatedLevyProcess, SampleJumps
export levy_density, log_levy_density, levy_drift, levy_tail_mass, inverse_levy_tail_mass
export marginal

export sample

abstract type LevyProcess{T} end

# Expected methods
function levy_density(p::LevyProcess, ::Real)
    return error("levy_density not implemented for $(typeof(p))")
end
function inverse_levy_tail_mass(p::LevyProcess{T}, ::T) where {T<:Real}
    return error("inverse_levy_tail_mass not implemented for $(typeof(p))")
end

# Default methods
function log_levy_density(p::LevyProcess, x::Real)
    @warn "log_levy_density not implemented for $(typeof(p)), using log(levy_density)"
    return log(levy_density(p, x))
end
"""
    levy_drift(p)

Drift in the Lévy–Khintchine convention `h(x) = x * (abs(x) <= 1)`.
This is not generally the deterministic drift of an uncompensated jump sum.
"""
levy_drift(p::LevyProcess{T}) where {T} = zero(T)
levy_variance(p::LevyProcess{T}) where {T} = zero(T)

include("truncate.jl")

struct SampleJumps{T<:Real}
    jump_times::Vector{T}
    jump_sizes::Vector{T}

    function SampleJumps{T}(times, sizes) where {T<:Real}
        length(times) == length(sizes) || throw(DimensionMismatch("jump times and sizes must have equal lengths"))
        return new{T}(times, sizes)
    end
end

SampleJumps(times::Vector{T}, sizes::Vector{T}) where {T<:Real} = SampleJumps{T}(times, sizes)

function Base.length(s::SampleJumps)
    return length(s.jump_times)
end
function Base.sort(s::SampleJumps)
    idx = sortperm(s.jump_times)
    return SampleJumps(s.jump_times[idx], s.jump_sizes[idx])
end
function Base.sort!(s::SampleJumps)
    idx = sortperm(s.jump_times)
    permute!(s.jump_times, idx)
    permute!(s.jump_sizes, idx)
    return s
end

struct MarginalisedSampleJumps{T<:Real}
    jump_times::Vector{T}
    jump_means::Vector{T}
    jump_variances::Vector{T}
end

"""
    log_unnormalised_sample_jumps_density(p, dt, path; sorted=false)

Retained-path log density without the Poisson compensator. The reference measure
uses a count and labelled time/mark pairs; `sorted=true` uses ordered times instead.
"""
function log_unnormalised_sample_jumps_density(
    p::TruncatedLevyProcess, dt::Real, path::SampleJumps; sorted::Bool=false
)
    isfinite(dt) && dt >= 0 || throw(ArgumentError("dt must be finite and nonnegative"))
    isfinite(p.mass) && p.mass >= 0 || throw(ArgumentError("path density requires finite retained intensity"))
    N = length(path.jump_sizes)
    length(path.jump_times) == N || throw(DimensionMismatch("jump times and sizes must have equal lengths"))
    dt == 0 && N > 0 && return -Inf
    all(t -> isfinite(t) && 0 <= t <= dt, path.jump_times) || return -Inf
    sorted && !issorted(path.jump_times) && return -Inf
    value = sum(x -> log_levy_density(p, x), path.jump_sizes; init=zero(p.mass))
    return sorted ? value : value - SpecialFunctions.logfactorial(N)
end

"""
    log_normalised_sample_jumps_density(p, dt, path; sorted=false)

Retained Poisson-path log density, including `-dt * p.mass`.
"""
function log_normalised_sample_jumps_density(
    p::TruncatedLevyProcess, dt::Real, path::SampleJumps; sorted::Bool=false
)
    return log_unnormalised_sample_jumps_density(p, dt, path; sorted) - dt * p.mass
end

function unnormalised_sample_jumps_density(p::TruncatedLevyProcess, dt::Real, path::SampleJumps; sorted::Bool=false)
    return exp(log_unnormalised_sample_jumps_density(p, dt, path; sorted))
end

function normalised_sample_jumps_density(p::TruncatedLevyProcess, dt::Real, path::SampleJumps; sorted::Bool=false)
    return exp(log_normalised_sample_jumps_density(p, dt, path; sorted))
end

# Process definitions
include("processes/gamma.jl")
include("processes/stable_subordinator.jl")
include("processes/stable.jl")
include("processes/nvm.jl")

# Sampling methods
include("sampling/jumps.jl")
include("sampling/residual_increments.jl")

# SDEs
include("sdes.jl")

end # module LevyProcesses
