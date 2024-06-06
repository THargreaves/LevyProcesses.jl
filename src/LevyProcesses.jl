module LevyProcesses

import QuadGK: quadgk
import Random: AbstractRNG, default_rng
import SpecialFunctions
import StatsBase: sample

export LevyProcess, Subordinator, TruncatedLevyProcess, TruncatedSubordinator, SampleJumps
export levy_density, log_levy_density, levy_drift, levy_tail_mass, inverse_levy_tail_mass
export marginal

export sample

abstract type LevyProcess end
abstract type Subordinator <: LevyProcess end
abstract type ConditionallyGaussianLevyProcess <: LevyProcess end

domain(::LevyProcess) = -Inf, Inf
domain(::Subordinator) = 0, Inf

# Expected methods
levy_density(p::LevyProcess, ::Real) = error("levy_density not implemented for $(typeof(p))")
levy_drift(p::LevyProcess) = error("levy_drift not implemented for $(typeof(p))")
inverse_levy_tail_mass(p::LevyProcess, ::Real) = error("inverse_levy_tail_mass not implemented for $(typeof(p))")

# Default methods
function log_levy_density(p::LevyProcess, x::Real)
    @warn "log_levy_density not implemented for $(typeof(p)), using log(levy_density)"
    return log(levy_density(p, x))
end

include("truncate.jl")

struct SampleJumps{T<:Real}
    jump_times::Vector{T}
    jump_sizes::Vector{T}
end

struct MarginalisedSampleJumps{T<:Real}
    jump_times::Vector{T}
    jump_means::Vector{T}
    jump_variances::Vector{T}
end

function unnormalised_sample_jumps_density(
    p::TruncatedLevyProcess,
    dt::Real,
    path::SampleJumps
)
    N = length(path.jump_sizes)
    return (
        prod(levy_density(p.process, path.jump_sizes)) *
        1 / dt^N *
        1 / factorial(N)
    )
end
function normalised_sample_jumps_density(
    p::TruncatedLevyProcess,
    dt::Real,
    path::SampleJumps
)
    unnormalised_density = unnormalised_sample_jumps_density(p, dt, path)
    return unnormalised_density * exp(-dt * p.mass)
end

function log_unnormalised_sample_jumps_density(
    p::TruncatedLevyProcess,
    dt::Real,
    path::SampleJumps
)
    N = length(path.jump_sizes)
    return (
        sum(log_levy_density(p.process, path.jump_sizes)) -
        N * log(dt) -
        SpecialFunctions.logfactorial(N)
    )
end
function log_normalised_sample_jumps_density(
    p::TruncatedLevyProcess,
    dt::Real,
    path::SampleJumps
)
    unnormalised_density = log_unnormalised_sample_path_density(p, dt, path)
    return unnormalised_density - dt * p.mass
end

include("utils.jl")

# Process definitions
include("processes/gamma.jl")
include("processes/stable.jl")
include("processes/nvm.jl")

# Sampling methods
include("sampling.jl")

# SDEs
include("sdes.jl")

end # module LevyProcesses
