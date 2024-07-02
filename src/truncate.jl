"""
    TruncatedLevyProcess(p, l, u)

Construct a trunctated Lévy process by restricting the absolute size of jumps.

# Arguments
- `p::LevyProcess`: The original Lévy process
- `l::Real`: The lower bound on the absolute jump size, which can be any non-negative value
- `u::Real`: The upper bound on the absolute jump size, which can be any positive value
  including `Inf`
"""

export TruncatedLevyProcess

struct TruncatedLevyProcess{P<:LevyProcess} <: LevyProcess
    process::P                    # the original process (untruncated)
    lower::Float64                # lower bound on absolute jump size
    upper::Float64                # upper bound on absolute jump size

    # Cached values
    lower_tail_mass::Float64      # upper tail mass of the lower bound
    upper_tail_mass::Float64      # upper tail mass of the upper bound
    mass::Float64                 # total mass between the bounds
end

### Constructors

# TODO: would it be better to use `nothing` for no bounds so that we can dispatch on this?
function TruncatedLevyProcess(p::LevyProcess, l::Float64, u::Float64)
    l < u || throw(ArgumentError("the lower bound must be less than the upper bound."))
    l >= 0 || throw(ArgumentError("the lower bound must be non-negative."))
    u > 0 || throw(ArgumentError("the upper bound must be positive."))
    # TODO: need a fallback for these, and only run when required
    lower_tail_mass = levy_tail_mass(p, l)
    upper_tail_mass = levy_tail_mass(p, u)
    mass = lower_tail_mass - upper_tail_mass
    return TruncatedLevyProcess(p, l, u, lower_tail_mass, upper_tail_mass, mass)
end

TruncatedLevyProcess(p::LevyProcess, l::Real, u::Real) = TruncatedLevyProcess(p, Float64(l), Float64(u))
TruncatedLevyProcess(p::LevyProcess; l=0.0, u=Inf) = TruncatedLevyProcess(p, l, u)

### Support
islowerbounded(p::TruncatedLevyProcess) = islowerbounded(p.process) || p.lower > 0
isupperbounded(p::TruncatedLevyProcess) = isupperbounded(p.process) || p.upper < Inf

### Evalutation

function levy_density(p::TruncatedLevyProcess, x::T) where {T<:Real}
    p.lower <= x <= p.upper ? levy_density(p.process, x) : zero(T)
end

function log_levy_density(p::TruncatedLevyProcess, x::T) where {T<:Real}
    p.lower <= x <= p.upper ? log_levy_density(p.process, x) : -T(Inf)
end

function levy_tail_mass(p::TruncatedLevyProcess, x::T) where {T<:Real}
    if x < p.lower
        return p.mass
    elseif x < p.upper
        return levy_tail_mass(p.process, x) - p.upper_tail_mass
    else
        return 0.0
    end
end

### Marginals

struct TruncatedLevyProcessMarginal{P<:TruncatedLevyProcess}
    process::P
    t::Float64
end

function marginal(p::TruncatedLevyProcess, t::Real)
    TruncatedLevyProcessMarginal(p, t)
end
