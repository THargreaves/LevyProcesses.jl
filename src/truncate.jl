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

struct TruncatedLevyProcess{T<:Real,P<:LevyProcess{T}} <: LevyProcess{T}
    process::P                    # the original process (untruncated)
    lower::T                      # lower bound on absolute jump size
    upper::T                      # upper bound on absolute jump size

    # Cached values
    drift::T                      # updated drift
    variance::T                   # updated variance
    lower_tail_mass::T            # upper tail mass of the lower bound
    upper_tail_mass::T            # upper tail mass of the upper bound
    mass::T                       # total mass between the bounds
end

### Constructors

# TODO: would it be better to use `nothing` for no bounds so that we can dispatch on this?
function TruncatedLevyProcess(p::LevyProcess, l::Real, u::Real; approximate_residual::Bool=false)
    l < u || throw(ArgumentError("the lower bound must be less than the upper bound."))
    l >= 0 || throw(ArgumentError("the lower bound must be non-negative."))
    u > 0 || throw(ArgumentError("the upper bound must be positive."))
    # TODO: need a fallback for these, and only run when required
    lower_tail_mass = levy_tail_mass(p, l)
    upper_tail_mass = levy_tail_mass(p, u)
    mass = lower_tail_mass - upper_tail_mass
    # Update drift and variance
    drift = levy_drift(p)
    variance = levy_variance(p)
    if approximate_residual
        residual_process = TruncatedLevyProcess(p, 0.0, l)
        drift += mean(residual_process)
        variance += var(residual_process)
    end
    return TruncatedLevyProcess(p, l, u, drift, variance, lower_tail_mass, upper_tail_mass, mass)
end

# TruncatedLevyProcess(p::LevyProcess, l::Real, u::Real) = TruncatedLevyProcess(p, Float64(l), Float64(u))
TruncatedLevyProcess(p::LevyProcess{T}; l=0.0, u=Inf) where {T} = TruncatedLevyProcess(p, T(l), T(u))

### Support
islowerbounded(p::TruncatedLevyProcess) = islowerbounded(p.process) || p.lower > 0
isupperbounded(p::TruncatedLevyProcess) = isupperbounded(p.process) || p.upper < Inf

### Evalutation

levy_drift(p::TruncatedLevyProcess) = p.drift
levy_variance(p::TruncatedLevyProcess) = p.dispersion

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

"""
    FixedLevyProcess(p, N)

Approximation of a Lévy process by fixing the number of jumps.

# Arguments
- `p::LevyProcess`: The original Lévy process
- `N::Int`: The fixed number of jumps
"""

export FixedLevyProcess

# TODO: is this even a Lévy process anymore?
struct FixedLevyProcess{T<:Real,P<:LevyProcess{T}} <: LevyProcess{T}
    process::P
    N::Int
end
