export GammaProcess

import Distributions: Gamma
import SpecialFunctions: expinti

struct GammaProcess{T<:Real} <: LevyProcess{T}
    γ::T
    λ::T
    function GammaProcess{T}(γ::Real, λ::Real) where {T<:Real}
        γ, λ = T(γ), T(λ)
        isfinite(γ) && γ > 0 || throw(ArgumentError("γ must be finite and positive"))
        isfinite(λ) && λ > 0 || throw(ArgumentError("λ must be finite and positive"))
        return new{T}(γ, λ)
    end
end

function GammaProcess(γ::Real, λ::Real)
    γ, λ = promote(float(γ), float(λ))
    return GammaProcess{typeof(γ)}(γ, λ)
end

levy_density(p::GammaProcess, x::Real) = x > 0 ? p.γ / x * exp(-p.λ * x) : zero(float(x))
log_levy_density(p::GammaProcess, x::Real) = x > 0 ? log(p.γ) - p.λ * x - log(x) : -Inf

function levy_drift(p::GammaProcess)
    return p.γ * (-expm1(-p.λ) / p.λ)
end

# Simulation of Lévy Random Fields, Wolpert and Ickstadt, 1998
function levy_tail_mass(p::GammaProcess, x::Real)
    x >= 0 || throw(DomainError(x, "the absolute jump threshold must be non-negative"))
    return x == 0 ? oftype(p.γ, Inf) : -p.γ * expinti(-x * p.λ)
end

function marginal(p::GammaProcess, t::Real)
    isfinite(t) && t > 0 || throw(ArgumentError("marginal time must be finite and positive"))
    return Gamma(p.γ * t, 1 / p.λ)
end

#################################
#### TRUNCATED GAMMA PROCESS ####
#################################

const TruncatedGammaProcess{T<:Real} = TruncatedLevyProcess{T,GammaProcess{T}}

#### Rejection Sampling ####

# Dominating process used for rejection sampling
struct GammaDominatingProcess{T<:Real} <: LevyProcess{T}
    γ::T
    λ::T
end

levy_density(p::GammaDominatingProcess, x::Real) = p.γ / (x * (1 + p.λ * x))
levy_tail_mass(p::GammaDominatingProcess, x::Real) = p.γ * log1p(1 / (p.λ * x))
# WILL: should this be parameter — collision with LevyProcesses.jl:19
# Try T<:Real
function inverse_levy_tail_mass(p::GammaDominatingProcess{T}, Γ::T) where {T<:Real}
    T(1.0) / (p.λ * expm1(Γ / p.γ))
end

function inverse_levy_tail_mass(
    p::GammaDominatingProcess{T}, Γs::AbstractVector{T}
) where {T}
    T(1.0) ./ (p.λ .* expm1.(Γs ./ p.γ))
end

const TruncatedGammaDominatingProcess{T<:Real} = TruncatedLevyProcess{
    T,GammaDominatingProcess{T}
}
function sample(rng::AbstractRNG, p::TruncatedGammaDominatingProcess, dt::Real)
    sample(rng, p, dt, Inversion)
end

# Default sampling method — roughly 2x faster than Inversion
function sample(rng::AbstractRNG, p::TruncatedGammaProcess{T}, dt::Real) where {T}
    p₀ = TruncatedLevyProcess(
        GammaDominatingProcess(p.process.γ, p.process.λ), p.lower, p.upper
    )
    levy_density_ratio(x) = (1 + p.process.λ * x) * exp(-p.process.λ * x)
    return sample(rng, p, dt, p₀, Rejection; levy_density_ratio=levy_density_ratio)
end

# Default batch rejection method
function sample(p::TruncatedGammaProcess{T}, dt::Real, N::Integer, BatchRejection) where {T}
    p₀ = TruncatedLevyProcess(
        GammaDominatingProcess(p.process.γ, p.process.λ), p.lower, p.upper
    )
    levy_density_ratio(x) = (1 + p.process.λ * x) * exp(-p.process.λ * x)
    return sample(p, dt, N, p₀, BatchRejection; levy_density_ratio=levy_density_ratio)
end
