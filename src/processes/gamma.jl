export GammaProcess

import Distributions: Gamma
import SpecialFunctions: expinti

struct GammaProcess{T<:Real} <: LevyProcess{T}
    γ::T
    λ::T
end

levy_density(p::GammaProcess, x::Real) = p.γ / x * exp(-p.λ * x)
log_levy_density(p::GammaProcess, x::Real) = log(p.γ) - p.λ * x - log(x)

function levy_drift(p::GammaProcess)
    return p.γ / p.λ * (1 - exp(-p.λ))
end

# Simulation of Lévy Random Fields, Wolpert and Ickstadt, 1998
levy_tail_mass(p::GammaProcess, x::Real) = -p.γ * expinti(-x * p.λ)

function marginal(p::GammaProcess, t::Real)
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
levy_tail_mass(p::GammaDominatingProcess, x::Real) = p.γ * log(1 + 1 / (p.λ * x))
# WILL: should this be parameter — collision with LevyProcesses.jl:19
# Try T<:Real
function inverse_levy_tail_mass(p::GammaDominatingProcess{T}, Γ::T) where {T<:Real}
    T(1.0) / (p.λ * (exp(Γ / p.γ) - T(1.0)))
end

function inverse_levy_tail_mass(
    p::GammaDominatingProcess{T}, Γs::AbstractVector{T}
) where {T}
    T(1.0) ./ (p.λ .* (exp.(Γs ./ p.γ) .- T(1.0)))
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
