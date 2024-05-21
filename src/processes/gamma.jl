export GammaProcess

import Distributions: Gamma
import SpecialFunctions: expinti

struct GammaProcess{T<:Real} <: Subordinator
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

# Computed using Halley iterations
# TODO: for C = 2.1, β = 0.5, c = 5.5, this is returning negative values
function inverse_levy_tail_mass(p::GammaProcess, Γ::Real; tol=1e-9, max_iter=100)
    # Initial guess from approximation E1(x) ≈ -γ - log(x)
    x = exp(-Γ / p.λ - MathConstants.eulergamma) / p.λ
    for _ in 1:max_iter
        f = levy_tail_mass(p, x) - Γ
        f_prime = -p.γ * exp(-p.λ * x) / x
        c1 = f / f_prime
        # c2 = f''(x) / f'(x)
        c2 = -(1 + x * p.λ) / x

        Δ = c1 / (1 - 0.5 * c1 * c2)
        x -= Δ
        if abs(Δ) < tol
            return x
        end
    end
    return x
end

function marginal(p::GammaProcess, t::Real)
    return Gamma(p.γ * t, 1 / p.λ)
end

#################################
#### TRUNCATED GAMMA PROCESS ####
#################################

# TODO: are the types correct here?
const TruncatedGammaProcess{T<:Real} = TruncatedLevyProcess{GammaProcess{T},T}

#### Rejection Sampling ####

# Dominating process used for rejection sampling
struct GammaDominatingProcess{T<:Real} <: Subordinator
    γ::T
    λ::T
end

levy_density(p::GammaDominatingProcess, x::Real) = p.γ / (x * (1 + p.λ * x))
levy_tail_mass(p::GammaDominatingProcess, x::Real) = p.γ * log(1 + 1 / (p.λ * x))
inverse_levy_tail_mass(p::GammaDominatingProcess, Γ::Real) = 1 / (p.λ * (exp(Γ / p.γ) - 1))

const TruncatedGammaDominatingProcess{T<:Real} = TruncatedSubordinator{GammaDominatingProcess{T},T}
sample(rng::AbstractRNG, p::TruncatedGammaDominatingProcess, dt::Real) = sample(rng, p, dt, Inversion)

# Default sampling method — roughly 2x faster than Inversion
function sample(rng::AbstractRNG, p::TruncatedGammaProcess{T}, dt::Real) where {T}
    p₀ = truncate(GammaDominatingProcess(p.process.γ, p.process.λ), p.ϵ)
    levy_density_ratio(x) = (1 + p.process.λ * x) * exp(-p.process.λ * x)
    return sample(rng, p, dt, p₀, Rejection; levy_density_ratio=levy_density_ratio)
end
