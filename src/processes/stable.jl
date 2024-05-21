import SpecialFunctions: gamma
import StableDistributions: Stable

export StableSubordinator

# Equivalent to the stable process with β = 1.0 and 
struct StableSubordinator{T<:Real} <: Subordinator
    α::T
    C::T
    # Cached constants
    C_α::T
    σ::T
end
function StableSubordinator(α::Real, C::Real)
    C_α = 1 / π * gamma(α) * sin(π * α / 2)
    σ = (C / (2 * C_α * α))^(1 / α)
    return StableSubordinator(α, C, C_α, σ)
end

levy_density(p::StableSubordinator, x::Real) = p.C / x^(1 + p.α)
log_levy_density(p::StableSubordinator, x::Real) = log(p.C) - (1 + p.α) * log(x)

levy_tail_mass(p::StableSubordinator, x::Real) = p.C / p.α * x^(-p.α)
inverse_levy_tail_mass(p::StableSubordinator, Γ::Real) = (p.α * Γ / p.C)^(-1 / p.α)

function marginal(p::StableSubordinator, t::Real)
    return Stable(p.α, 1.0, p.σ * t^(1 / p.α), 0.0)
end
