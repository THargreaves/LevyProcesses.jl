import Optim
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
    A_0::T
    A_1::T
    ζ::T
    M::T
    λ::T
end
function StableSubordinator(α::Real, C::Real)
    C_α = 1 / π * gamma(α) * sin(π * α / 2)
    σ = (C / (2 * C_α * α))^(1 / α)

    A_0 = (1 - α) * α^(α / (1 - α))
    A_1 = α * (1 - α)^(1 / α - 1)
    ζ = gamma(1 - α)^(-1)
    c_α = gamma(α) / π * sin(π * α / 2)

    # Optimise parameter rejection sampling
    log_cost(λ) = log(α) + log(A_0) + ζ^(1 / α) * λ^(1 - 1 / α) * A_1 + (α - 2) * log(A_0 - λ)
    res = Optim.optimize(λ -> log_cost(λ), 0, A_0)
    λ = Optim.minimizer(res)
    M = α * A_0 * exp(ζ^(1 / α) * λ^(1 - 1 / α) * A_1) * (A_0 - λ)^(α - 2)

    return StableSubordinator(α, C, C_α, σ, A_0, A_1, ζ, M, λ)
end

levy_density(p::StableSubordinator, x::Real) = p.C / x^(1 + p.α)
log_levy_density(p::StableSubordinator, x::Real) = log(p.C) - (1 + p.α) * log(x)

levy_tail_mass(p::StableSubordinator, x::Real) = p.C / p.α * x^(-p.α)
inverse_levy_tail_mass(p::StableSubordinator, Γ::Real) = (p.α * Γ / p.C)^(-1 / p.α)

function marginal(p::StableSubordinator, t::Real)
    return Stable(p.α, 1.0, p.σ * t^(1 / p.α), 0.0)
end
