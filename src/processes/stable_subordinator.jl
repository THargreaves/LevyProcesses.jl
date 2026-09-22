using Optim: Optim
import SpecialFunctions: gamma
import StableDistributions: Stable

export StableSubordinator

"""Positive stable subordinator with Lévy density `C * x^(-1-α)`, `0 < α < 1`."""
struct StableSubordinator{T<:Real} <: LevyProcess{T}
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
    isfinite(α) && 0 < α < 1 || throw(ArgumentError("α must lie in (0, 1)"))
    isfinite(C) && C > 0 || throw(ArgumentError("C must be finite and positive"))
    α, C = promote(float(α), float(C))
    C_α = 1 / π * gamma(α) * sin(π * α / 2)
    σ = (C / (2 * C_α * α))^(1 / α)

    A_0 = (1 - α) * α^(α / (1 - α))
    A_1 = α * (1 - α)^(1 / α - 1)
    ζ = gamma(1 - α)^(-1)

    # Optimise parameter rejection sampling
    log_cost(λ) =
        log(α) + log(A_0) + ζ^(1 / α) * λ^(1 - 1 / α) * A_1 + (α - 2) * log(A_0 - λ)
    res = Optim.optimize(λ -> log_cost(λ), 0, A_0)
    λ = Optim.minimizer(res)
    M = α * A_0 * exp(ζ^(1 / α) * λ^(1 - 1 / α) * A_1) * (A_0 - λ)^(α - 2)

    return StableSubordinator{typeof(α)}(α, C, C_α, σ, A_0, A_1, ζ, M, λ)
end

levy_density(p::StableSubordinator, x::Real) = x > 0 ? p.C / x^(1 + p.α) : zero(p.C)
log_levy_density(p::StableSubordinator, x::Real) = x > 0 ? log(p.C) - (1 + p.α) * log(x) : -Inf
levy_drift(p::StableSubordinator) = p.C / (1 - p.α)

function levy_tail_mass(p::StableSubordinator, x::Real)
    x >= 0 || throw(DomainError(x, "jump cutoff must be non-negative"))
    return p.C / p.α * x^(-p.α)
end
function inverse_levy_tail_mass(p::StableSubordinator, Γ::Real)
    Γ >= 0 || throw(DomainError(Γ, "tail mass must be non-negative"))
    return (p.α * Γ / p.C)^(-1 / p.α)
end

function marginal(p::StableSubordinator, t::Real)
    isfinite(t) && t > 0 || throw(ArgumentError("time must be finite and positive"))
    return Stable(p.α, 1.0, p.σ * t^(1 / p.α), 0.0)
end

###########################################
#### TRUNCATED STABLE SUBORDINATOR ####
###########################################

const TruncatedStableSubordinator{T<:Real} = TruncatedLevyProcess{T,StableSubordinator{T}}

# Default sampling method uses Inversion
function sample(rng::AbstractRNG, p::TruncatedStableSubordinator{T}, dt::Real) where {T}
    return sample(rng, p, dt, Inversion)
end
