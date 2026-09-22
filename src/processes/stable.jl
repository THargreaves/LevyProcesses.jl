import Distributions: Exponential, Normal, ContinuousUnivariateDistribution
import Distributions: logpdf, pdf, cdf, params
import QuadGK: quadgk
import StableDistributions: Stable
import SpecialFunctions: gamma, factorial
import HypergeometricFunctions: pFq
import Roots: find_zero, Bisection

export StableProcess, TruncatedStableProcess, sample_shot_noise, sample_marginalised
export StableGaussianConvolution
export to_nsm

"""Stable Lévy process with the S1 zero-location marginal; `0 < α < 2`, `α ≠ 1`."""
struct StableProcess{T<:Real} <: LevyProcess{T}
    α::T
    β::T
    σ::T
    C_α::T
end
function StableProcess(α::Real, β::Real, σ::Real)
    isfinite(α) && 0 < α < 2 && α != 1 ||
        throw(ArgumentError("α must lie in (0, 2), excluding 1"))
    isfinite(β) && abs(β) <= 1 || throw(ArgumentError("β must lie in [-1, 1]"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("σ must be finite and positive"))
    α, β, σ = promote(float(α), float(β), float(σ))
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))
    return StableProcess(α, β, σ, oftype(α, C_α))
end

function levy_density(p::StableProcess, x::Real)
    iszero(x) && return zero(p.α)
    weight = 1 + p.β * sign(x)
    iszero(weight) && return zero(p.α)
    return p.σ^p.α * p.C_α * weight * p.α / 2 * abs(x)^(-p.α - 1)
end
log_levy_density(p::StableProcess, x::Real) = log(levy_density(p, x))

function levy_tail_mass(p::StableProcess, x::Real)
    x >= 0 || throw(DomainError(x, "absolute jump cutoff must be non-negative"))
    return p.σ^p.α * p.C_α * x^(-p.α)
end

# Canonical truncation h(x) = x 1{|x| ≤ 1}.
levy_drift(p::StableProcess) = p.α * p.σ^p.α * p.C_α * p.β / (1 - p.α)

function marginal(p::StableProcess, t::Real)
    isfinite(t) && t > 0 || throw(ArgumentError("time must be finite and positive"))
    return Stable(p.α, p.β, p.σ * t^(1 / p.α), 0.0)
end

const TruncatedStableProcess{T} = TruncatedLevyProcess{T,StableProcess{T}}

"""Sample physical jumps satisfying `lower ≤ abs(x) ≤ upper`; no drift is added."""
function sample(rng::AbstractRNG, p::TruncatedStableProcess{T}, dt::Real) where {T}
    N = rand(rng, Poisson(_retained_jump_intensity(p, dt)))
    times = T(dt) .* rand(rng, T, N)
    Γs = p.upper_tail_mass .+ (1 .- rand(rng, T, N)) .* p.mass
    sizes = (p.process.σ^p.process.α * p.process.C_α ./ Γs) .^ (1 / p.process.α)
    signs = ifelse.(rand(rng, T, N) .< (1 + p.process.β) / 2, one(T), -one(T))
    return SampleJumps(times, T.(sizes .* signs))
end

function sample_shot_noise(rng::AbstractRNG, p::TruncatedStableProcess, dt::Real)
    throw(ArgumentError("Gaussian-mark shot noise requires an explicit NσMProcess; use to_nsm where supported"))
end

function sample_marginalised(rng::AbstractRNG, p::TruncatedStableProcess, dt::Real)
    throw(ArgumentError("Gaussian-mark marginalisation requires an explicit NσMProcess; use to_nsm where supported"))
end

#####################################
#### Stable Gaussian Convolution ####
#####################################

struct StableGaussianConvolution{T<:Real,S<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    stable::S
    normal::Normal{T}
end

function StableGaussianConvolution(S::Union{Stable,StableS0}, N::Normal)
    T = promote_type(typeof(S.α), typeof(N.μ))
    return StableGaussianConvolution{T,typeof(S)}(S, Normal(T(N.μ), T(N.σ)))
end

function StableGaussianConvolution(S::Union{Stable,StableS0}, σ::Real)
    return StableGaussianConvolution(S, Normal(0.0, σ))
end

function params(d::StableGaussianConvolution)
    return (d.stable, d.normal)
end

# TODO: implement adapative stopping
# TODO: considered direct logpdf computation
# S0 avoids the ill-conditioned S1 location in the finite-series formula near α=1.
function pdf(d::StableGaussianConvolution{T,S}, x::Real; rtol=1e-8, atol=1e-10) where {T,S<:StableS0}
    stable, normal = d.stable, d.normal
    iszero(normal.σ) && return pdf(stable, x - normal.μ)
    !isfinite(x) && return isnan(x) ? NaN : zero(T)
    scale = max(stable.σ, normal.σ)
    decay = -log(eps(Float64))
    cutoff = min(decay^(1 / stable.α) * scale / stable.σ,
                 sqrt(2decay) * scale / normal.σ)
    centered = StableS0(stable.α, stable.β, stable.σ, zero(stable.μ))
    f(u) = real(cis(-u * (x - stable.μ - normal.μ) / scale) *
                cf(centered, u / scale)) * exp(-0.5 * (normal.σ * u / scale)^2)
    value, error = quadgk(f, 0, 1, cutoff; rtol, atol)
    error <= max(atol, rtol * abs(value)) || throw(ErrorException("convolution quadrature did not converge"))
    value >= -error || throw(ErrorException("convolution quadrature returned a negative density"))
    return max(value, zero(value)) / (π * scale)
end

function pdf(d::StableGaussianConvolution, x::Real; M::Int=10)
    S = d.stable
    N = d.normal
    α, β, σ, μ = S.α, S.β, S.σ, S.μ

    σy = N.σ
    μy = N.μ

    γy = σy / sqrt(2)
    p = γy^2
    q = x - μ - μy
    B = β * tan(π * α / 2)

    T = 0.0
    for m in 0:M
        v1 = (α * m + 1) / 2
        term_1 =
            p^(-v1) *
            gamma(v1) *
            pFq([v1], [1 / 2], -(q^2) / (4 * p)) *
            real((-1 + im * B)^m)

        v2 = (α * m + 2) / 2
        term_2 =
            q *
            p^(-v2) *
            gamma(v2) *
            pFq([v2], [3 / 2], -(q^2) / (4 * p)) *
            imag((-1 + im * B)^m)

        T += (σ^(α * m) / factorial(m)) * (term_1 + term_2)
    end

    return T / (2π)
end

function logpdf(d::StableGaussianConvolution, x::Real; kwargs...)
    return log(pdf(d, x; kwargs...))
end

###################################################
#### Stable to Normal Scale Mixture Conversion ####
###################################################

"""
    _beta_from_lambda(λ::Real, α::Real) -> Real

Compute β as a function of λ = μ/σ (and α).
"""
function _beta_from_lambda(λ::Real, α::Real)
    z = -λ^2 / 2
    K = 2^(0.5 - α) * sqrt(π) * gamma(1 + α) / gamma((α + 1) / 2)^2
    num = pFq(((1 - α) / 2,), (3 / 2,), z)
    den = pFq((-(α / 2),), (1 / 2,), z)
    return λ * K * num / den
end

"""
    _lambda_from_beta(β::Real, α::Real; atol=1e-10, rtol=1e-10) -> Real

Solve for λ = μ/σ given β and α using bisection (β is monotonic in λ).
"""
function _lambda_from_beta(β::Real, α::Real; atol=1e-10, rtol=1e-10)
    if abs(β) < atol
        return 0.0
    end

    βabs = abs(β)
    f(λ) = _beta_from_lambda(λ, α) - βabs

    # Bracket root on [0, L]
    a = 0.0
    fa = f(a)
    b = 1.0
    fb = f(b)

    while fa * fb > 0
        b *= 2
        fb = f(b)
        if b > 100
            error("Could not bracket root for λ (β = $βabs, α = $α)")
        end
    end

    λpos = find_zero(f, (a, b), Bisection(); atol=atol, rtol=rtol)
    return sign(β) * λpos
end

"""
    _sigma_from_gamma(γ::Real, α::Real, λ::Real) -> Real

Recover σ from the stable scale parameter γ given α and λ = μ/σ.
Uses the relationship: E[|W|^α] = γ^α * C_α.
"""
function _sigma_from_gamma(γ::Real, α::Real, λ::Real)
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))
    C = 2^(α / 2) * gamma((α + 1) / 2) / sqrt(π)
    z = -λ^2 / 2
    J = pFq((-(α / 2),), (1 / 2,), z)
    return (γ^α * C_α / (C * J))^(1 / α)
end

"""
    to_nsm(p::StableProcess; C=p.α) -> NσMProcess

Convert an S0 stable process to a Gaussian-mark scale mixture, preserving its
location through a deterministic drift. Requires `α < 1` and `abs(β) < 1`;
indices above one require a compensated series, not a positive subordinator.

For latent Lévy density `C * z^(-1-α)` and Gaussian mark `W = μ + σZ`, the
physical positive/negative jump coefficients are `C * E[(±W)_+^α]`. Their ratio
fixes stable skewness; their sum fixes stable scale. We solve for `μ/σ` by
bisection and recover `σ` from the absolute α-moment of a Gaussian, following
the series representation of Samorodnitsky and Taqqu (1994), §1.4.

`C > 0` sets the latent subordinator intensity. Changing it rescales both mark
parameters by `(α/C)^(1/α)` relative to `C=α`, leaving the physical law unchanged.
The returned `μ` is the Gaussian-mark mean, not the S0 location. The returned
`drift` is the corresponding S1 location for this finite-variation representation.
When truncating its subordinator, preserve all three fields, for example
`NσMProcess(truncated, q.μ, q.σ; drift=q.drift)` for `q = to_nsm(p)`.
"""
function to_nsm(p::StableProcess; C=p.α)
    p.α < 1 || throw(ArgumentError("to_nsm requires α < 1 for a stable subordinator"))
    abs(p.β) < 1 || throw(ArgumentError("to_nsm requires abs(β) < 1 for non-degenerate Gaussian marks"))
    isfinite(C) && C > 0 || throw(ArgumentError("C must be finite and positive"))
    α = p.α
    β = p.β
    γ = p.σ

    # Inverse formulas to get μ̃, σ̃ (for the scaled marks)
    λ = _lambda_from_beta(β, α)
    σ̃ = _sigma_from_gamma(γ, α, λ)
    μ̃ = λ * σ̃

    # Absorb the chosen subordinator scaling into the mark parameters
    sf = (α / C)^(-1 / α)
    μ = μ̃ / sf
    σ = σ̃ / sf

    S = StableSubordinator(α, C)
    return NσMProcess(S, μ, σ; drift=to_s1(p).μ)
end
