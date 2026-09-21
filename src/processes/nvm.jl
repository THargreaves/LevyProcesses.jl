import Distributions: ContinuousUnivariateDistribution, pdf, cdf
import QuadGK: quadgk
import SpecialFunctions: gamma, besselk
import HypergeometricFunctions: pFq

export NormalVarianceMeanProcess, VarianceGammaProcess, NσMProcess
export AbstractNormalMixtureProcess
export to_stable

abstract type AbstractNormalMixtureProcess{T<:Real} <: LevyProcess{T} end

# HACK: should inherit from Subordinator, but a truncated subordinator does not know that
# it is a subordinator
struct NormalVarianceMeanProcess{T<:Real,P<:LevyProcess{T}} <:
       AbstractNormalMixtureProcess{T}
    subordinator::P
    μ::T
    σ::T
    function NormalVarianceMeanProcess{T,P}(subordinator::P, μ::Real, σ::Real) where {T<:Real,P<:LevyProcess{T}}
        isfinite(μ) || throw(ArgumentError("μ must be finite"))
        isfinite(σ) && σ >= 0 || throw(ArgumentError("σ must be finite and non-negative"))
        μ, σ = T(μ), T(σ)
        isfinite(μ) || throw(ArgumentError("μ must be finite"))
        isfinite(σ) && σ >= 0 || throw(ArgumentError("σ must be finite and non-negative"))
        return new{T,P}(subordinator, μ, σ)
    end
end

function NormalVarianceMeanProcess(subordinator::P, μ::Real, σ::Real) where {T<:Real,P<:LevyProcess{T}}
    return NormalVarianceMeanProcess{T,P}(subordinator, μ, σ)
end

const VarianceGammaProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{T,GammaProcess{T}}

# X is the difference of independent gamma processes with intensity γ.
function _vg_rates(p::VarianceGammaProcess)
    μ, σ, λ = p.μ, p.σ, p.subordinator.λ
    if iszero(σ)
        positive = μ > 0 ? λ / μ : oftype(λ, Inf)
        negative = μ < 0 ? -λ / μ : oftype(λ, Inf)
        return positive, negative
    end
    r = hypot(μ, σ * sqrt(2λ))
    # Rationalise the smaller numerator to avoid cancellation.
    positive = μ >= 0 ? 2λ / (r + μ) : (r - μ) / σ^2
    negative = μ <= 0 ? 2λ / (r - μ) : (r + μ) / σ^2
    return positive, negative
end

function log_levy_density(p::VarianceGammaProcess, x::Real)
    iszero(x) && return -Inf
    positive, negative = _vg_rates(p)
    rate = x > 0 ? positive : negative
    isinf(rate) && return -Inf
    return log(p.subordinator.γ) - log(abs(x)) - rate * abs(x)
end

levy_density(p::VarianceGammaProcess, x::Real) = exp(log_levy_density(p, x))

function levy_tail_mass(p::VarianceGammaProcess, x::Real)
    x >= 0 || throw(DomainError(x, "the absolute jump threshold must be non-negative"))
    positive, negative = _vg_rates(p)
    tail(rate) = isinf(rate) ? zero(p.μ) : levy_tail_mass(GammaProcess(p.subordinator.γ, rate), x)
    return tail(positive) + tail(negative)
end

function levy_drift(p::VarianceGammaProcess)
    positive, negative = _vg_rates(p)
    drift(rate) = isinf(rate) ? zero(p.μ) : levy_drift(GammaProcess(p.subordinator.γ, rate))
    return drift(positive) - drift(negative)
end

################################################
#### TRUNCATED NORMAL VARIANCE MEAN PROCESS ####
################################################

const PreTruncatedNormalVarianceMeanProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{
    T,TruncatedLevyProcess{T,GammaProcess{T}}
}

function sample(rng::AbstractRNG, p::AbstractNormalMixtureProcess{T}, dt::Real) where {T}
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_sizes = T[
        p.μ * unscaled_jump_mean(p, z) +
        p.σ * sqrt(unscaled_jump_variance(p, z)) * randn(rng, T) for
        z in subordinator_path.jump_sizes
    ]
    return SampleJumps(subordinator_path.jump_times, jump_sizes)
end

function sample_marginalised(rng::AbstractRNG, p::AbstractNormalMixtureProcess, dt::Real)
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_means = [p.μ * unscaled_jump_mean(p, z) for z in subordinator_path.jump_sizes]
    jump_variances = [
        p.σ^2 * unscaled_jump_variance(p, z) for z in subordinator_path.jump_sizes
    ]
    return MarginalisedSampleJumps(subordinator_path.jump_times, jump_means, jump_variances)
end

#################################
####  VARIANCE GAMMA PROCESS ####
#################################

const TruncatedVarianceGammaProcess{T<:AbstractFloat} = TruncatedLevyProcess{
    T,VarianceGammaProcess{T}
}

struct VarianceGammaMarginal{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    λ::T
    # Cached parameters
    γ::T
end

function VarianceGammaMarginal(μ::T, σ::T, γ::T, λ::T, t::T) where {T<:Real}
    # Scale subordinator to have unit mean
    κ = 1 / λ
    t *= γ / λ
    # Variance-gamma distribution parameters
    α = sqrt(μ^2 + 2σ^2 / κ) / σ^2
    β = μ / σ^2
    λ = t / κ
    γ = sqrt(α^2 - β^2)
    return VarianceGammaMarginal(α, β, λ, γ)
end

function sample(
    rng::AbstractRNG, p::TruncatedVarianceGammaProcess{T}, dt::Real
) where {T<:AbstractFloat}
    isfinite(dt) && dt >= 0 || throw(ArgumentError("sampling time must be finite and non-negative"))
    positive, negative = _vg_rates(p.process)
    γ = p.process.subordinator.γ
    function draw(rate)
        isinf(rate) && return SampleJumps(T[], T[])
        return sample(rng, TruncatedLevyProcess(GammaProcess(γ, rate), p.lower, p.upper), dt)
    end
    positive_jumps = draw(positive)
    negative_jumps = draw(negative)

    return SampleJumps(
        vcat(positive_jumps.jump_times, negative_jumps.jump_times),
        vcat(positive_jumps.jump_sizes, -negative_jumps.jump_sizes),
    )
end

function pdf(d::VarianceGammaMarginal, x::Real)
    α, β, γ, λ = d.α, d.β, d.γ, d.λ
    # HACK: replace with proper asymptotic expansions
    if x > 100
        return 0.0
    end
    if abs(x) < 1e-10
        x = 1e-10
    end
    return (
        γ^(2λ) * abs(x)^(λ - 0.5) * besselk(λ - 0.5, α * abs(x)) * exp(β * x) /
        (sqrt(π) * gamma(λ) * (2α)^(λ - 0.5))
    )
end

function cdf(d::VarianceGammaMarginal, x::Real)
    return quadgk(x -> pdf(d, x), -Inf, x)[1]
end

function marginal(p::VarianceGammaProcess{T}, t::Real) where {T<:AbstractFloat}
    p.σ > 0 || throw(ArgumentError("the variance-gamma marginal density requires σ > 0; use the scaled gamma law when σ = 0"))
    isfinite(t) && t > 0 || throw(ArgumentError("marginal time must be finite and positive"))
    return VarianceGammaMarginal(promote(p.μ, p.σ, p.subordinator.γ, p.subordinator.λ, float(t))...)
end

######################################
#### Normal Scale Mixture Process ####
######################################

struct NσMProcess{T<:Real,P<:LevyProcess{T}} <: AbstractNormalMixtureProcess{T}
    subordinator::P
    μ::T
    σ::T
    function NσMProcess{T,P}(subordinator::P, μ::Real, σ::Real) where {T<:Real,P<:LevyProcess{T}}
        isfinite(μ) || throw(ArgumentError("μ must be finite"))
        isfinite(σ) && σ >= 0 || throw(ArgumentError("σ must be finite and non-negative"))
        μ, σ = T(μ), T(σ)
        isfinite(μ) || throw(ArgumentError("μ must be finite"))
        isfinite(σ) && σ >= 0 || throw(ArgumentError("σ must be finite and non-negative"))
        return new{T,P}(subordinator, μ, σ)
    end
end

function NσMProcess(subordinator::P, μ::Real, σ::Real) where {T<:Real,P<:LevyProcess{T}}
    return NσMProcess{T,P}(subordinator, μ, σ)
end

################################################
#### Jump parameter computation methods #######
################################################

"""
Compute the unscaled mean contribution from a subordinator jump.
For both NVM and NsM, this is simply z.
"""
unscaled_jump_mean(::AbstractNormalMixtureProcess, z) = z

"""
Compute the unscaled variance contribution from a subordinator jump.
For NVM: variance scales linearly with z
For NsM: variance scales quadratically with z^2
"""
unscaled_jump_variance(::NormalVarianceMeanProcess, z) = z
unscaled_jump_variance(::NσMProcess, z) = z^2

# Conversion of Stable-NσM process to StableProcess using series representation
function to_stable(p::NσMProcess{T,StableSubordinator{T}}) where {T<:Real}
    α, C = p.subordinator.α, p.subordinator.C
    μ, σ = p.μ, p.σ
    if iszero(σ)
        iszero(μ) && throw(ArgumentError("the zero process has no non-degenerate stable marginal"))
        scale = abs(μ) * (C * gamma(1 - α) * cospi(α / 2) / α)^(1 / α)
        return StableProcess(α, sign(μ), scale)
    end
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))

    # Correct for scaling of subordinator
    sf = (α / C)^(-1 / α)
    μ̃ = μ * sf
    σ̃ = σ * sf

    # Compute γ, β from μ, σ, α
    z = -(μ̃ / σ̃)^2 / 2
    F_γ = pFq((-(α / 2),), (1 / 2,), z)
    γ = (σ̃^α * 2^(α / 2) * gamma((α + 1) / 2) / sqrt(π) * F_γ / C_α)^(1 / α)
    β = (
        (μ̃ / σ̃) * 2^(1 / 2 - α) * sqrt(π) * gamma(1 + α) / gamma((α + 1) / 2)^2 *
        pFq(((1 - α) / 2,), (3 / 2,), z) / F_γ
    )
    return StableProcess(α, β, γ)
end
