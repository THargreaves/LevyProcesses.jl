import Distributions: ContinuousUnivariateDistribution, pdf, cdf
import QuadGK: quadgk
import SpecialFunctions: gamma, besselk

export NormalVarianceMeanProcess, VarianceGammaProcess

# HACK: should inherit from Subordinator, but a truncated subordinator does not know that
# it is a subordinator
struct NormalVarianceMeanProcess{T<:Real,P<:LevyProcess{T}} <: LevyProcess{T}
    subordinator::P
    μ::T
    σ::T
end

const VarianceGammaProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{T,GammaProcess{T}}

# HACK: temporary fix whilst tail mass is mandatory
function levy_tail_mass(p::VarianceGammaProcess{T}, x::T) where {T<:AbstractFloat}
    return 0.0
end

################################################
#### TRUNCATED NORMAL VARIANCE MEAN PROCESS ####
################################################

const PreTruncatedNormalVarianceMeanProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{
    T,TruncatedLevyProcess{T,GammaProcess{T}}
}

function sample(rng::AbstractRNG, p::PreTruncatedNormalVarianceMeanProcess, dt::Real)
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_sizes = (
        p.μ .* subordinator_path.jump_sizes .+
        p.σ .* sqrt.(subordinator_path.jump_sizes) .*
        randn(rng, length(subordinator_path.jump_sizes))
    )
    return SampleJumps(subordinator_path.jump_times, jump_sizes)
end

function sample_marginalised(
    rng::AbstractRNG, p::PreTruncatedNormalVarianceMeanProcess, dt::Real
)
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_means = p.μ .* subordinator_path.jump_sizes
    jump_variances = p.σ^2 .* subordinator_path.jump_sizes
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
    rng::AbstractRNG, p::TruncatedVarianceGammaProcess{T}, dt::T
) where {T<:AbstractFloat}
    # Scale subordinator to have unit mean
    κ = 1 / p.process.subordinator.λ
    dt *= p.process.subordinator.γ / p.process.subordinator.λ

    # TODO: avoid recomputing these for every sample
    A = p.process.μ / p.process.σ^2
    B = sqrt(p.process.μ^2 + 2p.process.σ^2 / κ) / p.process.σ^2

    positive_process = TruncatedLevyProcess(GammaProcess(1 / κ, B - A), p.lower, p.upper)
    negative_process = TruncatedLevyProcess(GammaProcess(1 / κ, A + B), p.lower, p.upper)

    positive_jumps = sample(rng, positive_process, dt)
    negative_jumps = sample(rng, negative_process, dt)

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
    return VarianceGammaMarginal(p.μ, p.σ, p.subordinator.γ, p.subordinator.λ, t)
end
