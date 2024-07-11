import Distributions: ContinuousUnivariateDistribution, pdf, cdf
import QuadGK: quadgk
import SpecialFunctions: gamma, besselk

export NormalVarianceMeanProcess, VarianceGammaProcess

# HACK: should inherit from Subordinator, but a truncated subordinator does not know that
# it is a subordinator
struct NormalVarianceMeanProcess{P<:LevyProcess,T<:Real}
    subordinator::P
    μ::T
    σ::T
end

const VarianceGammaProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{GammaProcess{T},T}

################################################
#### TRUNCATED NORMAL VARIANCE MEAN PROCESS ####
################################################

const PreTruncatedNormalVarianceMeanProcess{T<:AbstractFloat} =
    NormalVarianceMeanProcess{TruncatedLevyProcess{GammaProcess{T}},T}

function sample(rng::AbstractRNG, p::PreTruncatedNormalVarianceMeanProcess, dt::Real)
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_sizes = (
        p.μ .* subordinator_path.jump_sizes .+
        p.σ .* sqrt.(subordinator_path.jump_sizes) .*
        randn(rng, length(subordinator_path.jump_sizes))
    )
    return SampleJumps(subordinator_path.jump_times, jump_sizes)
end

function sample_marginalised(rng::AbstractRNG, p::PreTruncatedNormalVarianceMeanProcess, dt::Real)
    subordinator_path = sample(rng, p.subordinator, dt)
    jump_means = p.μ .* subordinator_path.jump_sizes
    jump_variances = p.σ^2 .* subordinator_path.jump_sizes
    return MarginalisedSampleJumps(subordinator_path.jump_times, jump_means, jump_variances)
end

#################################
####  VARIANCE GAMMA PROCESS ####
#################################

const TruncatedVarianceGammaProcess{T<:AbstractFloat} = TruncatedLevyProcess{VarianceGammaProcess{T}}

struct VarianceGammaMarginal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    γ::T
    λ::T
    t::T
    # Cached values
    κ::T
    A::T
    B::T
    C::T
end

function VarianceGammaMarginal(μ::T, σ::T, γ::T, λ::T, t::T) where {T<:Real}
    κ = 1 / γ
    A = μ / σ^2
    B = sqrt(μ^2 + 2σ^2 / κ) / σ^2
    C = sqrt(σ^2 * κ / 2π) * (
        (μ^2 * κ + 2σ^2)^(1 / 4 - μ / 2κ)
    ) / gamma(t / κ)
    return VarianceGammaMarginal(μ, σ, γ, λ, t, κ, A, B, C)
end

function sample(rng::AbstractRNG, p::TruncatedVarianceGammaProcess{T}, dt::T) where {T<:AbstractFloat}
    κ = 1 / p.process.subordinator.γ
    shared_term = 0.5 * sqrt(p.process.μ^2 + 2p.process.σ^2 / κ)
    γ_pos = shared_term + p.process.μ / 2
    γ_neg = shared_term - p.process.μ / 2

    positive_process = TruncatedLevyProcess(GammaProcess(γ_pos, γ_pos^2 * κ), p.lower, p.upper)
    negative_process = TruncatedLevyProcess(GammaProcess(γ_neg, γ_neg^2 * κ), p.lower, p.upper)

    positive_jumps = sample(rng, positive_process, dt)
    negative_jumps = sample(rng, negative_process, dt)

    return SampleJumps(
        vcat(positive_jumps.jump_times, negative_jumps.jump_times),
        vcat(positive_jumps.jump_sizes, -negative_jumps.jump_sizes)
    )
end

function pdf(d::VarianceGammaMarginal, x::Real)
    t, μ, σ, γ, λ, κ, A, B, C = d.t, d.μ, d.σ, d.γ, d.λ, d.κ, d.A, d.B, d.C
    σ2 = σ^2
    θ = μ
    correction = 1 / 2 * σ2 * (θ^2 + 2σ2 / κ)^(-θ / 2κ + t / 2κ) * κ^(3 / 4 - θ / 2κ + t / κ)
    return C * abs(x)^(t / κ - 1 / 2) * exp(-A * x) * besselk(t / κ - 1 / 2, B * abs(x)) / correction
end

function cdf(d::VarianceGammaMarginal, x::Real)
    t, μ, σ, γ, λ, κ, A, B, C = d.t, d.μ, d.σ, d.γ, d.λ, d.κ, d.A, d.B, d.C
    return C * quadgk(Base.Fix1(pdf, d), 0, x)[1]
end

function marginal(p::VarianceGammaProcess{T}, t::Real) where {T<:AbstractFloat}
    return VarianceGammaMarginal(
        p.μ,
        p.σ,
        p.subordinator.γ,
        p.subordinator.λ,
        t
    )
end
