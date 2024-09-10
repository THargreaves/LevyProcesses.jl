import Distributions: ContinuousUnivariateDistribution, pdf, cdf
import QuadGK: quadgk
import SpecialFunctions: gamma, besselk

export NormalVarianceMeanProcess, VarianceGammaProcess

# HACK: should inherit from Subordinator, but a truncated subordinator does not know that
# it is a subordinator
struct NormalVarianceMeanProcess{P<:LevyProcess,T<:Real} <: LevyProcess
    subordinator::P
    μ::T
    σ::T
end

const VarianceGammaProcess{T<:AbstractFloat} = NormalVarianceMeanProcess{GammaProcess{T},T}

# HACK: temporary fix whilst tail mass is mandatory
function levy_tail_mass(p::VarianceGammaProcess{T}, x::T) where {T<:AbstractFloat}
    return 0.0
end

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
    # Scale subordinator to have unit mean
    κ = 1 / λ
    # Compute constants
    A = μ / σ^2
    B = sqrt(μ^2 + 2σ^2 / κ) / σ^2
    # NOTE: the normalising constant given in Cont and Tankov (2004) is incorrect and the
    # resulting density does not integrate to unity.
    C = sqrt(2 / (π * σ^2)) * (μ^2 + 2σ^2 / κ)^(1 / 4 - t / 2κ) / gamma(t / κ)
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
    v = t / κ - 1 / 2
    # Large-negative asymptotic expansion
    # TODO: calculate an exact cutoff
    # if x > 100.0
    #     # return C * (exp((-A + B) * x) * sqrt(π / 2B) * (-x)^(-0.5 + v))
    #     return 0.0
    # end
    # HACK: implement a rigorous asymptotic expansion
    cutoff = 1e-5
    bessel_arg = abs(x) < cutoff ? cutoff : abs(x)
    bessel_term = bessel_arg^v * besselk(v, B * bessel_arg)
    return C * exp(A * x) * bessel_term
end

function cdf(d::VarianceGammaMarginal, x::Real)
    return quadgk(x -> pdf(d, x), -Inf, x)[1]
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
