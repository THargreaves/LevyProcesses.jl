import Distributions: Exponential, Normal, ContinuousUnivariateDistribution
import Distributions: logpdf, pdf, cdf, params
import QuadGK: quadgk
import StableDistributions: Stable
import SpecialFunctions: gamma, factorial
import HypergeometricFunctions: pFq

export StableProcess, TruncatedStableProcess, sample_shot_noise, sample_marginalised
export StableGaussianConvolution

# TODO: add alternative constructor
struct StableProcess{T<:Real} <: LevyProcess{T}
    α::T
    μ_W::T
    σ_W::T
    # Canonical parameterisation
    β::T
    σ::T
    # Cached values
    C_α::T
end
function StableProcess(α::Real, μ_W::Real, σ_W::Real)
    α_moment = quadgk(x -> pdf(Normal(μ_W, σ_W), x) * abs(x)^α, -Inf, Inf)[1]
    α_sgn_moment = quadgk(x -> pdf(Normal(μ_W, σ_W), x) * abs(x)^α * sign(x), -Inf, Inf)[1]
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))

    σ = (α_moment / C_α)^(1 / α)
    β = α_sgn_moment / α_moment

    return StableProcess(α, μ_W, σ_W, β, σ, C_α)
end

function levy_density(p::StableProcess, x::Real)
    return (p.σ^p.α * p.C_α * (1 + p.β * sign(x)) * p.α * abs(x)^(-p.α - 1))
end

function levy_tail_mass(p::StableProcess, x::Real)
    return p.σ^p.α * p.C_α * 2 * x^(-p.α)
end

function marginal(p::StableProcess, t::Real)
    return Stable(p.α, p.β, p.σ * t^(1 / p.α), 0.0)
end

const TruncatedStableProcess{T} = TruncatedLevyProcess{T,StableProcess{T}}

# TODO: combine with regular LTM sampler
function sample_shot_noise(
    rng::AbstractRNG, p::TruncatedStableProcess{T}, dt::Real
) where {T}
    N = rand(rng, Poisson(dt * p.lower^(-p.process.α)))

    Γs = rand(rng, T, N)
    Γs *= p.lower^(-p.process.α)
    jump_sizes = Γs .^ (-1 / p.process.α)

    jump_times = rand(rng, Uniform(0, dt), length(jump_sizes))
    return SampleJumps(jump_times, jump_sizes)
end

# TODO: technically this is a different kind of truncation
function sample(rng::AbstractRNG, p::TruncatedStableProcess, dt::Real)
    shot_noise_path = sample_shot_noise(rng, p, dt)

    N = length(shot_noise_path.jump_sizes)
    μ_W = p.process.μ_W
    σ_W = p.process.σ_W

    jump_sizes = shot_noise_path.jump_sizes .* rand(rng, Normal(μ_W, σ_W), N)
    jump_times = rand(Uniform(0, dt), N)
    return SampleJumps(jump_times, jump_sizes)
end

function sample_marginalised(rng::AbstractRNG, p::TruncatedStableProcess, dt::Real)
    shot_noise_path = sample_shot_noise(rng, p, dt)

    jump_means = shot_noise_path.jump_sizes .* p.process.μ_W
    jump_variances = (shot_noise_path.jump_sizes .* p.process.σ_W) .^ 2

    return MarginalisedSampleJumps(shot_noise_path.jump_times, jump_means, jump_variances)
end

#####################################
#### Stable Gaussian Convolution ####
#####################################

struct StableGaussianConvolution{T<:Real} <: ContinuousUnivariateDistribution
    stable::Stable{T}
    normal::Normal{T}
end

function StableGaussianConvolution(S::Stable, N::Normal)
    T = promote_type(typeof(S.α), typeof(N.μ))
    return StableGaussianConvolution{T}(S, N)
end

function StableGaussianConvolution(S::Stable, σ::Real)
    return StableGaussianConvolution(S, Normal(0.0, σ))
end

function params(d::StableGaussianConvolution)
    return (d.stable, d.normal)
end

# TODO: implement adapative stopping
# TODO: considered direct logpdf computation
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

function logpdf(d::StableGaussianConvolution, x::Real; M::Int=10)
    return log(pdf(d, x; M=M))
end
