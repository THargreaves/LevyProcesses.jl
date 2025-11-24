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

# TODO: add alternative constructor
struct StableProcess{T<:Real} <: LevyProcess{T}
    α::T
    β::T
    σ::T
    # Cached values
    C_α::T
end
function StableProcess(α::Real, β::Real, σ::Real)
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))

    return StableProcess(α, β, σ, C_α)
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
    # jump_times = rand(Uniform(0, dt), N)
    # Generate sorted jump times
    jump_times = rand(rng, Exponential(1.0), N)
    tot = 0.0
    # Compute cummulative sum
    for i in 1:N
        tot += jump_times[i]
        jump_times[i] = tot
    end
    tot += rand(rng, Exponential(1.0))
    jump_times .= dt .* jump_times ./ tot
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

Convert a StableProcess to an equivalent NσMProcess with Gaussian marks.

The conversion uses the inverse of the series representation relationship for stable
distributions (Samorodnitsky and Taqqu, 1994 [1.4.10]) with Gaussian marks.

Given a stable process with parameters (α, β, γ), this function computes the Gaussian mark 
parameters (μ, σ) such that an NσM process subordinated by a StableSubordinator with the
same α and scale parameter C produces an equivalent stable marginal distribution.

The ratio λ = μ/σ is determined by solving for β numerically using a bisection method,
leveraging the monotonic relationship between β and λ. The pair (μ, σ) is then recovered
from λ and γ.

The subordinator levy measure parameter `C` can be specified; by default C = α
is used which results in a scaling factor of 1 between the theoretical and
implementation parameters.

# Arguments
- `p::StableProcess`: The stable process to convert
- `C::Real`: The levy measure parameter for the subordinator (default: p.α)

# Returns
- `NσMProcess`: An equivalent Normal Scale Mixture process with Gaussian marks
"""
function to_nsm(p::StableProcess; C=p.α)
    α = p.α
    β = p.β
    γ = p.σ  # call this γ for clarity    

    # Inverse formulas to get μ̃, σ̃ (for the scaled marks)
    λ = _lambda_from_beta(β, α)
    σ̃ = _sigma_from_gamma(γ, α, λ)
    μ̃ = λ * σ̃

    # Absorb the chosen subordinator scaling into the mark parameters
    sf = (α / C)^(-1 / α)
    μ = μ̃ / sf
    σ = σ̃ / sf

    S = StableSubordinator(α, C)
    return NσMProcess(S, μ, σ)
end
