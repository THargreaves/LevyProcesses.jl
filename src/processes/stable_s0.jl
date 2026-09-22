import Distributions: cf, mean, var, location, scale, shape, insupport
import Random

export StableS0, from_s1, to_s1

"""Nolan S0 stable marginal. Near α=1, density/CDF use checked Fourier quadrature."""
struct StableS0{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    σ::T
    μ::T
    function StableS0(α::Real, β::Real, σ::Real, μ::Real)
        p = StableProcess(α, β, σ, μ)
        return new{typeof(p.α)}(p.α, p.β, p.σ, p.μ)
    end
end

stable_s0_distribution(α, β, σ, μ) = StableS0(α, β, σ, μ)

_exprel(x) = abs(x) < 1e-5 ? evalpoly(x, (1, 1/2, 1/6, 1/24, 1/120, 1/720)) : expm1(x) / x
_log1prel(x) = abs(x) < 0.01 ? sum((-x)^(n - 1) / n for n in 1:10) : log1p(x) / x
_cot_factor(δ) = cospi(δ / 2) / sinc(δ / 2)
_s0_tan_difference(α, L) = -2 / π * _cot_factor(α - 1) * L * _exprel((α - 1) * L)

_s1_shift(α, β, σ) = α == 1 ? 2β * σ / π * log(σ) : β * σ * tanpi(α / 2)

"""Construct a process from unit-time S1 parameters (the pre-S0 location convention)."""
function from_s1(α::Real, β::Real, σ::Real, μ::Real=zero(σ))
    return StableProcess(α, β, σ, μ + _s1_shift(α, β, σ))
end
from_s1(d::Stable) = from_s1(params(d)...)

"""Convert to `StableDistributions.Stable` (S1); its location is ill-conditioned near α=1."""
to_s1(d::Union{StableProcess,StableS0}) = Stable(d.α, d.β, d.σ, d.μ - _s1_shift(d.α, d.β, d.σ))

params(d::StableS0) = (d.α, d.β, d.σ, d.μ)
shape(d::StableS0) = (d.α, d.β)
location(d::StableS0) = d.μ
scale(d::StableS0) = d.σ
mean(d::StableS0) = d.α > 1 ? d.μ - _s1_shift(d.α, d.β, d.σ) : oftype(d.μ, NaN)
var(d::StableS0) = oftype(d.σ, Inf)
Base.minimum(d::StableS0) = d.α < 1 && d.β == 1 ? d.μ - _s1_shift(d.α, d.β, d.σ) : -Inf
Base.maximum(d::StableS0) = d.α < 1 && d.β == -1 ? d.μ - _s1_shift(d.α, d.β, d.σ) : Inf
insupport(d::StableS0, x::Real) = minimum(d) <= x <= maximum(d)
Base.:+(d::StableS0, a::Real) = stable_s0_distribution(d.α, d.β, d.σ, d.μ + a)
Base.:*(a::Real, d::StableS0) = stable_s0_distribution(d.α, sign(a) * d.β, abs(a) * d.σ, a * d.μ)

function cf(d::StableS0, u::Real)
    iszero(u) && return complex(one(d.α))
    q = abs(d.σ * u)
    phase = d.μ * u + d.β * sign(u) * q * _s0_tan_difference(d.α, log(q))
    return exp(complex(-q^d.α, phase))
end

# CMS algebra rearranged before subtracting the divergent S1 location.
function Random.rand(rng::AbstractRNG, d::StableS0)
    v = π * (rand(rng) - 0.5)
    w = Random.randexp(rng)
    δ = d.α - 1
    k = -2d.β / π * _cot_factor(δ)  # δ β tan(πα/2)
    a = (sin(d.α * v) - k * v * sinc(δ * v / (2π)) * sin((d.α + 1) * v / 2)) / cos(v)
    r = (cos(δ * v) - k * v * sinc(δ * v / π)) / (w * cos(v))
    logr = log(r)
    z = -δ / d.α * logr
    x = a * exp(z) - k / d.α * logr * _exprel(z)
    return d.σ * x + d.μ
end

function _s0_fourier(d::StableS0, x::Real, cumulative::Bool)
    z = (x - d.μ) / d.σ
    standard = StableS0(d.α, d.β, one(d.σ), zero(d.μ))
    limit = (-log(eps(Float64)))^(1 / d.α)
    f(u) = cumulative ? imag(cis(-u * z) * cf(standard, u)) / u :
                        real(cis(-u * z) * cf(standard, u))
    value, abserror = quadgk(f, 0, 1, limit; rtol=1e-8, atol=1e-10, maxevals=10^6)
    abserror <= max(1e-10, 1e-8 * abs(value)) ||
        error("stable Fourier quadrature did not reach its requested accuracy")
    if cumulative
        return clamp(0.5 - value / π, 0, 1)
    end
    value >= -abserror || error("stable Fourier quadrature returned a negative density")
    return max(value, 0) / (π * d.σ)
end

function pdf(d::StableS0, x::Real)
    !isfinite(x) && return isnan(x) ? NaN : zero(d.σ)
    insupport(d, x) || return zero(d.σ)
    return abs(d.α - 1) <= 0.1 ? _s0_fourier(d, x, false) : pdf(to_s1(d), x)
end
logpdf(d::StableS0, x::Real) = log(pdf(d, x))
function cdf(d::StableS0, x::Real)
    isnan(x) && return NaN
    x <= minimum(d) && return zero(d.σ)
    x >= maximum(d) && return one(d.σ)
    return abs(d.α - 1) <= 0.1 ? _s0_fourier(d, x, true) : cdf(to_s1(d), x)
end
