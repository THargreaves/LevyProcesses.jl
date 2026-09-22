import Distributions: MvNormal, Normal
import ForwardDiff
using CUDA
using NNlib

export LinearDynamics, LangevinDynamics
export LevyDrivenLinearSDE

export sample_conditional_marginal, conditional_marginal, NormalMixtureDrivenSDE

export UnivariateLinearDynamics, StableDrivenSDE, TruncatedStableDrivenSDE
export LangevianStableDrivenSDE, projection_marginal

using LinearAlgebra

abstract type LinearDynamics end

struct LangevinDynamics{T<:Real} <: LinearDynamics
    θ::T
end
function Base.exp(dyn::LangevinDynamics, dt::Real)
    θ = dyn.θ
    exp_val = exp(θ * dt)
    response = iszero(θ) ? dt : expm1(θ * dt) / θ
    return @SMatrix [one(exp_val) response; zero(exp_val) exp_val]
end

function compute_expAs(dyn::LangevinDynamics, dt::CuVector{T}) where {T<:Number}
    expAs = CuArray{T}(undef, 2, 2, length(dt))
    exp_vals = exp.(T(dyn.θ) * dt)
    expAs[1, 1, :] .= T(1.0)
    if iszero(dyn.θ)
        expAs[1, 2, :] .= dt
    else
        expAs[1, 2, :] .= expm1.(T(dyn.θ) .* dt) ./ T(dyn.θ)
    end
    expAs[2, 1, :] .= T(0.0)
    expAs[2, 2, :] .= exp_vals
    return expAs
end

function _sde_exprel2(x)
    # Second exponential remainder, evaluated without cancellation near zero.
    return abs(x) < 1e-4 ? 1/2 + x * (1/6 + x * (1/24 + x * (1/120 + x/720))) :
        (expm1(x) - x) / x^2
end

function _integrated_response(dyn::LangevinDynamics, h::AbstractVector, t::Real)
    z = dyn.θ * t
    return @SVector [t * h[1] + t^2 * _sde_exprel2(z) * h[2], t * _exprel(z) * h[2]]
end

struct LevyDrivenLinearSDE{P<:LevyProcess,D<:LinearDynamics,V<:AbstractVector}
    driving_process::P
    linear_dynamics::D
    noise_scaling::V
end

###############################################
#### Normal Mixture-Driven SDEs (NVM/NsM) ####
###############################################

# TODO: generalise this to arbitrary conditionally Gaussian Levy processes
const NormalMixtureDrivenSDE =
    LevyDrivenLinearSDE{P,D,T} where {P<:AbstractNormalMixtureProcess,D,T}

function sample_conditional_marginal(
    rng::AbstractRNG,
    sde::NormalMixtureDrivenSDE,
    t::Real;
    x0::Union{Nothing,Vector}=nothing,
)
    # TODO: this needs to be generalised for other cases
    subordinator_path = sample(rng, sde.driving_process.subordinator, t)
    return conditional_marginal(subordinator_path, sde, t; x0)
end

function conditional_marginal(
    subordinator_path::SampleJumps,
    sde::NormalMixtureDrivenSDE,
    t::Real;
    x0::Union{Nothing,AbstractVector}=nothing,
)
    m, S = conditional_marginal_parameters(subordinator_path, sde, t; x0)
    # HACK: Force PSD
    S = (S + S') / 2 + 1e-4 * I
    return MvNormal(m, S)
end

function conditional_marginal_parameters(
    subordinator_path::SampleJumps,
    sde::NormalMixtureDrivenSDE,
    t::Real;
    x0::Union{Nothing,AbstractVector}=nothing,
)
    m, S = unscaled_conditional_marginal_parameters(subordinator_path, sde, t)
    m *= sde.driving_process.μ
    S *= sde.driving_process.σ^2
    drift = deterministic_drift(sde.driving_process)
    iszero(drift) || (m += drift * _integrated_response(sde.linear_dynamics, sde.noise_scaling, t))
    isnothing(x0) || (m += exp(sde.linear_dynamics, t) * x0)  # not scaled by μ_W
    return m, S
end

function unscaled_conditional_marginal_parameters(
    subordinator_path::SampleJumps{T},
    sde::NormalMixtureDrivenSDE,
    t::Real;
    x0::Union{Nothing,AbstractVector{T}}=nothing,
) where {T}
    dyn = sde.linear_dynamics
    D = length(sde.noise_scaling)
    process = sde.driving_process

    m = @SVector zeros(T, D)
    S = @SMatrix zeros(T, D, D)
    for (v, z) in zip(subordinator_path.jump_times, subordinator_path.jump_sizes)
        ft = exp(dyn, (t - v)) * sde.noise_scaling
        m += ft * unscaled_jump_mean(process, z)
        S += ft * ft' * unscaled_jump_variance(process, z)
    end

    isnothing(x0) || (m += exp(dyn, t) * x0)

    return m, S
end

function sum_blocks!(μs, Σs, μ, Σ, offsets, num_runs_ref)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    K = num_runs_ref[]
    for i in index:stride:K
        start = i == 1 ? 1 : offsets[i - 1] + 1
        finish = offsets[i]
        for j in start:finish
            @inbounds μ[1, i] += μs[1, j]
            @inbounds μ[2, i] += μs[2, j]
            @inbounds Σ[1, 1, i] += Σs[1, 1, j]
            @inbounds Σ[1, 2, i] += Σs[1, 2, j]
            @inbounds Σ[2, 2, i] += Σs[2, 2, j]
        end
        # Use symmetry to fill in remainder
        @inbounds Σ[2, 1, i] = Σ[1, 2, i]
    end
end

export conditional_marginal_parameters

function conditional_marginal_parameters(
    subordinator_paths::RaggedBatchSampleJumps{T}, sde::NormalMixtureDrivenSDE, t::Real
) where {T}
    μs, Σs = calc_jump_contributions(
        subordinator_paths.jump_times, subordinator_paths.jump_sizes, sde, t
    )

    N = length(subordinator_paths.offsets)
    num_runs_ref = CuArray([N])
    μ = CUDA.zeros(T, 2, N)
    Σ = CUDA.zeros(T, 2, 2, N)
    num_runs_ref = convert.(Int32, num_runs_ref)
    CUDA.@cuda threads = 256 blocks = 4096 sum_blocks!(
        μs, Σs, μ, Σ, subordinator_paths.offsets, num_runs_ref
    )

    _add_deterministic_mean!(μ, sde, t)
    return μ, Σ
end

function _add_deterministic_mean!(m::CuMatrix{T}, sde::NormalMixtureDrivenSDE, t::Real) where {T}
    drift = deterministic_drift(sde.driving_process)
    if !iszero(drift)
        mean = drift * _integrated_response(sde.linear_dynamics, sde.noise_scaling, t)
        m .+= CuArray(T.(collect(mean)))
    end
    return m
end

function calc_jump_contributions(
    jump_times::CuVector, jump_sizes::CuVector, sde::NormalMixtureDrivenSDE, t::Real
)
    dyn = sde.linear_dynamics
    μ_W, σ_W = sde.driving_process.μ, sde.driving_process.σ
    tot_N = length(jump_times)

    expAs = compute_expAs(dyn, t .- jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    μs = μ_W * expA_h .* jump_sizes'

    # Compute variance scaling based on process type
    variance_scaling = compute_variance_scaling(sde.driving_process, jump_sizes)
    Σs = (
        σ_W^2 *
        NNlib.batched_mul(reshape(expA_h, 2, 1, tot_N), reshape(expA_h, 1, 2, tot_N)) .*
        variance_scaling
    )
    return μs, Σs
end

# Variance scaling for NVM: linear in z
function compute_variance_scaling(::NormalVarianceMeanProcess, jump_sizes::CuVector)
    return reshape(jump_sizes, 1, 1, length(jump_sizes))
end

# Variance scaling for NsM: quadratic in z
function compute_variance_scaling(::NσMProcess, jump_sizes::CuVector)
    return reshape(jump_sizes .^ 2, 1, 1, length(jump_sizes))
end

function conditional_marginal_parameters(
    subordinator_paths::RegularBatchSampleJumps, sde::NormalMixtureDrivenSDE, t::Real
)
    # Flatten jumps
    jump_sizes_flat = reshape(
        subordinator_paths.jump_sizes, prod(size(subordinator_paths.jump_sizes))
    )
    jump_times_flat = reshape(
        subordinator_paths.jump_times, prod(size(subordinator_paths.jump_times))
    )

    μs, Σs = calc_jump_contributions(jump_times_flat, jump_sizes_flat, sde, t)

    # Reshape and reduce
    μs = reshape(μs, size(μs, 1), size(subordinator_paths.jump_sizes)...)
    Σs = reshape(Σs, size(Σs)[1:2]..., size(subordinator_paths.jump_sizes)...)

    μ = dropdims(sum(μs; dims=2); dims=2)
    Σ = dropdims(sum(Σs; dims=3); dims=3)

    _add_deterministic_mean!(μ, sde, t)
    return μ, Σ
end

function unscaled_conditional_marginal_parameters(
    subordinator_paths::RaggedBatchSampleJumps{T}, sde::NormalMixtureDrivenSDE, t::Real
) where {T}
    dyn = sde.linear_dynamics
    process = sde.driving_process

    expAs = compute_expAs(dyn, t .- subordinator_paths.jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    ms = expA_h .* subordinator_paths.jump_sizes'

    # Compute variance scaling based on process type
    variance_scaling = compute_variance_scaling(process, subordinator_paths.jump_sizes)
    Ss = (
        NNlib.batched_mul(
            reshape(expA_h, 2, 1, subordinator_paths.tot_N),
            reshape(expA_h, 1, 2, subordinator_paths.tot_N),
        ) .* variance_scaling
    )

    N = length(subordinator_paths.offsets)
    num_runs_ref = CuArray([N])
    m = CUDA.zeros(T, 2, N)
    S = CUDA.zeros(T, 2, 2, N)
    num_runs_ref = convert.(Int32, num_runs_ref)
    CUDA.@cuda threads = 256 blocks = 4096 sum_blocks!(
        ms, Ss, m, S, subordinator_paths.offsets, num_runs_ref
    )

    return m, S
end

function unscaled_conditional_marginal_parameters(
    subordinator_paths::RegularBatchSampleJumps{T}, sde::NormalMixtureDrivenSDE, t::Real
) where {T}
    dyn = sde.linear_dynamics
    process = sde.driving_process

    # Flatten jumps
    jump_sizes_flat = reshape(
        subordinator_paths.jump_sizes, prod(size(subordinator_paths.jump_sizes))
    )
    jump_times_flat = reshape(
        subordinator_paths.jump_times, prod(size(subordinator_paths.jump_times))
    )
    tot_N = length(jump_times_flat)

    expAs = compute_expAs(dyn, t .- jump_times_flat)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    ms = expA_h .* jump_sizes_flat'

    # Compute variance scaling based on process type
    variance_scaling = compute_variance_scaling(process, jump_sizes_flat)
    Ss = (
        NNlib.batched_mul(reshape(expA_h, 2, 1, tot_N), reshape(expA_h, 1, 2, tot_N)) .*
        variance_scaling
    )

    # Reshape and reduce
    ms = reshape(ms, size(ms, 1), size(subordinator_paths.jump_sizes)...)
    Ss = reshape(Ss, size(Ss)[1:2]..., size(subordinator_paths.jump_sizes)...)

    m = dropdims(sum(ms; dims=2); dims=2)
    S = dropdims(sum(Ss; dims=3); dims=3)

    return m, S
end

############################
#### Stable-Driven SDEs ####
############################

struct UnivariateLinearDynamics{T<:Real} <: LinearDynamics
    a::T
end

struct TruncatedStableDrivenSDE{P<:TruncatedStableProcess,D<:UnivariateLinearDynamics}
    driving_process::P
    linear_dynamics::D
end

struct StableDrivenSDE
    driving_process::StableProcess
    linear_dynamics::UnivariateLinearDynamics
end

function sample_conditional_marginal(
    rng::AbstractRNG, sde::TruncatedStableDrivenSDE, x0::Float64, t::Real
)
    shot_noise_path = sample_shot_noise(rng, sde.driving_process, t)
    return conditional_marginal(shot_noise_path, sde, x0, t)
end

function conditional_marginal(
    shot_noise_path::SampleJumps, sde::TruncatedStableDrivenSDE, x0::Float64, t::Real
)
    throw(ArgumentError("physical stable jumps are not Gaussian mixture marks; use NormalMixtureDrivenSDE with an explicit latent process"))
end

function marginal(sde::StableDrivenSDE, x0::Real, t::Real)
    dyn, p = sde.linear_dynamics, sde.driving_process
    isfinite(t) && t > 0 || throw(ArgumentError("t must be finite and positive"))
    if iszero(dyn.a)
        d = marginal(p, t)
        return stable_s0_distribution(d.α, d.β, d.σ, d.μ + x0)
    end
    endpoint = exp(dyn.a * t)
    response_scale = hypot(one(endpoint), endpoint)
    moment(α) = if dyn.a > 0
        t * (endpoint / response_scale)^α * _exprel(-α * dyn.a * t)
    else
        t * inv(response_scale)^α * _exprel(α * dyn.a * t)
    end
    absolute = moment(p.α)
    linear = moment(one(p.α))
    location = response_scale * _stable_response_location(p, absolute, absolute, linear, moment)
    return stable_s0_distribution(p.α, p.β, p.σ * response_scale * absolute^(1 / p.α),
                                  location + x0 * exp(dyn.a * t))
end

##############################
#### Langevian Stable SDE ####
##############################

struct LangevianStableDrivenSDE{P<:StableProcess,D<:LangevinDynamics}
    driving_process::P
    dynamics::D
end

# With u = g/B, ds = du / (θ(u - 1)). These real antiderivatives
# integrate |u|^α/(u-1) and its signed version across a response zero; the imaginary
# branch constant of ₂F₁ cancels between endpoints on the same side of u=1.
function _stable_F(u::Real, α::Real)
    iszero(u) && return zero(u)
    value = abs(u)^(α + 1) / (α + 1) * real(
        pFq((one(complex(u)), complex(α + 1)), (complex(α + 2),), complex(u)),
    )
    return -sign(u) * value
end
_signed_stable_F(u::Real, α::Real) = sign(u) * _stable_F(u, α)

function _stable_response_moments(θ, t, u, α)
    g0 = u[2]
    if iszero(θ)
        slope = u[1]
        iszero(slope) && return (t * abs(g0)^α, t * sign(g0) * abs(g0)^α)
        g1 = g0 + slope * t
        absolute = (sign(g1) * abs(g1)^(α + 1) - sign(g0) * abs(g0)^(α + 1)) / ((α + 1) * slope)
        signed = (abs(g1)^(α + 1) - abs(g0)^(α + 1)) / ((α + 1) * slope)
        return absolute, signed
    end
    A, B = u[1] / θ + u[2], -u[1] / θ
    iszero(A) && return (t * abs(B)^α, t * sign(B) * abs(B)^α)
    if iszero(B)
        absolute = abs(A)^α * t * _exprel(α * θ * t)
        return absolute, sign(A) * absolute
    end
    ratio = A / B * exp(max(θ * t, zero(θ)))
    if abs(ratio) <= 1/4
        # Near a constant response the antiderivative endpoints approach its
        # logarithmic singularity. The convergent binomial expansion avoids
        # subtracting those values, retaining the finite exponential component.
        total, coefficient = t, one(ratio)
        # Fixed terms retain α derivatives when a binomial coefficient vanishes.
        for k in 1:32
            coefficient *= (α - k + 1) / k * ratio
            term = coefficient * t * _exprel(-k * abs(θ) * t)
            total += term
        end
        absolute = abs(B)^α * total
        return absolute, sign(B) * absolute
    end
    # Evaluate g directly, rather than subtracting nearly equal z+c when θ≈0.
    g1 = u[1] * t * _exprel(θ * t) + u[2] * exp(θ * t)
    v0, v1 = g0 / B, g1 / B
    factor = abs(B)^α / θ
    absolute = factor * (_stable_F(v1, α) - _stable_F(v0, α))
    signed = sign(B) * factor * (_signed_stable_F(v1, α) - _signed_stable_F(v0, α))
    return absolute, signed
end

_moment_derivative(f, x, ::Val{0}) = f(x)
function _moment_derivative(f, x, ::Val{N}) where {N}
    return ForwardDiff.derivative(y -> _moment_derivative(f, y, Val(N - 1)), x)
end

function _stable_response_location(p, absolute, signed, linear, signed_moment)
    iszero(p.β) && return p.μ * linear
    δ = p.α - 1
    logscale = log(absolute) / p.α
    divided_difference = if abs(δ) <= 1e-3
        # Analytic moment derivatives resolve the removable singularity at α=1.
        # Responses are normalized to |g|≤1 before forming this local series.
        α0 = one(p.α)
        d1 = _moment_derivative(signed_moment, α0, Val(1))
        d2 = _moment_derivative(signed_moment, α0, Val(2))
        d3 = _moment_derivative(signed_moment, α0, Val(3))
        d4 = _moment_derivative(signed_moment, α0, Val(4))
        dj = evalpoly(δ, (d1, d2 / 2, d3 / 6, d4 / 24))
        q = -δ * logscale
        exp(q) * dj - linear * logscale * _exprel(q)
    else
        (signed * exp(-δ * logscale) - linear) / δ
    end
    return p.μ * linear - 2p.β * p.σ / π * _cot_factor(δ) * divided_difference
end

"""
    projection_marginal(sde, t, u)

S0 stable marginal along the unit direction `u / norm(u)`, from a zero initial
state. Analytic response moments and a local derivative expansion give a
quadrature-free location correction continuous through α=1.
"""
function projection_marginal(sde::LangevianStableDrivenSDE, t::Real, u::AbstractVector)
    length(u) == 2 || throw(ArgumentError("projection vector must have length 2"))
    all(isfinite, u) && norm(u) > 0 || throw(ArgumentError("projection direction must be finite and nonzero"))
    isfinite(t) && t > 0 || throw(ArgumentError("t must be finite and positive"))
    u = u / norm(u)
    θ, p = sde.dynamics.θ, sde.driving_process
    response(s) = u[1] * s * _exprel(θ * s) + u[2] * exp(θ * s)
    response_scale = hypot(u[2], response(t))
    normalized = u / response_scale
    absolute, signed = _stable_response_moments(θ, t, normalized, p.α)
    linear = dot(normalized, _integrated_response(sde.dynamics, [0, 1], t))
    signed_moment(α) = last(_stable_response_moments(θ, t, normalized, α))
    location = response_scale * _stable_response_location(p, absolute, signed, linear, signed_moment)
    β = clamp(p.β * signed / absolute, -one(p.β), one(p.β))
    return stable_s0_distribution(p.α, β, p.σ * response_scale * absolute^(1 / p.α), location)
end
