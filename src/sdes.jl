import Distributions: MvNormal, Normal
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
    return @SMatrix [1.0 (exp_val - 1)/θ; 0 exp_val]
end

function compute_expAs(dyn::LangevinDynamics, dt::CuVector{T}) where {T<:Number}
    expAs = CuArray{T}(undef, 2, 2, length(dt))
    exp_vals = exp.(T(dyn.θ) * dt)
    expAs[1, 1, :] .= T(1.0)
    expAs[1, 2, :] .= (exp_vals .- T(1.0)) ./ dyn.θ
    expAs[2, 1, :] .= T(0.0)
    expAs[2, 2, :] .= exp_vals
    return expAs
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

    return μ, Σ
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
    dyn = sde.linear_dynamics
    p = sde.driving_process.process
    μ_W, σ_W = p.μ_W, p.σ_W

    μ = exp(dyn.a * t) * x0
    σ2 = 0.0
    for (v, z) in zip(shot_noise_path.jump_times, shot_noise_path.jump_sizes)
        ft = exp(dyn.a * (t - v))
        μ += ft * μ_W * z
        σ2 += ft * ft' * σ_W^2 * z^2
    end

    return Normal(μ, sqrt(σ2))
end

function marginal(sde::StableDrivenSDE, x0::Float64, t::Real)
    dyn = sde.linear_dynamics
    p = sde.driving_process

    σ_new = if dyn.a == 0
        p.σ * t^p.α
    else
        p.σ * ((1 - exp(dyn.a * p.α * t)) / (-dyn.a * p.α))^(1 / p.α)
    end
    μ_new = x0 * exp(dyn.a * t)

    return Stable(p.α, p.β, σ_new, μ_new)
end

##############################
#### Langevian Stable SDE ####
##############################

struct LangevianStableDrivenSDE{P<:StableProcess,D<:LangevinDynamics}
    driving_process::P
    dynamics::D
end

function projection_marginal(sde::LangevianStableDrivenSDE, t::Real, u::AbstractVector)
    length(u) != 2 && throw(ArgumentError("Projection vector u must be of length 2."))

    # Normalise direction vector
    u = u / norm(u)

    E = exp(sde.dynamics.θ * t)
    A = u[1] / sde.dynamics.θ + u[2]
    B = -u[1] / sde.dynamics.θ
    c = B / A

    α_drive, β_drive, σ_drive = (
        sde.driving_process.α, sde.driving_process.β, sde.driving_process.σ
    )

    α_proj = α_drive
    # TODO: add shortcut for case when c ∉ (1, E)
    I = _stable_integral(E, c, α_drive)
    β_proj = β_drive * (sign(A) * _signed_stable_integral(E, c, α_drive)) / I
    σ_proj = σ_drive * abs(A) * (I / sde.dynamics.θ)^(1 / α_drive)
    μ_proj = 0.0  # TODO: generalise this

    return Stable(α_proj, β_proj, σ_proj, μ_proj)
end

"""
Computes the integral from z = 1 to t of sign(z + c) * |z + c|^a / z dz
"""
function _signed_stable_integral(t::Float64, c::Float64, α::Float64)
    # TODO: this is wrong if t < 0
    c == 0.0 && return (t^α - 1.0) / α

    u1 = (1.0 + c) / c
    ut = (t + c) / c

    return sign(c) * abs(c)^α * (_signed_F(ut, α) - _signed_F(u1, α))
end

function _signed_F(u::Float64, α::Float64)
    u == 0.0 && return 0.0

    I = if u > 0.0
        real(
            -u^(α + 1) / (α + 1) *
            pFq((1.0 + 0.0im, α + 1.0 + 0.0im), (α + 2.0 + 0.0im,), u + 0.0im),
        )
    else
        -real(
            (-u)^(α + 1) / (α + 1) *
            pFq((1.0 + 0.0im, α + 1.0 + 0.0im), (α + 2.0 + 0.0im,), u + 0.0im),
        )
    end

    return I
end

"""
Computes the integral from z = 1 to t of |z + c|^a / z dz
"""
function _stable_integral(t::Float64, c::Float64, α::Float64)
    c == 0.0 && return (t^α - 1.0) / α

    u1 = (1.0 + c) / c
    ut = (t + c) / c

    return abs(c)^α * (_F(ut, α) - _F(u1, α))
end

function _F(u::Float64, α::Float64)
    u == 0.0 && return 0.0

    I = if u > 0.0
        real(
            -u^(α + 1) / (α + 1) *
            pFq((1.0 + 0.0im, α + 1.0 + 0.0im), (α + 2.0 + 0.0im,), u + 0.0im),
        )
    else
        real(
            (-u)^(α + 1) / (α + 1) *
            pFq((1.0 + 0.0im, α + 1.0 + 0.0im), (α + 2.0 + 0.0im,), u + 0.0im),
        )
    end

    return I
end
