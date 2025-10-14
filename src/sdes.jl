import Distributions: MvNormal, Normal
using CUDA
using NNlib

export LinearDynamics, LangevinDynamics
export LevyDrivenLinearSDE

export sample_conditional_marginal, conditional_marginal, NVMDrivenSDE

export UnivariateLinearDynamics, StableDrivenSDE, TruncatedStableDrivenSDE

using LinearAlgebra

abstract type LinearDynamics end

struct LangevinDynamics{T<:Real} <: LinearDynamics
    θ::T
end
function Base.exp(dyn::LangevinDynamics, dt::Real)
    θ = dyn.θ
    exp_val = exp(θ * dt)
    return StaticArrays.@SMatrix [1.0 (exp_val-1)/θ; 0 exp_val]
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

#########################
#### NVM-Driven SDEs ####
#########################

# TODO: generalise this to arbitrary conditionally Gaussian Levy processes
const NVMDrivenSDE = LevyDrivenLinearSDE{P,D,T} where {P<:NormalVarianceMeanProcess,D,T}

function sample_conditional_marginal(
    rng::AbstractRNG,
    sde::NVMDrivenSDE,
    x0::Vector,
    t::Real
)
    # TODO: this needs to be generalised for other cases
    subordinator_path = sample(rng, sde.driving_process.subordinator, t)
    return conditional_marginal(subordinator_path, sde, x0, t)
end

function conditional_marginal(
    subordinator_path::SampleJumps,
    sde::NVMDrivenSDE,
    x0::Vector,
    t::Real
)
    dyn = sde.linear_dynamics
    μ_W, σ_W = sde.driving_process.μ, sde.driving_process.σ

    μ = exp(dyn, t) * x0
    Σ = zeros(length(x0), length(x0))
    for (v, z) in zip(subordinator_path.jump_times, subordinator_path.jump_sizes)
        ft = exp(dyn, (t - v)) * sde.noise_scaling
        μ += ft * μ_W * z
        # TODO: think this should be just z for the NVM case (z^2 for NσM, e.g. stable)
        Σ += ft * ft' * σ_W^2 * z^2
    end
    # HACK: Force to be PD
    Σ = (Σ + Σ') / 2 + 1e-5 * I

    return MvNormal(μ, Σ)
end

function sum_blocks!(μs, Σs, μ, Σ, offsets, num_runs_ref)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    K = num_runs_ref[]
    for i = index:stride:K
        start = i == 1 ? 1 : offsets[i-1] + 1
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
    subordinator_paths::RaggedBatchSampleJumps,
    sde::NVMDrivenSDE,
    t::Real
)
    dyn = sde.linear_dynamics
    μ_W, σ_W = sde.driving_process.μ, sde.driving_process.σ

    expAs = compute_expAs(dyn, t .- subordinator_paths.jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    μs = μ_W * expA_h .* subordinator_paths.jump_sizes'
    Σs = (
        σ_W^2 *
        NNlib.batched_mul(
            reshape(expA_h, 2, 1, subordinator_paths.tot_N),
            reshape(expA_h, 1, 2, subordinator_paths.tot_N)
        ) .*
        reshape((subordinator_paths.jump_sizes .^ 2), 1, 1, subordinator_paths.tot_N)
    )

    N = length(subordinator_paths.offsets)
    num_runs_ref = CuArray([N])
    μ = CUDA.zeros(2, N)
    Σ = CUDA.zeros(2, 2, N)
    num_runs_ref = convert.(Int32, num_runs_ref)
    CUDA.@cuda threads = 256 blocks = 4096 sum_blocks!(
        μs, Σs, μ, Σ, subordinator_paths.offsets, num_runs_ref
    )

    return μ, Σ
end

function unscaled_conditional_marginal_parameters(
    subordinator_paths::RaggedBatchSampleJumps,
    sde::NVMDrivenSDE,
    t::Real
)
    dyn = sde.linear_dynamics

    expAs = compute_expAs(dyn, t .- subordinator_paths.jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    ms = expA_h .* subordinator_paths.jump_sizes'
    Ss = (
        NNlib.batched_mul(
            reshape(expA_h, 2, 1, subordinator_paths.tot_N),
            reshape(expA_h, 1, 2, subordinator_paths.tot_N)
        ) .*
        reshape((subordinator_paths.jump_sizes .^ 2), 1, 1, subordinator_paths.tot_N)
    )

    N = length(subordinator_paths.offsets)
    num_runs_ref = CuArray([N])
    m = CUDA.zeros(2, N)
    S = CUDA.zeros(2, 2, N)
    num_runs_ref = convert.(Int32, num_runs_ref)
    CUDA.@cuda threads = 256 blocks = 4096 sum_blocks!(
        ms, Ss, m, S, subordinator_paths.offsets, num_runs_ref
    )

    return m, S
end

############################
#### Stable-Driven SDEs ####
############################

struct UnivariateLinearDynamics{T<:Real} <: LinearDynamics
    a::T
end

struct TruncatedStableDrivenSDE
    driving_process::TruncatedStableProcess
    linear_dynamics::UnivariateLinearDynamics
end

struct StableDrivenSDE
    driving_process::StableProcess
    linear_dynamics::UnivariateLinearDynamics
end

function sample_conditional_marginal(
    rng::AbstractRNG,
    sde::TruncatedStableDrivenSDE,
    x0::Float64,
    t::Real
)
    shot_noise_path = sample_shot_noise(rng, sde.driving_process, t)
    return conditional_marginal(shot_noise_path, sde, x0, t)
end

function conditional_marginal(
    shot_noise_path::SampleJumps,
    sde::TruncatedStableDrivenSDE,
    x0::Float64,
    t::Real
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

function marginal(
    sde::StableDrivenSDE,
    x0::Float64,
    t::Real
)
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
