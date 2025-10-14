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
    rng::AbstractRNG, sde::NVMDrivenSDE, t::Real; x0::Union{Nothing,Vector}=nothing
)
    # TODO: this needs to be generalised for other cases
    subordinator_path = sample(rng, sde.driving_process.subordinator, t)
    return conditional_marginal(subordinator_path, sde, t; x0)
end

function conditional_marginal(
    subordinator_path::SampleJumps,
    sde::NVMDrivenSDE,
    t::Real;
    x0::Union{Nothing,Vector}=nothing,
)
    m, S = conditional_marginal_parameters(subordinator_path, sde, t; x0)
    # HACK: Force PSD
    S = (S + S') / 2 + 1e-4 * I
    return MvNormal(m, S)
end

function conditional_marginal_parameters(
    subordinator_path::SampleJumps{T},
    sde::NVMDrivenSDE,
    t::Real;
    x0::Union{Nothing,Vector{T}}=nothing,
) where {T}
    m, S = unscaled_conditional_marginal_parameters(subordinator_path, sde, t)
    m *= sde.driving_process.μ
    S *= sde.driving_process.σ^2
    isnothing(x0) || (m += exp(sde.linear_dynamics, t) * x0)  # not scaled by μ_W
    return m, S
end

function unscaled_conditional_marginal_parameters(
    subordinator_path::SampleJumps{T},
    sde::NVMDrivenSDE,
    t::Real;
    x0::Union{Nothing,Vector{T}}=nothing,
) where {T}
    dyn = sde.linear_dynamics
    D = length(sde.noise_scaling)

    m = zeros(T, D)
    S = zeros(T, D, D)
    for (v, z) in zip(subordinator_path.jump_times, subordinator_path.jump_sizes)
        ft = exp(dyn, (t - v)) * sde.noise_scaling
        m += ft * z
        S += ft * ft' * z
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
    subordinator_paths::RaggedBatchSampleJumps{T}, sde::NVMDrivenSDE, t::Real
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
    jump_times::CuVector, jump_sizes::CuVector, sde::NVMDrivenSDE, t::Real
)
    dyn = sde.linear_dynamics
    μ_W, σ_W = sde.driving_process.μ, sde.driving_process.σ
    tot_N = length(jump_times)

    expAs = compute_expAs(dyn, t .- jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    μs = μ_W * expA_h .* jump_sizes'
    Σs = (
        σ_W^2 *
        NNlib.batched_mul(reshape(expA_h, 2, 1, tot_N), reshape(expA_h, 1, 2, tot_N)) .*
        reshape(jump_sizes, 1, 1, tot_N)
    )
    return μs, Σs
end

function conditional_marginal_parameters(
    subordinator_paths::RegularBatchSampleJumps, sde::NVMDrivenSDE, t::Real
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
    subordinator_paths::RaggedBatchSampleJumps{T}, sde::NVMDrivenSDE, t::Real
) where {T}
    dyn = sde.linear_dynamics

    expAs = compute_expAs(dyn, t .- subordinator_paths.jump_times)
    expA_h = NNlib.batched_vec(expAs, cu(sde.noise_scaling))
    ms = expA_h .* subordinator_paths.jump_sizes'
    Ss = (
        NNlib.batched_mul(
            reshape(expA_h, 2, 1, subordinator_paths.tot_N),
            reshape(expA_h, 1, 2, subordinator_paths.tot_N),
        ) .* reshape(subordinator_paths.jump_sizes, 1, 1, subordinator_paths.tot_N)
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
    subordinator_paths::RegularBatchSampleJumps{T}, sde::NVMDrivenSDE, t::Real
) where {T}
    dyn = sde.linear_dynamics

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
    Ss = (
        NNlib.batched_mul(reshape(expA_h, 2, 1, tot_N), reshape(expA_h, 1, 2, tot_N)) .*
        reshape(jump_sizes_flat, 1, 1, tot_N)
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

struct TruncatedStableDrivenSDE
    driving_process::TruncatedStableProcess
    linear_dynamics::UnivariateLinearDynamics
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
