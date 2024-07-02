import Distributions: MvNormal, Normal

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
    return [1.0 (exp_val-1)/θ; 0 exp_val]
end

struct LevyDrivenLinearSDE{P<:LevyProcess,D<:LinearDynamics,T<:Real}
    driving_process::P
    linear_dynamics::D
    # TODO: should this be a method to allow LangevinDynamics to be parameterised?
    noise_scaling::Vector{T}
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
        Σ += ft * ft' * σ_W^2 * z^2
    end
    # HACK: Force to be PD
    Σ = (Σ + Σ') / 2 + 1e-5 * I

    return MvNormal(μ, Σ)
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
