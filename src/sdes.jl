import Distributions: MvNormal, Normal

export LinearDynamics, LangevinDynamics
export LevyDrivenLinearSDE

export sample_conditional_marginal, conditional_marginal, NVMDrivenSDE

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
    Σ = (Σ + Σ') / 2 + 1e-6 * I

    return MvNormal(μ, Σ)
end



