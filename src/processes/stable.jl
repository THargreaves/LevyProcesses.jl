import Distributions: Exponential, Normal
import QuadGK: quadgk

export StableProcess

# TODO: add alternative constructor
struct StableProcess <: ConditionallyGaussianLevyProcess
    α::Float64
    μ_W::Float64
    σ_W::Float64
    # Canonical parameterisation
    β::Float64
    σ::Float64
    # Cached values
    C_α::Float64
end
function StableProcess(α::Float64, μ_W::Float64, σ_W::Float64)
    α_moment = quadgk(x -> pdf(Normal(μ_W, σ_W), x) * abs(x)^α, -Inf, Inf)[1]
    α_sgn_moment = quadgk(x -> pdf(Normal(μ_W, σ_W), x) * abs(x)^α * sign(x), -Inf, Inf)[1]
    C_α = (1 - α) / (gamma(2 - α) * cos(π * α / 2))

    σ = (α_moment / C_α)^(1 / α)
    β = α_sgn_moment / α_moment

    return StableProcess(α, μ_W, σ_W, β, σ, C_α)
end

function levy_density(p::StableProcess, x::Real)
    return (
        p.σ^p.α * p.C_α * (1 + p.β * sign(x)) * p.α * abs(x)^(-p.α - 1)
    )
end

function levy_tail_mass(p::StableProcess, x::Real)
    return p.σ^p.α * p.C_α * 2 * x^(-p.α)
end

function marginal(p::StableProcess, t::Real)
    return Stable(p.α, p.β, p.σ * t^(1 / p.α), 0.0)
end

const TruncatedStableProcess = TruncatedLevyProcess{StableProcess}

function sample_shot_noise(rng::AbstractRNG, p::TruncatedStableProcess, dt::Real)
    jump_sizes = Float64[]
    Γ = 0.0
    while true
        Γ += rand(rng, Exponential(1 / dt))
        Γα = Γ^(-1 / p.process.α)
        if Γα < p.lower
            break
        end
        push!(jump_sizes, Γα)
    end
    jump_times = rand(Uniform(0, dt), length(jump_sizes))
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
    show_noise_path = sample_shot_noise(rng, p, dt)

    jump_means = show_noise_path.jump_sizes .* p.process.μ_W
    jump_variances = (show_noise_path.jump_sizes .* process.σ_W) .^ 2

    return MarginalisedSampleJumps(shot_noise_path.jump_times, jump_means, jump_variances)
end
