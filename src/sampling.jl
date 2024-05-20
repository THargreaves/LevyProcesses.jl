abstract type LevySamplingMethod end

import Distributions: Exponential

export Inversion, Rejection

# # Expected methods
function sample(p::LevyProcess, dt::Real)
    error("no default sampling procedure defined for $(typeof(p))")
end

# """Method A of Rosiński, 2001"""
struct InversionMethod <: LevySamplingMethod end
const Inversion = InversionMethod()

# TODO: rewrite in terms of Poisson(dt * λ_ϵ)
function sample(rng::AbstractRNG, p::TruncatedSubordinator, dt::Real, ::InversionMethod)
    jump_sizes = Vector{Float64}()
    Γ = 0.0
    while true
        Γ += rand(rng, Exponential(1 / dt))
        x = inverse_levy_tail_mass(p.process, Γ)
        if x < p.ϵ
            break
        end
        push!(jump_sizes, x)
    end
    jump_times = dt * rand(rng, length(jump_sizes))
    return SampleJumps(jump_times, jump_sizes)
end

"""Method E of Rosiński, 2001"""
struct RejectionMethod <: LevySamplingMethod end
const Rejection = RejectionMethod()

function sample(
    rng::AbstractRNG,
    p::TruncatedSubordinator, dt::Real,
    p₀::TruncatedSubordinator,
    ::RejectionMethod;
    # TODO: Should this be a separate method?
    levy_density_ratio::Union{Function,Nothing}=nothing
)
    dominating_jumps = sample(rng, p₀, dt)
    xs = dominating_jumps.jump_sizes

    # Reject jumps
    ps = if levy_density_ratio === nothing
        levy_density.(Ref(p), xs) / levy_density.(Ref(p₀), xs)
    else
        levy_density_ratio.(xs)
    end
    keep = rand(rng, length(ps)) .< ps

    return SampleJumps(
        dominating_jumps.jump_times[keep],
        dominating_jumps.jump_sizes[keep]
    )
end
