abstract type LevySamplingMethod end

import CUDA
import CUDA: CuArray
import Distributions: Poisson, Uniform

export Inversion, Rejection, BatchInversion

# Expected methods
function sample(p::LevyProcess, dt::Real)
    error("no default sampling procedure defined for $(typeof(p))")
end

# """Method A of Rosiński, 2001"""
struct InversionMethod <: LevySamplingMethod end
const Inversion = InversionMethod()

# TODO: this is only valid for subordinators
function sample(rng::AbstractRNG, p::TruncatedLevyProcess, dt::Real, ::InversionMethod)
    N = rand(rng, Poisson(dt * p.mass))
    Γs = rand(rng, Uniform(p.upper_tail_mass, p.lower_tail_mass), N)
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_times = rand(rng, Uniform(0, dt), N)
    return SampleJumps(jump_times, jump_sizes)
end

"""Method E of Rosiński, 2001"""
struct RejectionMethod <: LevySamplingMethod end
const Rejection = RejectionMethod()

function sample(
    rng::AbstractRNG,
    p::TruncatedLevyProcess, dt::Real,
    p₀::TruncatedLevyProcess,
    ::RejectionMethod;
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

##################################
#### GPU-ACCELERATED SAMPLING ####
##################################

struct BatchSampleJumps
    jump_sizes::CuArray{Float32,1}
    jump_times::CuArray{Float32,1}
    offsets::CuArray{Int32,1}
    tot_N::Int
end

struct BatchInversionMethod <: LevySamplingMethod end
const BatchInversion = BatchInversionMethod()

function sample(p::TruncatedLevyProcess, dt::Real, N::Integer, ::BatchInversionMethod)
    Ns = CUDA.rand_poisson(UInt32, N; lambda=dt * p.mass)
    offsets = cumsum(Ns)
    tot_N = sum(Ns)
    Γs = CUDA.rand(tot_N) * (p.upper_tail_mass - p.lower_tail_mass) .+ p.lower_tail_mass
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_times = dt * CUDA.rand(tot_N)
    return BatchSampleJumps(jump_sizes, jump_times, offsets, tot_N)
end
