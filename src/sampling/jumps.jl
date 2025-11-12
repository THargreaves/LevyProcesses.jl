abstract type LevySamplingMethod end

using CUDA: CUDA
import CUDA: CuArray
import Distributions: Poisson, Uniform, Exponential

export Inversion, Rejection, BatchInversion, BatchRejection

# Expected methods
function sample(p::LevyProcess, dt::Real)
    return error("no default sampling procedure defined for $(typeof(p))")
end

# """Method A of Rosiński, 2001"""
struct InversionMethod <: LevySamplingMethod end
const Inversion = InversionMethod()

# TODO: this is only valid for subordinators
function sample(
    rng::AbstractRNG,
    p::TruncatedLevyProcess{T},
    dt::Real,
    ::InversionMethod;
    sort_sizes::Bool=false,
    sort_times::Bool=false,
) where {T}
    sort_sizes && sort_times && error("cannot sort both jump sizes and jump times")

    N = rand(rng, Poisson(dt * p.mass))

    # Sample jump sizes
    Γs = sample_uniforms(rng, N; sorted=sort_sizes)
    Γs .*= p.upper_tail_mass - p.lower_tail_mass
    Γs .+= p.lower_tail_mass
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)

    # Sample jump times
    jump_times = sample_uniforms(rng, N; sorted=sort_times)
    jump_times .*= dt

    return SampleJumps(jump_times, jump_sizes)
end

@inline function sample_uniforms(rng::AbstractRNG, N::Integer; sorted::Bool=false)
    if sorted
        Es = rand(rng, Exponential(1.0), N)
        tot = Es[1]
        for i in 2:N
            tot += Es[i]
            Es[i] = tot
        end
        tot += rand(rng, Exponential(1.0))
        Es ./= tot
        return Es
    else
        return rand(rng, N)
    end
end

"""Method E of Rosiński, 2001"""
struct RejectionMethod <: LevySamplingMethod end
const Rejection = RejectionMethod()

function sample(
    rng::AbstractRNG,
    p::TruncatedLevyProcess,
    dt::Real,
    p₀::TruncatedLevyProcess,
    ::RejectionMethod;
    levy_density_ratio::Union{Function,Nothing}=nothing,
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

    return SampleJumps(dominating_jumps.jump_times[keep], dominating_jumps.jump_sizes[keep])
end

##################################
#### GPU-ACCELERATED SAMPLING ####
##################################

struct BatchInversionMethod <: LevySamplingMethod end
const BatchInversion = BatchInversionMethod()

struct RaggedBatchSampleJumps{T<:Real}
    jump_sizes::CuArray{T,1}
    jump_times::CuArray{T,1}
    offsets::CuArray{Int32,1}
    tot_N::Int
end

function sample(
    p::TruncatedLevyProcess{T}, dt::T, N::Integer, ::BatchInversionMethod
) where {T}
    Ns = CUDA.rand_poisson(UInt32, N; lambda=dt * p.mass)
    offsets = cumsum(Ns)
    tot_N = sum(Ns)
    Γs = CUDA.rand(tot_N) * (p.upper_tail_mass - p.lower_tail_mass) .+ p.lower_tail_mass
    # jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_sizes = inverse_levy_tail_mass(p.process, Γs)
    jump_times = dt * CUDA.rand(tot_N)
    return RaggedBatchSampleJumps(jump_sizes, jump_times, offsets, tot_N)
end

struct BatchRejectionMethod <: LevySamplingMethod end
const BatchRejection = BatchRejectionMethod()

# TODO: add option to remove rejected jumps (and update offsets)
# 1. Sum rejections by offsets
# 2. Cumsum num rejections
# 3. Subtract from offsets
function sample(
    p::TruncatedLevyProcess,
    dt::T,
    N::Integer,
    p₀::TruncatedLevyProcess,
    ::BatchRejectionMethod;
    levy_density_ratio::Union{Function,Nothing}=nothing,
) where {T}
    # TODO: this should be more generic
    dominating_jumps = sample(p₀, dt, N, BatchInversion)
    xs = dominating_jumps.jump_sizes

    # Reject jumps
    ps = if isnothing(levy_density_ratio)
        levy_density.(Ref(p), xs) / levy_density.(Ref(p₀), xs)
    else
        levy_density_ratio.(xs)
    end
    reject = CUDA.rand(length(ps)) .> ps
    dominating_jumps.jump_sizes[reject] .= 0.0

    return RaggedBatchSampleJumps(
        dominating_jumps.jump_sizes,
        dominating_jumps.jump_times,
        dominating_jumps.offsets,
        dominating_jumps.tot_N,
    )
end

struct RegularBatchSampleJumps{T<:Real}
    jump_sizes::CuArray{T,2}
    jump_times::CuArray{T,2}
    N::Int
end

function sample(p::FixedLevyProcess, dt::Real, N::Integer, ::BatchInversionMethod)
    Us = CUDA.rand(p.N, N)
    Es = -log.(Us) ./ dt
    Γs = cumsum(Es; dims=1)
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_times = dt * CUDA.rand(p.N, N)
    return RegularBatchSampleJumps(jump_sizes, jump_times, N)
end
