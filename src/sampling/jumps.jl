abstract type LevySamplingMethod end

using CUDA: CUDA
import CUDA: CuArray
import Distributions: Poisson, Uniform, Exponential
import Random: randexp

export Inversion, Rejection, BatchInversion, BatchRejection

# Expected methods
sample(p::LevyProcess, dt::Real) = sample(default_rng(), p, dt)

function sample(rng::AbstractRNG, p::LevyProcess, dt::Real)
    return error("no default sampling procedure defined for $(typeof(p))")
end

# """Method A of Rosiński, 2001"""
struct InversionMethod <: LevySamplingMethod end
const Inversion = InversionMethod()

function _retained_jump_intensity(p::TruncatedLevyProcess, dt::Real)
    isfinite(dt) && dt >= 0 || throw(ArgumentError("dt must be finite and non-negative"))
    isfinite(p.mass) && p.mass >= 0 || throw(ArgumentError(
        "jump sampling requires finite retained Lévy mass; use a positive lower cutoff",
    ))
    intensity = dt * p.mass
    isfinite(intensity) || throw(ArgumentError("retained jump intensity must be finite"))
    return intensity
end

_check_inversion_process(::LevyProcess) = nothing
function _check_inversion_process(::Union{StableProcess,AbstractNormalMixtureProcess})
    throw(ArgumentError("generic tail inversion does not support signed jumps; use the process-specific sampler"))
end
function sample(
    rng::AbstractRNG,
    p::TruncatedLevyProcess{T},
    dt::Real,
    ::InversionMethod;
    sort_sizes::Bool=false,
    sort_times::Bool=false,
) where {T}
    sort_sizes && sort_times && error("cannot sort both jump sizes and jump times")

    _check_inversion_process(p.process)
    N = rand(rng, Poisson(_retained_jump_intensity(p, dt)))

    # Sample jump sizes
    Γs = sample_uniforms(rng, T, N; sorted=sort_sizes)
    Γs .*= p.upper_tail_mass - p.lower_tail_mass
    Γs .+= p.lower_tail_mass
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)

    # Sample jump times
    jump_times = sample_uniforms(rng, T, N; sorted=sort_times)
    jump_times .*= dt

    return SampleJumps(jump_times, jump_sizes)
end

sample_uniforms(rng::AbstractRNG, N::Integer; sorted::Bool=false) =
    sample_uniforms(rng, Float64, N; sorted=sorted)

@inline function sample_uniforms(
    rng::AbstractRNG, ::Type{T}, N::Integer; sorted::Bool=false
) where {T<:AbstractFloat}
    N >= 0 || throw(ArgumentError("sample count must be non-negative"))
    N == 0 && return T[]
    if sorted
        Es = randexp(rng, T, N)
        tot = Es[1]
        for i in 2:N
            tot += Es[i]
            Es[i] = tot
        end
        tot += randexp(rng, T)
        Es ./= tot
        return Es
    else
        return rand(rng, T, N)
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
    _retained_jump_intensity(p, dt)
    dominating_jumps = sample(rng, p₀, dt)
    xs = dominating_jumps.jump_sizes

    # Reject jumps
    ps = if levy_density_ratio === nothing
        levy_density.(Ref(p), xs) ./ levy_density.(Ref(p₀), xs)
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
    p::TruncatedLevyProcess{T}, dt::Real, N::Integer, ::BatchInversionMethod
) where {T}
    N >= 0 || throw(ArgumentError("batch size must be non-negative"))
    _check_inversion_process(p.process)
    intensity = _retained_jump_intensity(p, dt)
    intensity <= typemax(UInt32) || throw(ArgumentError("jump intensity exceeds GPU count capacity"))
    Ns = N == 0 || iszero(intensity) ? CUDA.zeros(Int64, N) :
        Int64.(CUDA.rand_poisson(UInt32, N; lambda=intensity))
    tot_N = Int(sum(Ns))
    tot_N <= typemax(Int32) || throw(ArgumentError("batch exceeds GPU offset capacity"))
    offsets = Int32.(cumsum(Ns))
    Γs = CUDA.rand(T, tot_N) .* p.mass .+ p.upper_tail_mass
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_times = T(dt) .* CUDA.rand(T, tot_N)
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
    _retained_jump_intensity(p, dt)
    dominating_jumps = sample(p₀, dt, N, BatchInversion)
    xs = dominating_jumps.jump_sizes

    # Reject jumps
    ps = if isnothing(levy_density_ratio)
        levy_density.(Ref(p), xs) ./ levy_density.(Ref(p₀), xs)
    else
        levy_density_ratio.(xs)
    end
    reject = CUDA.rand(eltype(xs), length(ps)) .>= ps
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

function sample(p::FixedLevyProcess{T}, dt::Real, N::Integer, ::BatchInversionMethod) where {T}
    isfinite(dt) && dt > 0 || throw(ArgumentError("fixed-count sampling requires finite positive dt"))
    N >= 0 && p.N >= 0 || throw(ArgumentError("jump and batch counts must be non-negative"))
    _check_inversion_process(p.process)
    Us = CUDA.rand(T, p.N, N)
    Es = -log.(Us) ./ T(dt)
    Γs = cumsum(Es; dims=1)
    jump_sizes = inverse_levy_tail_mass.(Ref(p.process), Γs)
    jump_times = T(dt) .* CUDA.rand(T, p.N, N)
    return RegularBatchSampleJumps(jump_sizes, jump_times, N)
end
