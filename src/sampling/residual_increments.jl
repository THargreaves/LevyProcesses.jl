struct HittingTimeMethod <: LevySamplingMethod end
const HittingTime = HittingTimeMethod()

export HittingTime

function sample(rng::AbstractRNG, m::TruncatedLevyProcessMarginal, ::HittingTimeMethod)
    m.process.lower == 0.0 ||
        error("HittingTimeMethod only supports processes with lower bound 0.")

    Z = 0.0
    S = 0.0

    # TODO: this only holds for the stable subordinator
    α = m.process.process.α
    t = m.t * m.process.process.C / (α / gamma(1 - α))

    while true
        T, W = sample_hit(rng, m.process.process)
        if S + T > t
            break
        end
        S += T
        Z += m.process.upper + W
    end

    ΔZ = sample_conditional(rng, m.process.process, t - S, m.process.upper)
    return Z + ΔZ
end

### Special case for the stable subordinator

# TODO: ensure this satisies l = 0
function sample_hit(rng::AbstractRNG, p::StableSubordinator)
    R = 0.0
    Y = 0.0
    while true
        U = rand(rng, Uniform(0, π))
        U1 = rand(rng, Uniform(0, 1))
        Y = 1 - U1^(1 / (1 - p.α))
        A_U = (sin(p.α * U)^p.α * sin((1 - p.α) * U)^(1 - p.α) / sin(U))^(1 / (1 - p.α))
        R = rand(rng, Gamma(2 - p.α, 1 / (A_U - p.λ)))  # scale param so flip expression from paper
        V = rand(rng, Uniform(0, 1))
        accept_p = (
            p.α *
            A_U *
            exp(p.ζ * R^(1 - p.α) * Y^p.α) *
            exp(-p.λ * R) *
            (A_U - p.λ)^(p.α - 2) *
            Y^(p.α - 1) *
            (1 - (1 - Y)^p.α) / p.M
        )
        if V <= accept_p
            break
        end
    end
    U2 = rand(rng, Uniform(0, 1))
    T = R^(1 - p.α) * Y^p.α
    W = Y - 1 + ((1 - Y)^(-p.α) - U2 * ((1 - Y)^(-p.α) - 1))^(-1 / p.α)
    return T, W
end

function sample_conditional(rng::AbstractRNG, p::StableSubordinator, t::Real, upper::Real)
    while true
        U1 = rand(rng, Uniform(0, π))
        A_U = (sin(p.α * U1)^p.α * sin((1 - p.α) * U1)^(1 - p.α) / sin(U1))^(1 / (1 - p.α))
        U2 = rand(rng, Uniform(0, 1))
        Z = (-log(U2) / (A_U * t^(1 / (1 - p.α))))^(-p.α / (1 - p.α))
        if Z < upper
            return Z
        end
    end
end
