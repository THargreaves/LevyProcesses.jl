using Test
using LevyProcesses

@testitem "Conditional marginal" begin
    using LevyProcesses
    using Random
    using Statistics
    using Test

    test_t = 0.8
    test_ϵ = 1e-10

    γ = 1.3
    λ = 10.8
    μ_W = 1.4
    σ_W = 1.5
    θ = -0.5
    h = [0.3, 0.5]
    x0 = [0.2, 0.5]

    S = GammaProcess(γ, λ)
    S̄ = TruncatedLevyProcess(S; l=1e-10)
    W = NormalVarianceMeanProcess(S̄, μ_W, σ_W)

    dyn = LangevinDynamics(θ)
    sde = LevyDrivenLinearSDE(W, dyn, h)

    rng = MersenneTwister(1234)
    jumps = sample(rng, S̄, test_t)

    cond_marginal = conditional_marginal(jumps, sde, test_t; x0)

    # Brute force compute distribution
    REPS = 10000
    sort!(jumps)
    final_states = Vector{Vector{Float64}}(undef, REPS)
    for r in 1:REPS
        x = copy(x0)
        for i in 1:length(jumps.jump_times)
            last_jump = i > 1 ? jumps.jump_times[i - 1] : 0.0
            dt = jumps.jump_times[i] - last_jump
            subordinator_increment = (
                μ_W * jumps.jump_sizes[i] + σ_W * sqrt(jumps.jump_sizes[i]) * randn(rng)
            )
            x = exp(dyn, dt) * x + h * subordinator_increment
        end
        dt = test_t - jumps.jump_times[end]
        final_states[r] = exp(dyn, dt) * x
    end

    @test mean(final_states) ≈ cond_marginal.μ rtol = 1e-2
    @test cov(final_states) ≈ cond_marginal.Σ rtol = 1e-2
end
