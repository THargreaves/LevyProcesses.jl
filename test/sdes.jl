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

@testitem "Langevian-Stable SDE projection marginals" begin
    using LevyProcesses
    using Random
    using Statistics
    using Test
    using HypothesisTests
    using StaticArrays
    using LinearAlgebra

    t = 0.8
    ϵ = 1e-6
    REPS = 1000

    α = 0.7
    β = 0.3
    γ = 1.2
    θ = -0.5
    h = @SVector [0.0, 1.0]

    rng = MersenneTwister(1234)

    S = StableProcess(α, β, γ)
    W = to_nsm(S)
    W̄ = NσMProcess(
        TruncatedLevyProcess(StableSubordinator(α, W.subordinator.C); l=ϵ), W.μ, W.σ
    )
    dyn = LangevinDynamics(θ)
    sde = LangevianStableDrivenSDE(S, dyn)

    marginals = Vector{SVector{2,Float64}}(undef, REPS)
    for r in 1:REPS
        jumps = sample(rng, W̄, t)
        x = @SVector [0.0, 0.0]
        for i in 1:length(jumps.jump_sizes)
            last_jump = i > 1 ? jumps.jump_times[i - 1] : 0.0
            dt = jumps.jump_times[i] - last_jump
            x = exp(dyn, dt) * x + h * jumps.jump_sizes[i]
        end
        dt = t - jumps.jump_times[end]
        x = exp(dyn, dt) * x
        marginals[r] = x
    end

    # Test over range of projection angles
    n_tests = 32
    ϕs = range(0, π; length=n_tests)
    bonferroni_α = 0.05 / n_tests
    p_values = Vector{Float64}(undef, n_tests)
    for (i, ϕ) in enumerate(ϕs)
        proj = @SVector [cos(ϕ), sin(ϕ)]
        projected_samples = [dot(m, proj) for m in marginals]

        analytical_dist = projection_marginal(sde, t, proj)
        test = ExactOneSampleKSTest(projected_samples, analytical_dist)
        p_values[i] = pvalue(test)
    end
    @test all(pv -> pv > bonferroni_α, p_values)
end
