using Test
using LevyProcesses

@testitem "Stable: Lévy tail mass" begin
    using LevyProcesses
    using QuadGK
    using Test

    α = 0.5
    μ_W = 0.5
    σ_W = 1.2
    p = StableProcess(α, μ_W, σ_W)

    ϵ = 1e-8
    test_x = 1.5
    test_process = TruncatedLevyProcess(p; l=ϵ)

    numerical = (
        quadgk(x -> levy_density(p, x), -Inf, -test_x)[1] +
        quadgk(x -> levy_density(p, x), test_x, Inf)[1]
    )
    @test levy_tail_mass(test_process, test_x) ≈ numerical
end

@testitem "Stable: Shot-noise sampling" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    α = 0.5
    μ_W = 0.5
    σ_W = 1.2
    p = StableProcess(α, μ_W, σ_W)

    test_t = 1.5
    ϵ = 1e-8
    test_process = TruncatedLevyProcess(p; l=ϵ)

    REPS = 1000
    rng = MersenneTwister(1234)

    # Generate samples
    marginal_samples = [sum(sample(rng, test_process, test_t).jump_sizes) for _ in 1:REPS]

    # Compare with ground truth
    test = ExactOneSampleKSTest(marginal_samples, marginal(p, test_t))
    @test pvalue(test) > 0.1
end

@testitem "Stable: SDE marginal" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    α = 0.5
    μ_W = 0.5
    σ_W = 1.2
    p = StableProcess(α, μ_W, σ_W)

    test_t = 1.5
    ϵ = 1e-10
    test_process = TruncatedLevyProcess(p; l=ϵ)

    a = -0.5
    rng = MersenneTwister(1234)

    dyn = UnivariateLinearDynamics(a)
    true_sde = StableDrivenSDE(p, dyn)
    truncated_sde = TruncatedStableDrivenSDE(test_process, dyn)

    test_x0 = 0.5

    # Generate samples
    REPS = 1000
    marginal_samples = Vector{Float64}(undef, REPS)
    for i in 1:REPS
        dist = sample_conditional_marginal(rng, truncated_sde, test_x0, test_t)
        marginal_samples[i] = rand(rng, dist)
    end

    # Compare with ground truth
    test = ExactOneSampleKSTest(marginal_samples, marginal(true_sde, test_x0, test_t))
    @test pvalue(test) > 0.1
end
