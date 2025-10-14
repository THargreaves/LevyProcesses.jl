using Test
using LevyProcesses

@testitem "NVM: Subordinated sampling" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    test_t = 0.8

    γ = 1.3
    λ = 10.8
    μ_W = 1.4
    σ_W = 1.5

    test_subordinator = GammaProcess(γ, λ)
    test_truncated_subordinator = TruncatedLevyProcess(test_subordinator; l=1e-10)
    test_process = NormalVarianceMeanProcess(test_truncated_subordinator, μ_W, σ_W)

    test_true_process = NormalVarianceMeanProcess(test_subordinator, μ_W, σ_W)
    test_marginal = marginal(test_true_process, test_t)

    REPS = 1000
    rng = MersenneTwister(1234)

    # Generate samples
    marginal_samples = [
        sum(sample(rng, test_process, test_t).jump_sizes)
        for _ in 1:REPS
    ]

    # Compare with ground truth
    test = ExactOneSampleKSTest(marginal_samples, test_marginal)
    @test pvalue(test) > 0.1
end

@testitem "NVM: Direct sampling" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    test_t = 0.8
    test_ϵ = 1e-10

    γ = 1.3
    λ = 10.8
    μ_W = 1.4
    σ_W = 1.5

    test_subordinator = GammaProcess(γ, λ)
    test_true_process = NormalVarianceMeanProcess(test_subordinator, μ_W, σ_W)
    test_marginal = marginal(test_true_process, test_t)

    test_truncated = TruncatedLevyProcess(test_true_process; l=test_ϵ)

    REPS = 1000
    rng = MersenneTwister(1234)

    # Generate samples
    marginal_samples = [
        sum(sample(rng, test_truncated, test_t).jump_sizes)
        for _ in 1:REPS
    ]

    # Compare with ground truth
    test = ExactOneSampleKSTest(marginal_samples, test_marginal)
    @test pvalue(test) > 0.1
end

@testitem "NVM: Direct == subordinated" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    test_t = 0.8
    test_ϵ = 1e-10

    γ = 1.3
    λ = 10.8
    μ_W = 1.4
    σ_W = 1.5

    test_subordinator = GammaProcess(γ, λ)
    test_truncated_subordinator = TruncatedLevyProcess(test_subordinator; l=1e-10)
    test_process = NormalVarianceMeanProcess(test_truncated_subordinator, μ_W, σ_W)

    test_true_process = NormalVarianceMeanProcess(test_subordinator, μ_W, σ_W)
    test_truncated = TruncatedLevyProcess(test_true_process; l=test_ϵ)

    REPS = 1000
    rng = MersenneTwister(1234)

    # Generate samples
    direct_samples = [
        sum(sample(rng, test_truncated, test_t).jump_sizes)
        for _ in 1:REPS
    ]
    subordinated_samples = [
        sum(sample(rng, test_process, test_t).jump_sizes)
        for _ in 1:REPS
    ]

    # Compare with ground truth
    test = ApproximateTwoSampleKSTest(direct_samples, subordinated_samples)
    @test pvalue(test) > 0.1
end
