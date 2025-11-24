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
    marginal_samples = [sum(sample(rng, test_process, test_t).jump_sizes) for _ in 1:REPS]

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
    marginal_samples = [sum(sample(rng, test_truncated, test_t).jump_sizes) for _ in 1:REPS]

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
    direct_samples = [sum(sample(rng, test_truncated, test_t).jump_sizes) for _ in 1:REPS]
    subordinated_samples = [
        sum(sample(rng, test_process, test_t).jump_sizes) for _ in 1:REPS
    ]

    # Compare with ground truth
    test = ApproximateTwoSampleKSTest(direct_samples, subordinated_samples)
    @test pvalue(test) > 0.1
end

@testitem "Stable-NσM to Stable conversion" begin
    using LevyProcesses
    using Random
    using Test
    using HypothesisTests
    using StableDistributions

    α = 0.7
    C = 0.5
    μ = 0.8
    σ = 1.3
    t = 1.2
    ϵ = 1e-7

    REPS = 1000
    rng = MersenneTwister(1234)

    S = StableSubordinator(α, C)
    L = NσMProcess(S, μ, σ)

    S̄ = TruncatedLevyProcess(S; l=ϵ)
    L̄ = NσMProcess(S̄, μ, σ)

    stable_process = to_stable(L)

    # Generate samples from NσM process
    samples = [sum(sample(rng, L̄, t).jump_sizes) for _ in 1:REPS]

    # Compare with Stable process marginal distribution
    test = ExactOneSampleKSTest(samples, marginal(stable_process, t))
    @test pvalue(test) > 0.1
end

@testitem "Stable to Stable-NσM conversion" begin
    using LevyProcesses

    # We just check that a full conversion cycle returns the original process and then rely
    # on the validatiy of the Stable-NσM to Stable conversion test above.
    α = 0.7
    C = 0.5
    μ_true = 0.8
    σ_true = 1.3

    S = StableSubordinator(α, C)
    L_nsm = NσMProcess(S, μ_true, σ_true)

    # Convert to Stable process
    stable_process = to_stable(L_nsm)

    # Convert back to NσM process
    L_nsm_converted = to_nsm(stable_process; C=C)

    @test L_nsm.μ ≈ L_nsm_converted.μ
    @test L_nsm.σ ≈ L_nsm_converted.σ
end
