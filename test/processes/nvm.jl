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
    using Distributions
    using QuadGK
    using SpecialFunctions

    α, C, μ, σ = 0.7, 0.5, 0.8, 1.3
    marks = Normal(μ, σ)
    positive = quadgk(w -> w^α * pdf(marks, w), 0, Inf)[1]
    negative = quadgk(w -> w^α * pdf(marks, -w), 0, Inf)[1]
    C_α = 1 / (gamma(1 - α) * cospi(α / 2))
    p = to_stable(NσMProcess(StableSubordinator(α, C), μ, σ))
    @test p.σ^α ≈ (C / α) * (positive + negative) / C_α
    @test p.β ≈ (positive - negative) / (positive + negative)
    one_sided = to_stable(NσMProcess(StableSubordinator(α, C), -2, 0))
    @test one_sided.β == -1
    @test one_sided.σ^α ≈ (C / α) * 2^α / C_α
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

@testitem "Variance gamma: jump measure and time interval" begin
    using QuadGK
    using Random
    p = NormalVarianceMeanProcess(GammaProcess(6, 3), -0.7, 1.2)
    @test_throws ArgumentError NormalVarianceMeanProcess(p.subordinator, 0, -1)
    @test_throws ArgumentError NσMProcess(p.subordinator, Inf, 1)
    @test levy_tail_mass(p, 0.2) ≈ quadgk(x -> levy_density(p, x) + levy_density(p, -x), 0.2, Inf)[1]
    @test levy_drift(p) ≈ quadgk(x -> x * (levy_density(p, x) - levy_density(p, -x)), 0, 1)[1]
    # The gamma decomposition must use the original time interval on both sides.
    path = sample(MersenneTwister(12), TruncatedLevyProcess(p; l=1e-4), 0.3)
    @test !isempty(path.jump_times)
    @test all(t -> 0 <= t <= 0.3, path.jump_times)
    q = NormalVarianceMeanProcess(GammaProcess(2, 3), -2, 0)
    @test levy_density(q, 0.5) == 0
    @test levy_density(q, -0.5) ≈ levy_density(GammaProcess(2, 1.5), 0.5)
    @test_throws ArgumentError marginal(q, 1)
    z = NormalVarianceMeanProcess(GammaProcess(2, 3), 0, 0)
    @test levy_tail_mass(z, 0) == levy_drift(z) == 0
    @test isempty(sample(MersenneTwister(12), TruncatedLevyProcess(z; l=0.1), 1).jump_sizes)
end
