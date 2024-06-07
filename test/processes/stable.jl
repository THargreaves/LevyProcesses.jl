let
    test_process = StableSubordinator(0.5, 0.1)
    test_t = 0.8
    test_ϵ = 1e-10
    test_x = 1.2

    @testset "Lévy tail mass" begin
        @test levy_tail_mass(test_process, test_x) ≈ quadgk(Base.Fix1(levy_density, test_process), test_x, Inf)[1]
    end

    @testset "Inverse Lévy tail mass" begin
        Γ = levy_tail_mass(test_process, test_x)
        @test inverse_levy_tail_mass(test_process, Γ) ≈ test_x
    end

    @testset "Inverse Lévy measure sampling" begin
        p = TruncatedLevyProcess(test_process; l=test_ϵ)
        REPS = 1000
        rng = MersenneTwister(1234)

        # Generate samples
        marginal_samples = [
            sum(sample(rng, p, test_t, Inversion).jump_sizes)
            for _ in 1:REPS
        ]

        # Compare with ground truth
        test = ExactOneSampleKSTest(marginal_samples, marginal(p.process, test_t))
        @test pvalue(test) > 0.1
    end

    @testset "Small jump increments" begin
        α = 0.5
        # TODO: generalise to other values of C
        C = α / gamma(1 - α)
        t = 2.0

        p = TruncatedLevyProcess(StableSubordinator(α, C); u=1.0)
        rng = MersenneTwister(1234)

        # Simulate increment via jumps
        n_approx = 1000
        p_approx = TruncatedLevyProcess(StableSubordinator(α, C); u=1.0, l=1e-9)
        approx_samples = [
            sum(sample(rng, p_approx, t, Inversion).jump_sizes)
            for _ in 1:n_approx
        ]

        # Simulate increment exactly
        n_exact = 1000
        m = marginal(p, t)
        exact_samples = [
            sample(rng, m, HittingTime)
            for _ in 1:n_exact
        ]

        # Compare distributions
        test = ApproximateTwoSampleKSTest(approx_samples, exact_samples)
        @test pvalue(test) > 0.1
    end
end
