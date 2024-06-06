let
    test_t = 0.8
    test_ϵ = 1e-10

    test_subordinator = GammaProcess(1.0, 1.0)
    test_truncated_subordinator = TruncatedLevyProcess(test_subordinator; l=1e-10)
    test_process = NormalVarianceMeanProcess(test_truncated_subordinator, 0.1, 0.2)

    test_true_process = NormalVarianceMeanProcess(test_subordinator, 1.0, 1.0)
    test_marginal = marginal(test_true_process, test_t)

    @testset "Subordinated sampling" begin
        REPS = 100000
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

    test_truncated = TruncatedLevyProcess(test_true_process; l=test_ϵ)

    @testset "Direct sampling" begin
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
end
