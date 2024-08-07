let
    test_process = GammaProcess(0.9, 1.1)
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

    # Tests for the dominating process used in rejection sampling
    @testset "Dominating process" begin
        dominating_process = LevyProcesses.GammaDominatingProcess(0.9, 1.1)
        @test levy_tail_mass(dominating_process, test_x) ≈ quadgk(Base.Fix1(levy_density, dominating_process), test_x, Inf)[1]
        @test inverse_levy_tail_mass(dominating_process, levy_tail_mass(dominating_process, test_x)) ≈ test_x
    end

    @testset "Rejection sampling" begin
        p = TruncatedLevyProcess(test_process; l=test_ϵ)
        REPS = 1000
        rng = MersenneTwister(1234)

        # Generate samples
        marginal_samples = Vector{Float64}(undef, REPS)
        for i in 1:REPS
            sample_jumps = sample(rng, p, test_t)
            marginal_samples[i] = sum(sample_jumps.jump_sizes)
        end

        # Compare with ground truth
        test = ExactOneSampleKSTest(marginal_samples, marginal(p.process, test_t))
        @test pvalue(test) > 0.1
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

end
