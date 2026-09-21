@testitem "Retained-path density" begin
    using Distributions
    using LevyProcesses
    const LP = LevyProcesses
    p = TruncatedLevyProcess(GammaProcess(1.3, 0.8); l=0.2, u=3.0)
    dt = 2.5
    path = SampleJumps([0.4, 1.7], [0.6, 1.2])
    expected = logpdf(Poisson(dt * p.mass), 2) - 2log(dt) +
        sum(x -> log_levy_density(p, x) - log(p.mass), path.jump_sizes)
    actual = LP.log_normalised_sample_jumps_density(p, dt, path)
    @test actual ≈ expected
    @test LP.normalised_sample_jumps_density(p, dt, path) ≈ exp(expected)
    @test LP.log_normalised_sample_jumps_density(p, dt, path; sorted=true) ≈ expected + log(2)
    @test LP.log_normalised_sample_jumps_density(p, dt, SampleJumps(Float64[], Float64[])) == -dt * p.mass
    @test LP.log_normalised_sample_jumps_density(p, dt, SampleJumps([dt + 1], [0.6])) == -Inf
    @test LP.log_normalised_sample_jumps_density(p, dt, SampleJumps([0.4], [0.1])) == -Inf
    @test SampleJumps{Float32}([0.4], [0.6]).jump_sizes == Float32[0.6]
    @test_throws DimensionMismatch SampleJumps([0.4], Float64[])
end

@testitem "Truncation contract" begin
    using LevyProcesses
    p = StableProcess(0.7, 0.2, 1.0)
    truncated = TruncatedLevyProcess(p; l=0.2, u=3.0)
    @test levy_density(truncated, -0.6) == levy_density(p, -0.6)
    @test levy_density(truncated, -0.2) == levy_density(p, -0.2)
    @test LevyProcesses.levy_variance(truncated) == 0
    @test levy_tail_mass(truncated, 0.0) == truncated.mass
    @test levy_tail_mass(truncated, Inf) == 0
    @test_throws ArgumentError TruncatedLevyProcess(p; l=0.2, approximate_residual=true)
end
