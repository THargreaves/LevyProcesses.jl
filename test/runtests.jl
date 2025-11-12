using TestItems

@run_package_tests

@testitem "Sorted uniforms" begin
    using LevyProcesses
    using Random
    using Distributions
    using HypothesisTests

    rng = MersenneTwister(1234)
    N = 10000

    us_sorted = LevyProcesses.sample_uniforms(rng, N; sorted=true)
    @test issorted(us_sorted)
    @test all(0 .<= us_sorted .<= 1)
    test = ExactOneSampleKSTest(us_sorted, Uniform(0, 1))
    @test pvalue(test) > 0.1
end
