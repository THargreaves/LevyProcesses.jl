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

@testitem "Stable: Gaussian-mark conversion" begin
    p = StableProcess(0.6, 0.3, 1.2)
    recovered = to_stable(to_nsm(p; C=0.4))
    @test [recovered.α, recovered.β, recovered.σ] ≈ [p.α, p.β, p.σ]
end

@testitem "Stable-Gaussian convolution" begin
    using StableDistributions
    using Distributions
    using QuadGK
    stable = Stable(0.8, 0.3, 1.0, 0.5)
    normal = Normal(0.2, 0.8)
    convolution = StableGaussianConvolution(stable, normal)
    for x in (0.0, 0.7, 1.5)
        reference = quadgk(u -> real(cis(-u * x) * cf(stable, u) * cf(normal, u)),
                          0, Inf; rtol=1e-9)[1] / π
        @test pdf(convolution, x) ≈ reference rtol=0.005
        @test pdf(convolution, x; M=20) ≈ reference rtol=2e-8
    end
end

@testitem "Stable: canonical drift and normalisation" begin
    using QuadGK
    for α in (0.5, 1.5)
        p = from_s1(α, 1, 0.8)
        integrand(x) = (expm1(-x) + (x <= 1 ? x : 0)) * levy_density(p, x)
        exponent = -levy_drift(p) + quadgk(integrand, 0, 1, Inf; rtol=1e-6)[1]
        @test exponent ≈ -p.σ^α / cospi(α / 2) rtol=1e-5
    end
    @test_throws ArgumentError StableProcess(2, 0, 1)
    @test_throws ArgumentError StableProcess(0.5, 2, 1)
    @test_throws ArgumentError StableProcess(0.5, 0, 0)
    @test_throws ArgumentError to_nsm(StableProcess(1.5, 0, 1))
end

@testitem "Stable: physical jump sampling" begin
    using Random
    p = StableProcess(0.7, -0.4, 1.2)
    truncated = TruncatedLevyProcess(p; l=0.2, u=2.0)
    dt = 2000.0
    path = sample(MersenneTwister(42), truncated, dt)
    expected = dt * truncated.mass
    @test abs(length(path) - expected) < 6sqrt(expected)
    @test all(0.2 .<= abs.(path.jump_sizes) .<= 2.0)
    @test all(0 .<= path.jump_times .<= dt)
    npositive = count(>(0), path.jump_sizes)
    @test abs(npositive - 0.3length(path)) < 6sqrt(0.21length(path))
    @test isempty(sample(MersenneTwister(42), truncated, 0).jump_sizes)
    @test_throws ArgumentError sample_marginalised(MersenneTwister(42), truncated, 1)
end

@testitem "Stable: S0 continuity and increments" begin
    using Distributions
    base = StableProcess(1, 0.7, 1.2, 0.3)
    for α in (1 - 1e-9, 1.0, 1 + 1e-9)
        p = StableProcess(α, base.β, base.σ, base.μ)
        @test cf(marginal(p, 1), 0.6) ≈ cf(marginal(base, 1), 0.6) atol=1e-8
        @test levy_drift(p) ≈ levy_drift(base) atol=1e-8
        @test cf(marginal(p, 0.4), 0.6) * cf(marginal(p, 0.7), 0.6) ≈
              cf(marginal(p, 1.1), 0.6)
    end
    p = from_s1(0.7, 0.4, 1.2, -0.3)
    @test collect(params(to_s1(p))) ≈ [0.7, 0.4, 1.2, -0.3]
    @test_throws ArgumentError StableS0(1, 2, 1, 0)
end

@testitem "Stable: S0 marginal evaluation and sampling" begin
    using Distributions
    using Random
    using Statistics
    cauchy = marginal(StableProcess(1, 0, 1.2, 0.3), 1)
    @test pdf(cauchy, 0.2) ≈ pdf(Cauchy(0.3, 1.2), 0.2)
    @test cdf(cauchy, 0.2) ≈ cdf(Cauchy(0.3, 1.2), 0.2)
    rng = MersenneTwister(55)
    for α in (0.7, 1.0, 1.3)
        d = marginal(StableProcess(α, 0.7, 1.2, 0.3), 1)
        @test abs(mean(cis.(0.6 .* rand(rng, d, 4000))) - cf(d, 0.6)) < 0.06
    end
end

@testitem "Stable: alpha-one canonical exponent" begin
    using QuadGK
    p = StableProcess(1, 1, 0.8, 0.3)
    compensated(x) = x < 1e-3 ? x^2 * evalpoly(x, (1/2, -1/6, 1/24, -1/120)) : expm1(-x) + x
    integrand(x) = (x <= 1 ? compensated(x) : expm1(-x)) * levy_density(p, x)
    exponent = -levy_drift(p) + quadgk(integrand, 0, 1, Inf)[1]
    @test exponent ≈ 2p.σ / π * log(p.σ) - p.μ
end
