@testitem "Conditional transition statistics" begin
    using LevyProcesses
    using LinearAlgebra
    using QuadGK
    times, sizes = [0.2, 0.7], [0.4, 0.9]
    path = SampleJumps(times, sizes)
    dyn = LangevinDynamics(-0.5)
    h, x0, t = [0.3, 0.5], [0.2, 0.5], 0.8
    for mixture in (NormalVarianceMeanProcess, NσMProcess)
        p = mixture === NσMProcess ?
            mixture(GammaProcess(1.3, 10.8), 1.4, 1.5; drift=0.35) :
            mixture(GammaProcess(1.3, 10.8), 1.4, 1.5)
        sde = LevyDrivenLinearSDE(p, dyn, h)
        m, Q = conditional_marginal_parameters(path, sde, t; x0)
        # Compose independent Gaussian jump updates chronologically.
        expected_m, expected_Q, previous = copy(x0), zeros(2, 2), 0.0
        for (v, z) in zip(times, sizes)
            F = exp(dyn, v - previous)
            expected_m = F * expected_m + p.μ * z * h
            variance = mixture === NormalVarianceMeanProcess ? z : z^2
            expected_Q = F * expected_Q * F' + p.σ^2 * variance * h * h'
            previous = v
        end
        F = exp(dyn, t - previous)
        drift = LevyProcesses.deterministic_drift(p)
        deterministic_mean = drift * quadgk(s -> exp(dyn, s) * h, 0.0, t)[1]
        @test m ≈ F * expected_m + deterministic_mean
        @test Q ≈ F * expected_Q * F'
    end
end

@testitem "Linear dynamics and stable marginals" begin
    using LevyProcesses
    using Distributions
    using LinearAlgebra
    t = 2.5
    @test exp(LangevinDynamics(0.0), t) == [1.0 t; 0.0 1.0]
    @test exp(LangevinDynamics(-1e-12), t) ≈ exp([0.0 1.0; 0.0 -1e-12] * t)
    for α in (0.7, 1.4)
        p = StableProcess(α, 0.3, 1.2)
        sde = StableDrivenSDE(p, UnivariateLinearDynamics(0.0))
        @test params(marginal(sde, 0.0, t)) == params(marginal(p, t))
        langevin = LangevianStableDrivenSDE(p, LangevinDynamics(0.0))
        @test collect(params(projection_marginal(langevin, t, [0.0, 1.0]))) ≈ collect(params(marginal(p, t)))
        projection = projection_marginal(langevin, 1.0, [1.0, -0.5])
        integral = 2 * 0.5^(α + 1) / (α + 1) / norm([1.0, -0.5])^α
        @test projection.β ≈ 0 atol=1e-12
        @test projection.σ ≈ p.σ * integral^(1 / α)
    end
end

@testitem "Stable projection closed forms and S0 location" begin
    using Distributions, LinearAlgebra, QuadGK
    using ForwardDiff
    LP = LevyProcesses
    t = 1.3
    # Ordinary damping, a crossing, tiny nonzero damping, and analytic limits.
    for (θ, v) in ((-2.0, [1.0, 0.2]), (0.4, [1.0, -0.5]),
                   (1e-8, [1.0, -0.5]), (0.0, [1.0, -0.5]),
                   (-0.5, [1.0, 2.0]), (-0.5, [1.0, 2.0 + 1e-12]),
                   (-0.5, [0.0, 1.0]))
        u = v / norm(v)
        g = s -> dot(u, exp(LangevinDynamics(θ), s) * [0.0, 1.0])
        for α in (0.7, 1.4)
            expected = quadgk(s -> [abs(g(s))^α, sign(g(s)) * abs(g(s))^α], 0.0, t; atol=1e-11)[1]
            @test collect(LP._stable_response_moments(θ, t, u, α)) ≈ expected rtol=1e-8 atol=1e-10
        end
    end
    # Integrating the driver's log characteristic function independently checks
    # the S0 location, including the continuous logarithmic α=1 limit.
    dyn, u, ω = LangevinDynamics(0.4), normalize([1.0, -0.5]), 0.3
    g = s -> dot(u, exp(dyn, s) * [0.0, 1.0])
    for α in (0.7, 1 - 1e-6, 1.0, 1 + 1e-6, 1.4)
        p = StableProcess(α, 0.3, 1.2, 0.2)
        d = projection_marginal(LangevianStableDrivenSDE(p, dyn), t, u)
        driver = marginal(p, 1.0)
        expected = exp(quadgk(s -> log(cf(driver, ω * g(s))), 0.0, t; atol=1e-11)[1])
        @test cf(d, ω) ≈ expected rtol=1e-8
        scalar = marginal(StableDrivenSDE(p, UnivariateLinearDynamics(-0.4)), 0.2, t)
        scalar_cf = exp(im * ω * 0.2 * exp(-0.4t) +
            quadgk(s -> log(cf(driver, ω * exp(-0.4s))), 0.0, t; atol=1e-11)[1])
        @test cf(scalar, ω) ≈ scalar_cf rtol=1e-8
    end
    f(α) = real(cf(projection_marginal(
        LangevianStableDrivenSDE(StableProcess(α, 0.3, 1.2, 0.2), dyn), t, u,
    ), ω))
    @test ForwardDiff.derivative(f, 1.0) ≈ (f(1 + 1e-5) - f(1 - 1e-5)) / 2e-5 rtol=1e-6
end

@testitem "Deterministic drift in batched transitions" begin
    using CUDA
    if CUDA.functional()
        LP = LevyProcesses
        times, sizes, t = [0.2, 0.7], [0.4, 0.9], 0.8
        p = NσMProcess(GammaProcess(1.3, 10.8), 1.4, 1.5; drift=0.35)
        sde = LevyDrivenLinearSDE(p, LangevinDynamics(-0.5), [0.3, 0.5])
        expected, _ = conditional_marginal_parameters(SampleJumps(times, sizes), sde, t)
        ragged = LP.RaggedBatchSampleJumps(CuArray(sizes), CuArray(times), CuArray(Int32[2]), 2)
        regular = LP.RegularBatchSampleJumps(CuArray(reshape(sizes, 2, 1)), CuArray(reshape(times, 2, 1)), 1)
        for paths in (ragged, regular)
            mean, _ = conditional_marginal_parameters(paths, sde, t)
            @test vec(Array(mean)) ≈ expected rtol=1e-6
        end
    else
        @test_skip CUDA.functional()
    end
end
