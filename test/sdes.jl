@testitem "Conditional transition statistics" begin
    using LevyProcesses
    using LinearAlgebra
    times, sizes = [0.2, 0.7], [0.4, 0.9]
    path = SampleJumps(times, sizes)
    dyn = LangevinDynamics(-0.5)
    h, x0, t = [0.3, 0.5], [0.2, 0.5], 0.8
    for mixture in (NormalVarianceMeanProcess, NσMProcess)
        p = mixture(GammaProcess(1.3, 10.8), 1.4, 1.5)
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
        @test m ≈ F * expected_m
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
