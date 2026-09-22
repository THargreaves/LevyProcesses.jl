@testitem "S0 inference derivatives at alpha one" begin
    using Distributions
    using ForwardDiff
    functions = (
        a -> levy_drift(StableProcess(a, 0.4, 1.3, 0.2)),
        a -> real(cf(marginal(StableProcess(a, 0.4, 1.3, 0.2), 0.7), 0.8)),
        a -> imag(cf(marginal(StableProcess(a, 0.4, 1.3, 0.2), 0.7), 0.8)),
    )
    for f in functions
        reference = (f(1 + 1e-5) - f(1 - 1e-5)) / 2e-5
        @test ForwardDiff.derivative(f, 1.0) ≈ reference rtol=1e-6 atol=1e-8
    end
end

@testitem "S0 Gaussian convolution" begin
    using Distributions
    using QuadGK
    stable = marginal(StableProcess(1, 0, 1.2, 0.3), 1)
    normal = Normal(-0.2, 0.7)
    convolution = StableGaussianConvolution(stable, normal)
    x = 0.8
    reference = quadgk(y -> pdf(Cauchy(0.3, 1.2), x - y) * pdf(normal, y), -Inf, Inf)[1]
    @test pdf(convolution, x) ≈ reference rtol=1e-7
    @test logpdf(convolution, x) ≈ log(reference) rtol=1e-7
end
