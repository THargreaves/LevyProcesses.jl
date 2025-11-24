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

@testitem "Stable: Shot-noise sampling" begin
    using LevyProcesses
    using HypothesisTests
    using Random
    using Test

    α = 0.5
    β = 0.3
    γ = 1.2
    stable_p = StableProcess(α, β, γ)
    NσM_p = to_nsm(stable_p)

    test_t = 1.5
    ϵ = 1e-8
    test_process = NσMProcess(
        TruncatedLevyProcess(StableSubordinator(α, NσM_p.subordinator.C); l=ϵ),
        NσM_p.μ,
        NσM_p.σ,
    )

    REPS = 1000
    rng = MersenneTwister(1234)

    # Generate samples
    marginal_samples = [sum(sample(rng, test_process, test_t).jump_sizes) for _ in 1:REPS]

    # Compare with ground truth
    test = ExactOneSampleKSTest(marginal_samples, marginal(stable_p, test_t))
    @test pvalue(test) > 0.1
end

# TODO: merge with NσM
# @testitem "Stable: SDE marginal" begin
#     using LevyProcesses
#     using HypothesisTests
#     using Random
#     using Test

#     α = 0.5
#     μ_W = 0.5
#     σ_W = 1.2
#     p = StableProcess(α, μ_W, σ_W)

#     test_t = 1.5
#     ϵ = 1e-10
#     test_process = TruncatedLevyProcess(p; l=ϵ)

#     a = -0.5
#     rng = MersenneTwister(1234)

#     dyn = UnivariateLinearDynamics(a)
#     true_sde = StableDrivenSDE(p, dyn)
#     truncated_sde = TruncatedStableDrivenSDE(test_process, dyn)

#     test_x0 = 0.5

#     # Generate samples
#     REPS = 1000
#     marginal_samples = Vector{Float64}(undef, REPS)
#     for i in 1:REPS
#         dist = sample_conditional_marginal(rng, truncated_sde, test_x0, test_t)
#         marginal_samples[i] = rand(rng, dist)
#     end

#     # Compare with ground truth
#     test = ExactOneSampleKSTest(marginal_samples, marginal(true_sde, test_x0, test_t))
#     @test pvalue(test) > 0.1
# end

@testitem "Stable-Gaussian convolution" begin
    using LevyProcesses
    using Random
    using Test
    using StableDistributions
    using Distributions
    using StatsBase

    α = 0.8
    β = 0.3
    σ = 1.0
    μ = 0.5
    S = Stable(α, β, σ, μ)

    σy = 0.8
    μy = 0.2
    N = Normal(μy, σy)

    conv = StableGaussianConvolution(S, N)

    REPS = 10^6
    rng = MersenneTwister(5678)

    marginal_samples = Vector{Float64}(undef, REPS)
    for i in 1:REPS
        marginal_samples[i] = rand(rng, S) + rand(rng, N)
    end

    # Compare pdf to histogram on filtered samples
    filtered_samples = marginal_samples[abs.(marginal_samples .- (μ - μy)) .< 10.0]
    Z = length(filtered_samples) / REPS
    h = fit(Histogram, filtered_samples; nbins=200)
    bin_centers = (h.edges[1][1:(end - 1)] .+ h.edges[1][2:end]) ./ 2
    bin_width = h.edges[1][2] - h.edges[1][1]
    hist_pdf = h.weights ./ (sum(h.weights) * bin_width)
    analytical_pdf = pdf.(conv, bin_centers) ./ Z
    @test maximum(abs.(hist_pdf .- analytical_pdf) / analytical_pdf) < 1e-3

    # Plotting code for visual inspection
    # using Plots
    # Z = length(filtered_samples) / REPS
    # p = histogram(
    #     filtered_samples,
    #     nbins=200,
    #     normalize=true,
    #     label="Empirical",
    #     alpha=0.5,
    #     legend=:topright,
    # )
    # xs = range(μ - μy - 10.0, μ - μy + 10.0; length=1000)
    # plot!(xs, pdf.(conv, xs) ./ Z; label="Analytical", lw=2, color=:red)
end
