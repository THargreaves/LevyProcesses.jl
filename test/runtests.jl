using HypothesisTests
using LevyProcesses
using QuadGK
using Random
using Test

@testset "Process tests" begin

    @testset "Gamma process tests" begin
        include("processes/gamma.jl")
    end

    @testset "Stable process tests" begin
        include("processes/stable.jl")
    end

end