using HypothesisTests
using QuadGK
using Random
using SpecialFunctions
using Statistics
using Test

using LevyProcesses

@testset "Process tests" begin

    @testset "Gamma process tests" begin
        include("processes/gamma.jl")
    end

    @testset "Stable subordinator tests" begin
        include("processes/stable_subordinator.jl")
    end

    @testset "Stable process tests" begin
        include("processes/stable.jl")
    end

    @testset "Normal variance mean process tests" begin
        include("processes/nvm.jl")
    end

    @testset "SDE tests" begin
        include("sdes.jl")
    end

    @testset "Jumps tests" begin
        include("jumps.jl")
    end

end