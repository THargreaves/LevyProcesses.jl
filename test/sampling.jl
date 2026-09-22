@testitem "Finite jump sampling" begin
    using Random

    rng = MersenneTwister(42)
    LP = LevyProcesses
    p = TruncatedLevyProcess(LP.GammaDominatingProcess(2f0, 1f0), 0.1f0, 2f0)
    empty_path = sample(rng, p, 0f0, Inversion; sort_times=true)
    @test isempty(empty_path.jump_times)
    @test isempty(sample(p, 0f0).jump_times)
    @test eltype(empty_path.jump_times) == eltype(empty_path.jump_sizes) == Float32
    @test isempty(LP.sample_uniforms(rng, 0; sorted=true))

    jumps = sample(rng, p, 3f0, Inversion; sort_times=true)
    @test eltype(jumps.jump_times) == eltype(jumps.jump_sizes) == Float32
    @test issorted(jumps.jump_times) && all(0 .<= jumps.jump_times .<= 3)
    @test all(p.lower .<= jumps.jump_sizes .<= p.upper)

    # A process used as its own envelope must retain every proposed jump.
    proposal = sample(MersenneTwister(7), p, 3f0)
    accepted = sample(MersenneTwister(7), p, 3f0, p, Rejection)
    @test accepted.jump_sizes == proposal.jump_sizes
    @test accepted.jump_times == proposal.jump_times

    @test_throws ArgumentError sample(rng, p, -1f0)
    @test_throws ArgumentError sample(rng, p, Inf32)
    infinite_activity = TruncatedLevyProcess(LP.GammaDominatingProcess(2.0, 1.0), 0.0, Inf)
    @test_throws ArgumentError sample(rng, infinite_activity, 1.0)
    signed = TruncatedLevyProcess(StableProcess(0.7, 0.2, 1.0), 0.1, 2.0)
    @test_throws ArgumentError sample(rng, signed, 1.0, Inversion)
end

@testitem "GPU jump sampling" begin
    using CUDA

    if CUDA.functional()
        p = TruncatedLevyProcess(StableSubordinator(0.7, 1.0), 0.1, 2.0)
        jumps = sample(p, 1.0, 8, BatchInversion)
        offsets = Array(jumps.offsets)
        @test eltype(offsets) == Int32
        @test issorted(offsets) && last(offsets) == jumps.tot_N == length(jumps.jump_sizes)
        @test all(p.lower .<= Array(jumps.jump_sizes) .<= p.upper)
        empty_batch = sample(p, 0.0, 8, BatchInversion)
        @test iszero(empty_batch.tot_N) && all(iszero, Array(empty_batch.offsets))
    else
        @test_skip CUDA.functional()
    end
end
