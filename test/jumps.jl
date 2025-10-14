using Test
using LevyProcesses

@testitem "Ragged Jumps Replacement" begin
    using LevyProcesses
    using Test

    p = TruncatedLevyProcess(GammaProcess(1.3f0, 10.8f0); l=Float32(1e-10))
    dt = 1.0f0
    N = 10
    K = 2

    new_jumps = sample(p, dt, K, BatchRejection)
    cpu_jump_offsets = Array(new_jumps.offsets)
    cpu_jump_nums = [cpu_jump_offsets[1], diff(cpu_jump_offsets)...]

    jumps = sample(p, dt, N - K, BatchRejection; spare_slots=cpu_jump_nums)
    jumps[:, (N - K + 1):N] = new_jumps

    @test jumps.jump_sizes == vcat(jumps.jump_sizes[1:(N - K)], new_jumps.jump_sizes)
end
