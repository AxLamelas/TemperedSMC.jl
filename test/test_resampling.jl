using Test, Random, StatsBase, TemperedSMC

@testset "ResidualResampler" begin
    Random.seed!(50)

    N = 1000
    weights = normalize([1.5, 2.3, 0.8, 1.4], 1)
    resampler = TemperedSMC.ResidualResampler()

    indices = resampler(weights, N)

    @test length(indices) == N
    counts = StatsBase.countmap(indices)

    for i in 1:length(weights)
        count = get(counts, i, 0)
        min_count = floor(Int, N * weights[i])
        max_count = ceil(Int, N * weights[i])
        @test min_count ≤ count ≤ max_count
    end
    @test sum(values(counts)) == N
end

@testset "SystematicResampler" begin
    Random.seed!(51)

    N = 1000
    weights = normalize([1.5, 2.3, 0.8, 1.4], 1)
    resampler = TemperedSMC.SystematicResampler()

    indices = resampler(weights, N)

    @test length(indices) == N
    counts = StatsBase.countmap(indices)

    for i in 1:length(weights)
        count = get(counts, i, 0)
        min_count = floor(Int, N * weights[i])
        max_count = ceil(Int, N * weights[i])
        @test min_count ≤ count ≤ max_count
    end
    @test sum(values(counts)) == N
end

@testset "StratifiedResampler" begin
    Random.seed!(52)

    N = 1000
    weights = normalize([1.5, 2.3, 0.8, 1.4], 1)
    resampler = TemperedSMC.StratifiedResampler()

    indices = resampler(weights, N)

    @test length(indices) == N
    counts = StatsBase.countmap(indices)

    for i in 1:length(weights)
        count = get(counts, i, 0)
        min_count = floor(Int, N * weights[i])
        max_count = ceil(Int, N * weights[i])
        @test min_count ≤ count ≤ max_count
    end
    @test sum(values(counts)) == N
end

@testset "SSPResampler" begin
    Random.seed!(53)

    N = 1000
    weights = normalize([1.5, 2.3, 0.8, 1.4], 1)
    resampler = TemperedSMC.SSPResampler()

    indices = resampler(weights, N)

    @test length(indices) == N
    counts = StatsBase.countmap(indices)

    for i in 1:length(weights)
        count = get(counts, i, 0)
        min_count = floor(Int, N * weights[i])
        max_count = ceil(Int, N * weights[i])
        @test min_count ≤ count ≤ max_count
    end
    @test sum(values(counts)) == N
end

@testset "MultinomialResampler statistical" begin
    Random.seed!(54)

    N = 5000
    weights = normalize([1.5, 2.3, 0.8, 1.4], 1)
    resampler = TemperedSMC.MultinomialResampler()

    indices = resampler(weights, N)

    @test length(indices) == N
    counts = StatsBase.countmap(indices)

    for i in 1:length(weights)
        count = get(counts, i, 0)
        expected = N * weights[i]
        # Binomial variance: np(1-p); allow ±3σ for statistical tolerance
        variance = N * weights[i] * (1 - weights[i])
        std_dev = sqrt(variance)
        @test abs(count - expected) ≤ 3 * std_dev
    end
    @test sum(values(counts)) == N
end
