using Test, Random, TemperedSMC, LogDensityProblems

include("test_utils.jl")

LD = LogDensityProblems

@testset "MetaNumber" begin
    Random.seed!(42)

    m1 = TemperedSMC.MetaNumber(3.0, (;meta=:test1))
    m2 = TemperedSMC.MetaNumber(2.0, (;meta=:test2))

    # Arithmetic demotes to plain values
    @test +(m1, m2) == 5.0
    @test -(m1, m2) == 1.0
    @test *(m1, m2) == 6.0
    @test /(m1, m2) ≈ 1.5
    @test m1^2 == 9.0

    # Special values (zero and one return plain numbers, not MetaNumbers)
    @test zero(m1) == 0.0
    @test one(m1) == 1.0

    # Comparisons with infinity
    @test !isinf(m1)
    m_inf = TemperedSMC.MetaNumber(Inf, (;))
    @test isinf(m_inf)

    # Type conversion
    @test convert(Float64, m1) == 3.0
end

@testset "FullLogDensity arithmetic and gradients" begin
    Random.seed!(43)

    prior, likelihood, _ = get_test_problem()
    full = TemperedSMC.FullLogDensity(prior, likelihood)

    θ = [1.0, -0.5]

    # Test logdensity value
    ld_prior = LD.logdensity(prior, θ)
    ld_likelihood = LD.logdensity(likelihood, θ)
    ld_full = LD.logdensity(full, θ)
    @test ld_full ≈ ld_prior + ld_likelihood atol=1e-10

    # Test gradient
    _, g_prior = LD.logdensity_and_gradient(prior, θ)
    _, g_likelihood = LD.logdensity_and_gradient(likelihood, θ)
    lp_full, g_full = LD.logdensity_and_gradient(full, θ)

    @test lp_full ≈ ld_prior + ld_likelihood atol=1e-10
    @test g_full ≈ g_prior + g_likelihood atol=1e-10
end

@testset "FullLogDensity short-circuit at -Inf reference" begin
    Random.seed!(44)

    struct DummyRefLD
        logdensity_return::Float64
    end
    LD.dimension(::DummyRefLD) = 2
    LD.capabilities(::Type{DummyRefLD}) = LD.LogDensityOrder{1}()
    LD.logdensity(d::DummyRefLD, θ) = d.logdensity_return
    function LD.logdensity_and_gradient(d::DummyRefLD, θ)
        return d.logdensity_return, zeros(2)
    end

    ref_minus_inf = DummyRefLD(-Inf)
    _, likelihood, _ = get_test_problem()
    full = TemperedSMC.FullLogDensity(ref_minus_inf, likelihood)

    θ = [1.0, -0.5]

    # Should short-circuit and return -Inf
    @test LD.logdensity(full, θ) == -Inf
    lp, g = LD.logdensity_and_gradient(full, θ)
    @test lp == -Inf
    @test g == zeros(2)
end
