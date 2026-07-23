using Test, Random, TemperedSMC, Distributions, LinearAlgebra, Statistics, LogDensityProblems

include("test_utils.jl")

LD = LogDensityProblems

@testset "waste_free_smc on conjugate Gaussian problem" begin
    Random.seed!(100)

    prior_ld, likelihood_ld, prior_dist = get_test_problem()
    μ_post_true, Σ_post_true, log_evidence_true = get_test_ground_truth()

    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    n_samples = 600
    initial_samples = rand(prior_dist, n_samples)

    result = TemperedSMC.waste_free_smc(
        seq,
        prior_ld,
        initial_samples;
        mcmc_kernel=TemperedSMC.RWMH(),
        show_progress=false,
        store_trace=false
    )

    # NaN canary
    @test !any(isnan, result.samples)

    # Check posterior mean and covariance
    μ_emp = mean(result.samples, dims=2) |> vec
    Σ_emp = cov(result.samples, dims=2)

    atol_mean = 0.15
    atol_cov = 0.25

    @test norm(μ_emp - μ_post_true) ≤ atol_mean
    @test norm(Σ_emp - Σ_post_true) ≤ atol_cov

    # Check evidence
    atol_evidence = 0.3
    @test abs(result.log_evidence - log_evidence_true) ≤ atol_evidence
end

@testset "waste_free_smc regression: n_starting indivisible by n_samples" begin
    Random.seed!(101)

    prior_ld, likelihood_ld, prior_dist = get_test_problem()

    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    # Non-divisible pair: this used to trigger DimensionMismatch in commit dd01a6c
    # Use n_starting=6 because adjustment formula div(100, round(Int,100/6)) yields 5
    n_samples = 100
    n_starting = 6

    initial_samples = rand(prior_dist, n_samples)

    # Should not crash with DimensionMismatch
    result = TemperedSMC.waste_free_smc(
        seq,
        prior_ld,
        initial_samples;
        mcmc_kernel=TemperedSMC.RWMH(),
        n_starting=n_starting,
        show_progress=false,
        store_trace=false
    )

    # Verify output structure is correct
    @test size(result.samples, 1) == TemperedSMC.LD.dimension(prior_ld)
    @test size(result.samples, 2) == n_samples

    # No stale/untouched columns (would manifest as NaN or all zeros)
    @test !any(isnan, result.samples)
    @test !all(iszero, result.samples)

    @test !any(isnan, result.log_evidence)
end
