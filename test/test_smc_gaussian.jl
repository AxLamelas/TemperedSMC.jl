using Test, Random, TemperedSMC, Distributions, LinearAlgebra, Statistics, LogDensityProblems

include("test_utils.jl")

LD = LogDensityProblems

@testset "smc with RWMH on conjugate Gaussian problem" begin
    Random.seed!(90)

    prior_ld, likelihood_ld, prior_dist = get_test_problem()
    μ_post_true, Σ_post_true, log_evidence_true = get_test_ground_truth()

    # Create tempered sequence with the likelihood
    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    # Initial samples from the prior
    n_samples = 600
    initial_samples = rand(prior_dist, n_samples)

    # Run SMC with RWMH
    # smc(seq, ref_logdensity, initial_samples; ...)
    result = TemperedSMC.smc(
        seq,
        prior_ld,
        initial_samples;
        mcmc_kernel=TemperedSMC.RWMH(),
        mcmc_steps=25,
        adapt_mcmc_steps=false,
        show_progress=false,
        store_trace=false
    )

    # NaN canary
    @test !any(isnan, result.samples)

    # Check posterior mean and covariance against analytic ground truth
    μ_emp = mean(result.samples, dims=2) |> vec
    Σ_emp = cov(result.samples, dims=2)

    # Tolerances calibrated for n_samples~600, accounting for resampling correlation
    atol_mean = 0.15
    atol_cov = 0.25

    @test norm(μ_emp - μ_post_true) ≤ atol_mean
    @test norm(Σ_emp - Σ_post_true) ≤ atol_cov

    # Check evidence estimate
    atol_evidence = 0.3
    @test abs(result.log_evidence - log_evidence_true) ≤ atol_evidence
end

@testset "smc with MALA on conjugate Gaussian problem" begin
    Random.seed!(91)

    prior_ld, likelihood_ld, prior_dist = get_test_problem()
    μ_post_true, Σ_post_true, log_evidence_true = get_test_ground_truth()

    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    n_samples = 600
    initial_samples = rand(prior_dist, n_samples)

    result = TemperedSMC.smc(
        seq,
        prior_ld,
        initial_samples;
        mcmc_kernel=TemperedSMC.MALA(),
        mcmc_steps=25,
        adapt_mcmc_steps=false,
        show_progress=false,
        store_trace=false
    )

    # NaN canary
    @test !any(isnan, result.samples)

    # Check posterior
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
