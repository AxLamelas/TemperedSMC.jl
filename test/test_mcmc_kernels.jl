using Test, Random, TemperedSMC, Distributions, LinearAlgebra, Statistics, PDMats, LogDensityProblems

include("test_utils.jl")

LD = LogDensityProblems

function test_kernel_stationarity(kernel, kernel_name; atol_mean=0.1, atol_cov=0.15)
    Random.seed!(80)

    # Fixed 2D test problem
    prior_ld, _, prior_dist = get_test_problem()

    # Oracle covariance (ground truth)
    Σ_true = cov(prior_dist)
    μ_true = mean(prior_dist)

    # Initial state
    x0 = rand(prior_dist)
    lp0 = LD.logdensity(prior_ld, x0)

    # Initialize state based on kernel's gradient requirement
    chain_state = if TemperedSMC.usesgrad(kernel)
        lp0, grad0 = LD.logdensity_and_gradient(prior_ld, x0)
        TemperedSMC.GradientChainState(x0, lp0, grad0)
    else
        TemperedSMC.ChainState(x0, lp0)
    end
    ker_state = TemperedSMC.init_kernel_state(kernel, x0, 1.0, PDMat(Σ_true))

    # Run MCMC chain
    chain_length = 3000
    burn_in = 500

    chain_samples = Matrix{Float64}(undef, length(x0), chain_length)
    accepted = 0

    for i in 1:chain_length
        chain_state, acc, _, ker_state = kernel(prior_ld, chain_state, ker_state)
        accepted += acc
        chain_samples[:, i] = chain_state.x
    end

    # Discard burn-in
    samples = chain_samples[:, burn_in+1:end]

    # Compute empirical mean and covariance
    μ_emp = mean(samples, dims=2) |> vec
    Σ_emp = cov(samples, dims=2)

    # Tolerance accounting for MCMC autocorrelation
    # Use a looser tolerance than iid MC error to account for correlation
    @test norm(μ_emp - μ_true) ≤ atol_mean
    @test norm(Σ_emp - Σ_true) ≤ atol_cov

    acceptance_rate = accepted / chain_length
    return acceptance_rate
end

@testset "RWMH kernel stationarity" begin
    kernel = TemperedSMC.RWMH()
    acc_rate = test_kernel_stationarity(kernel, "RWMH"; atol_mean=0.12, atol_cov=0.2)
    # RWMH should have acceptance rate in (0.1, 0.7) for Gaussian target
    @test 0.1 < acc_rate < 0.7
end

@testset "MALA kernel stationarity" begin
    kernel = TemperedSMC.MALA()
    acc_rate = test_kernel_stationarity(kernel, "MALA"; atol_mean=0.1, atol_cov=0.15)
    # MALA should have higher acceptance than RWMH on smooth targets
    @test 0.2 < acc_rate < 0.9
end

@testset "ULA kernel stationarity (with looser tolerance for discretization bias)" begin
    kernel = TemperedSMC.ULA()
    # ULA has O(ε) discretization bias; use looser tolerance than RWMH/MALA
    acc_rate = test_kernel_stationarity(kernel, "ULA"; atol_mean=0.25, atol_cov=0.35)
    # ULA is unadjusted, so no acceptance rate constraint
end
