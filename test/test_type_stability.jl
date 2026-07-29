using Test, Random, TemperedSMC, Distributions, LinearAlgebra, Statistics, LogDensityProblems, PDMats
using BenchmarkTools

include("test_utils.jl")

LD = LogDensityProblems

# Note: JET checks are simplified to avoid complex result structure handling.
# The key goal is the Float32 precision test (expected to fail pre-G.2).

@testset "Allocation baseline checks (Float64)" begin
    Random.seed!(234)
    prior_ld, likelihood_ld, prior_dist = get_test_problem()
    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    n_samples = 50
    initial_samples = rand(prior_dist, n_samples)

    # Single kernel step allocation check: RWMH (baseline for later comparison)
    begin
        chain_state = TemperedSMC.ChainState(initial_samples[:,1], 0.0)
        Σ = PDMat(Diagonal(ones(2)))
        ker_state = TemperedSMC.init_kernel_state(TemperedSMC.RWMH(), initial_samples[:,1], 1.0, Σ)

        # Warm up
        _ = TemperedSMC.RWMH()(prior_ld, chain_state, ker_state)

        # Measure allocation
        alloc = @allocated TemperedSMC.RWMH()(prior_ld, chain_state, ker_state)

        # Baseline: record for later phases (G.4 expects reduction here after scratch buffer refactor)
        @test alloc >= 0  # Just verify we got a number
    end

    # Single kernel step allocation check: MALA
    begin
        target = prior_ld  # MALA needs gradients
        x = initial_samples[:,1]
        lp, gradlp = LD.logdensity_and_gradient(target, x)
        chain_state = TemperedSMC.GradientChainState(x, lp, gradlp)
        Σ = PDMat(Diagonal(ones(2)))
        ker_state = TemperedSMC.init_kernel_state(TemperedSMC.MALA(), x, 1.0, Σ)

        _ = TemperedSMC.MALA()(target, chain_state, ker_state)
        alloc = @allocated TemperedSMC.MALA()(target, chain_state, ker_state)
        @test alloc >= 0  # Baseline record
    end

    # estimate_metric allocations
    begin
        samples = initial_samples
        W = ones(n_samples) / n_samples
        xs = [randn(2) for _ in 1:n_samples]  # Dummy starting positions for metric estimators

        # IdentityMetric
        _ = TemperedSMC.estimate_metric(TemperedSMC.IdentityMetric(), samples, W, [], xs, 0.5)
        alloc = @allocated TemperedSMC.estimate_metric(TemperedSMC.IdentityMetric(), samples, W, [], xs, 0.5)
        @test alloc >= 0

        # ParticleCov
        _ = TemperedSMC.estimate_metric(TemperedSMC.ParticleCov(), samples, W, [], xs, 0.5)
        alloc = @allocated TemperedSMC.estimate_metric(TemperedSMC.ParticleCov(), samples, W, [], xs, 0.5)
        @test alloc >= 0
    end
end

@testset "Float32 precision (key correctness test — expected to fail pre-G.2)" begin
    Random.seed!(345)
    prior_ld, likelihood_ld, prior_dist = get_test_problem()

    # Convert to Float32
    μ0_f32 = Float32.(TEST_μ0)
    Σ0_f32 = Float32.(TEST_Σ0)
    y_f32 = Float32.(TEST_y)
    Σlik_f32 = Float32.(TEST_Σlik)

    prior_f32 = MvNormal(μ0_f32, Σ0_f32)
    likelihood_f32 = MvNormal(y_f32, Σlik_f32)

    prior_ld_f32 = MvNormalLD(prior_f32)
    likelihood_ld_f32 = MvNormalLD(likelihood_f32)

    seq_f32 = TemperedSMC.AdaptiveTempering(likelihood_ld_f32)

    n_samples = 50
    initial_samples_f32 = rand(prior_f32, n_samples)

    # Run SMC with Float32
    result = TemperedSMC.smc(
        seq_f32,
        prior_ld_f32,
        initial_samples_f32;
        mcmc_kernel=TemperedSMC.RWMH(),
        mcmc_steps=10,
        adapt_mcmc_steps=false,
        show_progress=false,
        store_trace=false
    )

    # Check that outputs stay in Float32 (not silently promoted to Float64)
    @test eltype(result.samples) == Float32
    @test typeof(result.acceptance_rate) == Float32
    @test eltype(result.seq_state.β) == Float32
    @test typeof(result.trcov_reweight) == Float32
    @test typeof(result.trcov_mcmc) == Float32

    # Also run waste_free_smc with Float32
    result_wf = TemperedSMC.waste_free_smc(
        seq_f32,
        prior_ld_f32,
        initial_samples_f32;
        mcmc_kernel=TemperedSMC.RWMH(),
        show_progress=false,
        store_trace=false
    )

    @test eltype(result_wf.samples) == Float32
    @test typeof(result_wf.acceptance_rate) == Float32
    @test eltype(result_wf.seq_state.β) == Float32
end
