using Test, Random, TemperedSMC, LogDensityProblems, Statistics

include("test_utils.jl")

LD = LogDensityProblems

function compute_cess(lw::Vector, α::Float64)
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    ess = 1.0 / sum(w .^ 2)
    ness = ess / length(lw)
    return ness
end

@testset "AdaptiveTempering CESS bisection convergence" begin
    Random.seed!(60)

    # Create synthetic tempered state and log-weight vector
    dim = 2
    n_particles = 500

    _, _, prior_dist = get_test_problem()

    # Generate particles from the prior
    samples = rand(prior_dist, n_particles)

    # Compute log-densities at β=0 (reference only) and at various β values
    prior_ld, likelihood_ld, _ = get_test_problem()
    log_ref = [LD.logdensity(prior_ld, samples[:, i]) for i in 1:n_particles]

    # Create a state object as it would appear in the algorithm
    # TemperedState(ℓ, β) where ℓ is log density and β is temperature vector
    state = TemperedSMC.TemperedState(copy(log_ref), [0.0])

    # Compute likelihood increments for a range of β
    function compute_loglik_increments(β)
        [LD.logdensity(likelihood_ld, samples[:, i]) for i in 1:n_particles] .* β
    end

    # Simulate one adaptive tempering step
    log_lik_increments = compute_loglik_increments(0.5)
    lw = log_lik_increments

    α = 0.8
    β_next = TemperedSMC._next_β(state, lw, α)

    @test 0.0 < β_next ≤ 1.0
    @test !isnan(β_next)
    @test !isinf(β_next)

    # Verify the returned β satisfies the CESS condition
    log_lik_full = compute_loglik_increments(β_next)
    cess_achieved = compute_cess(log_lik_full, α)
    @test abs(cess_achieved - α) ≤ 1e-7
end

@testset "AdaptiveTempering near-degenerate case" begin
    Random.seed!(61)

    # Create a nearly degenerate case with tiny log-weight spread
    dim = 2
    n_particles = 100

    prior_ld, likelihood_ld, prior_dist = get_test_problem()
    samples = rand(prior_dist, n_particles)
    log_ref = [LD.logdensity(prior_ld, samples[:, i]) for i in 1:n_particles]

    state = TemperedSMC.TemperedState(copy(log_ref), [0.0])

    # Create nearly constant log-weights (small likelihood variation)
    lw = fill(1e-8, n_particles) .+ randn(n_particles) .* 1e-10

    α = 0.8
    β_next = TemperedSMC._next_β(state, lw, α)

    @test 0.0 < β_next ≤ 1.0
    @test !isnan(β_next)
    @test !isinf(β_next)
end
