using Test, Random, TemperedSMC, Distributions, LinearAlgebra, Statistics, PDMats

include("test_utils.jl")

@testset "IdentityMetric exact" begin
    Random.seed!(70)

    dim = 3
    metric = TemperedSMC.IdentityMetric()

    samples = randn(dim, 100)
    weights = ones(100) / 100

    Σ_est_list = TemperedSMC.estimate_metric(metric, samples, weights, [], [randn(dim)])

    # IdentityMetric returns a Fill array of identity matrices
    expected = I(dim)
    @test Σ_est_list[1] isa TemperedSMC.PDMat
    # Check that the diagonal is all ones
    @test diag(Matrix(Σ_est_list[1])) ≈ ones(dim) atol=1e-10
end

@testset "ParticleCov vs true covariance" begin
    Random.seed!(71)

    # Use the fixed 2D test problem
    _, _, prior_dist = get_test_problem()

    n_samples = 2000
    samples = rand(prior_dist, n_samples)
    weights = ones(n_samples) / n_samples

    metric = TemperedSMC.ParticleCov()
    Σ_est_list = TemperedSMC.estimate_metric(metric, samples, weights, [], [randn(2)])
    Σ_est = Matrix(Σ_est_list[1])

    # True covariance
    Σ_true = cov(prior_dist)

    # Monte Carlo error at n=2000 is roughly sqrt(tr(Σ_true^2) / n) in Frobenius norm
    # Allow ±2σ for sampling variability
    error_bound = 2.0 * sqrt(tr(Σ_true^2) / n_samples)
    @test norm(Σ_est - Σ_true) ≤ error_bound
end

@testset "ParticleVar vs true variance (scalar case)" begin
    Random.seed!(72)

    # 1D Gaussian
    dim = 1
    μ = 0.0
    σ = 1.5
    dist = Normal(μ, σ)

    n_samples = 2000
    samples = reshape(rand(dist, n_samples), 1, n_samples)
    weights = ones(n_samples) / n_samples

    metric = TemperedSMC.ParticleVar()
    Σ_est_list = TemperedSMC.estimate_metric(metric, samples, weights, [], [randn(1)])
    Σ_est = Matrix(Σ_est_list[1])

    # True variance
    σ_true_sq = σ^2

    # In the scalar case, ParticleVar returns a 1x1 matrix
    Σ_true = fill(σ_true_sq, 1, 1)

    # Monte Carlo error tolerance
    error_bound = 2.0 * sqrt(2 * σ_true_sq^2 / n_samples)
    @test norm(Σ_est - Σ_true) ≤ error_bound
end

@testset "ParticleVar vs true covariance (matrix case)" begin
    Random.seed!(73)

    _, _, prior_dist = get_test_problem()

    n_samples = 2000
    samples = rand(prior_dist, n_samples)
    weights = ones(n_samples) / n_samples

    metric = TemperedSMC.ParticleVar()
    Σ_est_list = TemperedSMC.estimate_metric(metric, samples, weights, [], [randn(2)])
    Σ_est = Matrix(Σ_est_list[1])

    # True covariance
    Σ_true = cov(prior_dist)

    # Statistical tolerance
    error_bound = 2.0 * sqrt(tr(Σ_true^2) / n_samples)
    @test norm(Σ_est - Σ_true) ≤ error_bound
end
