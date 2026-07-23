using Distributions, LogDensityProblems, LinearAlgebra, Statistics, Random

LD = LogDensityProblems

struct MvNormalLD{D<:AbstractMvNormal}
    d::D
end

LD.dimension(ℓ::MvNormalLD) = length(ℓ.d)
LD.capabilities(::Type{<:MvNormalLD}) = LD.LogDensityOrder{1}()

function LD.logdensity(ℓ::MvNormalLD, θ)
    logpdf(ℓ.d, θ)
end

function LD.logdensity_and_gradient(ℓ::MvNormalLD, θ)
    lp = logpdf(ℓ.d, θ)
    g = -(invcov(ℓ.d) * (θ - mean(ℓ.d)))
    return lp, g
end

function gaussian_conjugate_posterior(μ0, Σ0, y, Σlik)
    Λ0, Λlik = inv(Σ0), inv(Σlik)
    Σpost = inv(Λ0 + Λlik)
    μpost = Σpost * (Λ0 * μ0 + Λlik * y)
    return μpost, Symmetric(Σpost)
end

function gaussian_log_evidence(μ0, Σ0, y, Σlik)
    logpdf(MvNormal(μ0, Σ0 + Σlik), y)
end

# Fixed 2D test problem: conjugate Gaussian-Gaussian
const TEST_μ0 = [0.0, 0.0]
const TEST_Σ0 = [1.0 0.3; 0.3 1.0]
const TEST_y = [1.5, -0.5]
const TEST_Σlik = [0.5 -0.1; -0.1 0.8]

function get_test_problem()
    prior = MvNormal(TEST_μ0, TEST_Σ0)
    likelihood = MvNormal(TEST_y, TEST_Σlik)
    return MvNormalLD(prior), MvNormalLD(likelihood), prior
end

function get_test_ground_truth()
    μ_post, Σ_post = gaussian_conjugate_posterior(TEST_μ0, TEST_Σ0, TEST_y, TEST_Σlik)
    log_evidence = gaussian_log_evidence(TEST_μ0, TEST_Σ0, TEST_y, TEST_Σlik)
    return μ_post, Σ_post, log_evidence
end
