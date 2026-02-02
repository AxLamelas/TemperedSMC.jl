abstract type AbstractCovEstimator end

function estimate_cov(_::AbstractCovEstimator,samples,weights) end

struct IdentityCov <: AbstractCovEstimator end

estimate_cov(_::IdentityCov,samples,weights,xs) = fill(Matrix(one(eltype(samples)) * I, size(samples,1),size(samples,1)),length(xs))

Base.@kwdef @concrete struct ParticleCov <: AbstractCovEstimator 
  method = LinearShrinkage(DiagonalUnequalVariance())
end

function estimate_cov(c::ParticleCov,samples,weights,xs) 
  if size(samples,1) == 1
    v = var(samples,FrequencyWeights(weights))
    [PDMat(reshape([v],1,1)) for _ in xs]
  else
    fill(
      PDMat(cov(
        c.method,samples,FrequencyWeights(weights),dims=2
      ))
      ,length(xs))
  end
end

Base.@kwdef @concrete struct KernelCov <: AbstractCovEstimator
  max_samples = 1000
  resampler = ResidualResampler()
  γ = 0.05
end

struct ParticleVar <: AbstractCovEstimator end

function estimate_cov(c::ParticleVar,samples,weights,xs) 
  if size(samples,1) == 1
    v = var(samples,FrequencyWeights(weights))
    [PDMat(reshape([v],1,1)) for _ in xs]
  else
    fill(PDMat(Diagonal(var(samples,FrequencyWeights(weights),2))),length(xs))
  end
end

function _kernel_estimate(c::KernelCov,V,ref_samples,wfun,xs)
  n_dims,n_samples = size(ref_samples)
  H = I - ones(n_samples,n_samples)/n_samples
  M = similar(H,n_dims,n_samples)

  refdists = pairwise(Euclidean(),ref_samples)
  lengthscale = median(refdists) / sqrt(2) + 1e-8

  # Covariance in kernel space from Kernel Adaptive Metropolis-Hastings
  xzdists = pairwise(SqEuclidean(),xs,eachcol(ref_samples)) 
  return map(eachindex(xs)) do i
    for (j,z) in enumerate(eachcol(ref_samples))
      @. M[:,j] = wfun(j) * 2/lengthscale^2 * exp(-0.5*xzdists[i,j]/lengthscale^2) * (z - xs[i])
    end
    PDMat(c.γ*V + M * H * M')
  end
end

function estimate_cov(c::KernelCov,samples,weights,xs)
  V = Diagonal(var(samples,FrequencyWeights(weights),2))
  n_samples = size(samples,2)
  if n_samples > c.max_samples
    inds = c.resampler(weights,c.max_samples)
    _kernel_estimate(c,V,samples[:,inds],i -> 1,xs)
  else
    _kernel_estimate(c,V,samples, i->sqrt(n_samples*weights[i]),xs)
  end
end

