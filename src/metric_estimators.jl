abstract type AbstractMetric end

function estimate_metric(_::AbstractMetric,samples,weights,states,xs) end

struct IdentityMetric <: AbstractMetric end

estimate_metric(_::IdentityMetric,samples,weights,_,xs) = fill(PDMat(Diagonal(ones(eltype(samples),size(samples,1)))),length(xs))

struct ParticleCov <: AbstractMetric end

function estimate_metric(c::ParticleCov,samples,weights,_,xs)
	if size(samples,1) == 1
		v = var(samples,FrequencyWeights(weights))
		Fill(PDMat(reshape([v],1,1)),length(xs))
	else
		Fill(PDMat(ensure_posdef(cov(
			samples,FrequencyWeights(weights),2
		))),length(xs))
	end
end

Base.@kwdef @concrete struct KernelCov <: AbstractMetric
  max_samples = 1000
  resampler = ResidualResampler()
  γ = 0.05
end

struct ParticleVar <: AbstractMetric end

function estimate_metric(c::ParticleVar,samples,weights,_,xs)
	if size(samples,1) == 1
		v = var(samples,FrequencyWeights(weights))
		Fill(PDMat(reshape([v],1,1)),length(xs))
	else
		Fill(PDMat(Diagonal(vec(var(samples,FrequencyWeights(weights),2)))),length(xs))
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

function estimate_metric(c::KernelCov,samples,weights,_,xs)
  V = Diagonal(var(samples,FrequencyWeights(weights),2))
  n_samples = size(samples,2)
  if n_samples > c.max_samples
    inds = c.resampler(weights,c.max_samples)
    _kernel_estimate(c,V,samples[:,inds],i -> 1,xs)
  else
    _kernel_estimate(c,V,samples, i->sqrt(n_samples*weights[i]),xs)
  end
end

struct EmpiricalFisher <: AbstractMetric end

function estimate_metric(c::EmpiricalFisher,samples,weights,states::AbstractVector{<:GradientChainState},xs)
  # TODO:  Make this more efficient
  F = mean(states) do s
    s.gradlogp * s.gradlogp'
  end

  Fill(PDMat(ensure_posdef_and_invert(F)),length(xs))
end

struct ParticleRepresentation <: AbstractMetric end

function estimate_metric(_::ParticleRepresentation,samples,weights,states,xs)
	Fill(samples,length(xs))
end
