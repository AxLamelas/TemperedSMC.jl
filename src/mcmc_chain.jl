struct MCMCProgressCallback{P <: ProgressMeter.Progress}
  prog::P
end

function MCMCProgressCallback(n::Int)
  return MCMCProgressCallback(Progress(n,desc="Sampling: ",showspeed=true))
end

function (cb::MCMCProgressCallback)(info)
  ProgressMeter.next!(cb.prog,
                      showvalues=[
                      ("log density",info.log_density),
                      ("accepted?",info.accepted)])
  nothing
end

struct NoCallback end
(cb::NoCallback)(args...) = nothing

function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x,state,n_samples::Int;callback=NoCallback())
  n_accepts = 0

  ref_lp, ref_grad = LD.logdensity_and_gradient(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  gradlps = Vector{Vector{T}}(undef,n_samples)
  γ = Vector{T}(undef,n_samples-1)

  samples[1] = x
  lps[1] = ref_lp
  gradlps[1] = ref_grad

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],gradlps[i+1],acc,γ[i],state =
      mcmc_kernel(target,samples[i],lps[i],gradlps[i],state)
    n_accepts += acc
    callback(
      (;
        sample = samples[i+1],
        log_density=lps[i+1],
        grad_log_density=gradlps[i+1],
        accepted=acc,
        γ=γ[i],
        state)
    )
  end


  return (;n_accepts,samples,lps,gradlps,state,γ)
end

function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{false}},target,x,state,n_samples::Int;callback=NoCallback())
  n_accepts = 0

  ref_lp = LD.logdensity(target,x)
  T = eltype(x)
  samples = Vector{Vector{T}}(undef,n_samples)
  lps = Vector{typeof(ref_lp)}(undef,n_samples)
  γ = Vector{T}(undef,n_samples-1)

  samples[1] = x
  lps[1] = ref_lp

  for i in 1:n_samples-1
    samples[i+1],lps[i+1],acc,γ[i],state = mcmc_kernel(target,samples[i],lps[i],state)
    n_accepts += acc
    callback(
      (;
        sample = samples[i+1],
        log_density=lps[i+1],
        accepted=acc,
        γ=γ[i],
        state)
    )
  end

  return (;n_accepts,samples,lps,state,γ)
end

