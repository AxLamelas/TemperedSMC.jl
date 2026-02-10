
struct MCMCProgressCallback{P <: ProgressMeter.Progress}
  prog::P
end

function MCMCProgressCallback(n::Int)
  return MCMCProgressCallback(Progress(n,desc="Sampling: ",showspeed=true))
end

function (cb::MCMCProgressCallback)(info)
  ProgressMeter.next!(cb.prog,
                      showvalues=[
                      ("log density",info.chain_state.logp),
                      ("accepted?",info.accepted)])
  nothing
end

struct NoCallback end
(cb::NoCallback)(args...) = nothing

iterate_mcmc(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x::AbstractVector,state,n_steps::Int) = 
  iterate_mcmc(mcmc_kernel,target,GradientChainState(x,LD.logdensity_and_gradient(target,x)...),state,n_steps)

iterate_mcmc(mcmc_kernel::AbstractMCMCKernel{Val{false}},target,x::AbstractVector,state,n_steps::Int) = 
  iterate_mcmc(mcmc_kernel,target,ChainState(x,LD.logdensity(target,x)),state,n_steps)

function iterate_mcmc(mcmc_kernel::AbstractMCMCKernel,target,chain_state::AbstractChainState,state,n_steps::Int;
                         γ = 1.,n_accepts =0)
  for i in 1:n_steps
    chain_state,acc,γi,state = mcmc_kernel(target,chain_state,state)
    n_accepts += acc
    γ *= γi
  end

  return (;n_accepts,chain_state,kernel_state=state,γ)
end


function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{true}},target,x,state,n_samples::Int;callback=NoCallback())
  n_accepts = 0

  T = eltype(x)
  chain_state = GradientChainState(x,LD.logdensity_and_gradient(target,x)...)
  chain = Vector{typeof(chain_state)}(undef,n_samples)
  chain[1] = chain_state
  γ = Vector{T}(undef,n_samples-1)


  for i in 1:n_samples-1
    chain[i+1],acc,γ[i],state =
      mcmc_kernel(target,chain[i],state)
    n_accepts += acc
    callback(
      (;
        chain_state = chain[i+1],
        accepted=acc,
        γ=γ[i],
        state
      )
    )
  end


  return (;n_accepts,states=chain,kernel_state=state,γ)
end


function mcmc_chain(mcmc_kernel::AbstractMCMCKernel{Val{false}},target,x,state,n_samples::Int;callback=NoCallback())
  n_accepts = 0

  T = eltype(x)
  chain_state = ChainState(x,LD.logdensity(target,x)...)
  chain = Vector{typeof(chain_state)}(undef,n_samples)
  chain[1] = chain_state
  γ = Vector{T}(undef,n_samples-1)


  for i in 1:n_samples-1
    chain[i+1],acc,γ[i],state =
      mcmc_kernel(target,chain[i],state)
    n_accepts += acc
    callback(
      (;
        chain_state = chain[i+1],
        accepted=acc,
        γ=γ[i],
        state
      )
    )
  end


  return (;n_accepts,states=chain,kernel_state=state,γ)
end
