
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
					  logγ = 0.,n_accepts =0)
	if isnan(logγ)
		throw(error("Called with NaN logγ"))
	end
	for i in 1:n_steps
		chain_state,acc,γi,state = mcmc_kernel(target,chain_state,state)
		n_accepts += acc
		logγ += if acc 
			log(γi)
		else
			log(1-γi)
		end
		if isnan(logγ)
			throw(error("Resulted in NaN logγ when updating with $(γi) and move was acc=$(acc)"))
		end
	end

	return (;n_accepts,chain_state,kernel_state=state,logγ)
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
  chain_state = ChainState(x,LD.logdensity(target,x))
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

# ============================================================================
# Population Kernel MCMC: mcmc_chain overload
# ============================================================================

"""
    mcmc_chain(mcmc_kernel::AbstractPopulationKernel, target, xs::Vector{<:AbstractVector}, state::PopulationKernelState, n_samples)

Run population MCMC chain for n_samples steps, tracking full history.
Returns time-major data: Vector{PopulationChainState}, each storing all particles at a time step.
"""
function mcmc_chain(mcmc_kernel::AbstractPopulationKernel, target,
                    xs::Vector{<:AbstractVector}, state::PopulationKernelState, n_samples::Int;
                    callback=NoCallback())
    chain_state = init_chain_state(mcmc_kernel, target, xs)
    n = length(chain_state)
    n_accepts = zeros(Int, n)
    chain = Vector{typeof(chain_state)}(undef, n_samples)
    chain[1] = chain_state
    γ = Vector{Vector{Float64}}(undef, n_samples - 1)

    for i in 1:n_samples-1
        chain[i+1], acc, γ[i], state = mcmc_kernel(target, chain[i], state)
        n_accepts .+= acc
        callback((;chain_state=chain[i+1], accepted=acc, γ=γ[i], state))
    end

    return (;n_accepts, states=chain, kernel_state=state, γ)
end

# ============================================================================
# Transpose helper: convert time-major to per-particle NamedTuple shape
# ============================================================================

"""
    per_particle_chains(result)

Transpose population-level MCMC result (time-major) to per-particle chains (particle-major).
Converts Vector{PopulationChainState} indexed by time step to Vector of per-particle NamedTuples
with fields (n_accepts, states::Vector, γ::Vector, kernel_state), reproducing the shape
expected by kernel_parameters.jl and smc.jl's write-back loops.
"""
function per_particle_chains(result)
    n_particles = length(result.states[1])
    n_time = length(result.states)
    n_γsteps = length(result.γ)
    [
        (;n_accepts = result.n_accepts[j],
          states    = [result.states[i][j] for i in 1:n_time],
          γ         = [result.γ[i][j] for i in 1:n_γsteps],
          kernel_state = result.kernel_state.states[j])
        for j in 1:n_particles
    ]
end
