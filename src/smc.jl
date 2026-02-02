mutable struct SMCState{T,S<:AbstractSequenceState{T},M}
  # Log Density sequence
  seq_state::S
  # Particles
  samples::Matrix{T}
  # Normalized Weights
  W::Vector{T}
  # unormalized log weights
  lw::Vector{T}
  # Target log density
  ℓ::M
  log_evidence::T
  acceptance_rate::Float64
  resampled::Bool
end

function SMCState(seq::AbstractDistributionSequence,initial_samples,map_func)
  target = initial_logdensity(seq)
  ℓ = stabilized_map(
    Base.Fix1(LD.logdensity,target),eachcol(initial_samples),map_func)

  seq_state = TemperedState(ℓ)
  T = eltype(seq_state)
  lw = convert.(T,ℓ) # Could possibly be a metanumber 
  lw_norm_constant = logsumexp(lw)
  W = exp.(lw .- lw_norm_constant)

  n_samples = length(ℓ)

  return SMCState(seq_state,deepcopy(initial_samples),W,lw,ℓ,zero(T),0.,false)
end

function smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples;
             mcmc_kernel::AbstractMCMCKernel = _default_sampler(ref_logdensity,mul_logdensity),
             cov_estimator::AbstractCovEstimator = ParticleCov(),
             resampler::AbstractResampler = SSPResampler(),
             resampling_α = 0.5,
             mcmc_steps = 5,
             map_func = map,
             callback=(_) -> false,
             store_trace = true,
             show_progress = true
             )


  n_dims, n_samples = size(initial_samples)
  lN = -log(n_samples)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9,enabled = show_progress)

  state = SMCState(seq,initial_samples,map_func)
  trace = typeof(state)[]

  indices_no_resampling = collect(1:n_samples)

  ker_param_prior = kernel_parameter_prior(mcmc_kernel,n_dims)

  ker_params = rand(ker_param_prior,n_samples)
  ker_param_weights = Vector{eltype(initial_samples)}(undef,n_samples)

  ProgressMeter.update!(loop_prog,0)
  while true
    # `state` contains information regarding the previous step in the sequence
    if store_trace
      push!(trace,deepcopy(state))
    end

    if callback(trace)
      @warn "Stopped by callback"
      return store_trace ? trace : state
    end

    mul_logdensity = next_logdensity!(seq,state.seq_state,state.lw,state.ℓ)

    if !isnothing(mul_logdensity)
      add_incremental_weights!(state.lw,seq,state.seq_state)
    end

    lw_norm_constant = logsumexp(state.lw)
    for i in eachindex(state.W)
      state.W[i] = exp(state.lw[i]-lw_norm_constant)
    end

    if isnothing(mul_logdensity)
      if !iszero(lw_norm_constant)
        state.log_evidence += lw_norm_constant + lN
      end
      break
    end

    # Evidence estimate and resample
    ess = 1/sum(abs2,state.W)
    indices = if ess < resampling_α*n_samples
      v = resampler(state.W)
      state.log_evidence += lw_norm_constant + lN
      fill!(state.lw,0.)
      state.resampled = true
      v
    else
      state.resampled = false
      indices_no_resampling
    end

    # Propagate
    target = FullLogDensity(ref_logdensity,mul_logdensity)
    starting_x = [view(state.samples,:,i) for i in indices]
    cov_estimate = estimate_cov(cov_estimator, state.samples,state.W,starting_x)
    chains = stabilized_map(collect(zip(starting_x,ker_params,cov_estimate)),map_func) do (x,p,Σ)
      kernel_state = init_kernel_state(mcmc_kernel,x,p,Σ)
      mcmc_chain(mcmc_kernel,target,x,kernel_state,mcmc_steps+1)
    end

    # Update particle information
    for (i,c) in enumerate(chains)
      state.samples[:,i] .= c.samples[end]
      state.ℓ[i] = c.lps[end].info.mul
    end

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / (mcmc_steps*n_samples)

    # Resample scale following 10.1214/13-BA814
    for j in 1:n_samples
      c = chains[j]
      Σ = cov_estimate[j]
      # # # Rao-Blackwellized estimator of the Expected squared jump distance
      # w = 0.
      # for i in 1:mcmc_steps
      #   δ = c.samples[i+1]-c.samples[i]
      #   w += c.γ[i] * dot(δ,δ)
      # end
      # w /= mcmc_steps
      # state.scale_weights[j] = w
      # Average acceptance probability
      w = 1.
      for i in 1:mcmc_steps
        w *= c.γ[i]
      end

      ker_param_weights[j] = w^(1/mcmc_steps) * invquad(Σ,c.samples[end] - c.samples[1])
    end
    n = sum(ker_param_weights)
    if n == 0
      ker_params = rand(ker_param_prior,n_samples)
    else
      ker_param_weights ./= n
    end
    param_inds = resampler(ker_param_weights)

    for (j,i) in enumerate(param_inds)
      ker_params[j] = rand(kernel_param_perturbative_dist(mcmc_kernel,ker_params[i]))
    end

    ProgressMeter.next!(loop_prog,
                        showvalues=[
                        progress_info(state.seq_state)...,
                        ("Resampled?",state.resampled),
                        ("Log evidence",state.log_evidence),
                        ("Acceptance rate",state.acceptance_rate),
                        ])

  end


  ProgressMeter.finish!(loop_prog)

  if store_trace
    push!(trace,state)
    return trace
  end

  return state
end

function waste_free_smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples;
             mcmc_kernel::AbstractMCMCKernel = _default_sampler(ref_logdensity,mul_logdensity),
             cov_estimator::AbstractCovEstimator = ParticleCov(),
             resampler::AbstractResampler = ResidualResampler(),
             n_starting = max(2,round(Int,0.5cbrt(size(initial_samples,2)))),
             mcmc_steps = 5,
             map_func = map,
             callback=(_) -> false,
             store_trace = true,
             show_progress = true
             )


  n_dims, n_samples = size(initial_samples)
  lN = -log(n_samples)

  if mod(n_samples,n_starting) != 0
    n_starting = div(n_samples,round(Int,n_samples/n_starting))
  end
  chain_length = div(n_samples, n_starting)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9,enabled = show_progress)

  state = SMCState(seq,initial_samples,map_func)
  trace = typeof(state)[]

  indices_no_resampling = collect(1:n_samples)

  ker_param_prior = kernel_parameter_prior(mcmc_kernel,n_dims)

  ker_params = rand(ker_param_prior,n_starting)
  ker_param_weights = Vector{eltype(initial_samples)}(undef,n_starting)

  ProgressMeter.update!(loop_prog,0)
  while true
    # `state` contains information regarding the previous step in the sequence
    if store_trace
      push!(trace,deepcopy(state))
    end

    if callback(trace)
      @warn "Stopped by callback"
      return store_trace ? trace : state
    end

    mul_logdensity = next_logdensity!(seq,state.seq_state,state.lw,state.ℓ)

    if isnothing(mul_logdensity) break end

    add_incremental_weights!(state.lw,seq,state.seq_state)

    lw_norm_constant = logsumexp(state.lw)
    for i in eachindex(state.W)
      state.W[i] = exp(state.lw[i]-lw_norm_constant)
    end

    indices = resampler(state.W,n_starting)
    state.log_evidence += lw_norm_constant + lN
    fill!(state.lw,0.)
    state.resampled = true

    # Propagate
    target = FullLogDensity(ref_logdensity,mul_logdensity)
    starting_x = [view(state.samples,:,i) for i in indices]
    cov_estimate = estimate_cov(cov_estimator, state.samples,state.W,starting_x)
    chains = stabilized_map(collect(zip(starting_x,ker_params,cov_estimate)),map_func) do (x,p,Σ)
      kernel_state = init_kernel_state(mcmc_kernel,x,p,Σ)
      mcmc_chain(mcmc_kernel,target,x,kernel_state,chain_length)
    end

    # Update particle information
    offset = 0
    for c in chains
      for j in 1:chain_length
        i = offset+j
        state.samples[:,i] .= c.samples[j]
        state.ℓ[i] = c.lps[j].info.mul
      end
      offset += chain_length
    end

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / (mcmc_steps*n_samples)

    # Resample scale following 10.1214/13-BA814
    for j in 1:n_starting
      c = chains[j]
      Σ = cov_estimate[j]
      # Rao-Blackwellized estimator of the Expected squared jump distance
      w = 0.
      for i in 1:chain_length-1
        δ = c.samples[i+1]-c.samples[i]
        w += c.γ[i] * invquad(Σ,δ)
      end
      w /= chain_length - 1

      ker_param_weights[j] = w
    end
    n = sum(ker_param_weights)
    if n == 0
      ker_params = rand(ker_param_prior,n_samples)
    else
      ker_param_weights ./= n
    end
    param_inds = resampler(ker_param_weights)

    for (j,i) in enumerate(param_inds)
      ker_params[j] = rand(kernel_param_perturbative_dist(mcmc_kernel,ker_params[i]))
    end

    ProgressMeter.next!(loop_prog,
                        showvalues=[
                        progress_info(state.seq_state)...,
                        ("Resampled?",state.resampled),
                        ("Log evidence",state.log_evidence),
                        ("Acceptance rate",state.acceptance_rate),
                        ])

  end


  ProgressMeter.finish!(loop_prog)

  if store_trace
    push!(trace,state)
    return trace
  end

  return state
end
