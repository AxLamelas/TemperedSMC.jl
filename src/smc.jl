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
  trcov_reweight::T
  trcov_mcmc::T
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

  return SMCState(seq_state,deepcopy(initial_samples),W,lw,
                  ℓ,zero(T),1.,zero(T),zero(T),false)
end

function smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples;
             mcmc_kernel::AbstractMCMCKernel = PathDelayedRejection(),
             cov_estimator::AbstractCovEstimator = _default_cov_estimator(size(initial_samples)...),
             resampler::AbstractResampler = ResidualResampler(),
             resampling_α = 0.5,
             mcmc_steps = max(50,2LD.dimension(ref_logdensity)),
             adapt_mcmc_steps=true,
             adapt_stability=0.01,
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

  n_batch_steps = adapt_mcmc_steps ? 1 : mcmc_steps

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
    add_incremental_weights!(state.lw,seq,state.seq_state)

    lw_norm_constant = logsumexp(state.lw)
    for i in eachindex(state.W)
      state.W[i] = exp(state.lw[i]-lw_norm_constant)
    end

    state.trcov_reweight = sum(var(state.samples,FrequencyWeights(state.W),2))

    # Evidence estimate and resample
    ess = 1/sum(abs2,state.W)
    indices = if ess < resampling_α*n_samples || islast(seq,state.seq_state)
      v = resampler(state.W)
      state.log_evidence += lw_norm_constant + lN
      fill!(state.lw,0.)
      fill!(state.W,1/n_samples)
      state.resampled = true
      v
    else
      state.resampled = false
      indices_no_resampling
    end

    # Propagate
    target = FullLogDensity(ref_logdensity,mul_logdensity)
    starting_x = [state.samples[:,i] for i in indices]
    cov_estimate = estimate_cov(cov_estimator, state.samples,state.W,starting_x)
    chains = stabilized_map(collect(zip(starting_x,ker_params,cov_estimate)),map_func) do (x,p,Σ)
      kernel_state = init_kernel_state(mcmc_kernel,x,p,Σ)
      iterate_mcmc(mcmc_kernel,target,x,kernel_state,n_batch_steps)
    end

    n_steps = n_batch_steps
    prev_msjd = 0.
    while adapt_mcmc_steps # Apply kernel until msjd stabilizes
      msjd = mean(zip(starting_x,chains,cov_estimate)) do (x,c,Σ)
        invquad(Σ,c.chain_state.x - x)
      end
      if (msjd-prev_msjd) < adapt_stability*prev_msjd || n_steps >= mcmc_steps break end
      chains = stabilized_map(chains,map_func) do c
        iterate_mcmc(mcmc_kernel,target,c.chain_state,c.kernel_state,n_batch_steps,
                     n_accepts = c.n_accepts, γ = c.γ)
      end
      prev_msjd = msjd
      n_steps += n_batch_steps
    end

    # Update particle information
    for (i,c) in enumerate(chains)
      state.samples[:,i] .= c.chain_state.x
      state.ℓ[i] = c.chain_state.logp.info.mul

      Σ = cov_estimate[i]
      # Update param weights following 10.1214/13-BA814
      ker_param_weights[i] = (c.γ)^(1/n_steps) * invquad(Σ,c.chain_state.x - starting_x[i])
    end

    n = sum(ker_param_weights)
    if n == 0
      ker_params = rand(ker_param_prior,n_samples)
      fill!(ker_param_weights,1/length(ker_param_weights))
    else
      ker_param_weights ./= n
    end
    param_inds = resampler(ker_param_weights)

    for (j,i) in enumerate(param_inds)
      ker_params[j] = rand(kernel_param_perturbative_dist(mcmc_kernel,ker_params[i]))
    end

    state.trcov_mcmc = sum(var(state.samples,FrequencyWeights(state.W),2))

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / (n_steps*n_samples)

    ProgressMeter.next!(loop_prog,
                        showvalues=[
                        progress_info(state.seq_state)...,
                        ("Resampled?",state.resampled),
                        ("Log evidence",state.log_evidence),
                        ("Acceptance rate",state.acceptance_rate),
                        ("Rejuvenation steps",n_steps)
                        ])

    if islast(seq,state.seq_state) break end
  end


  ProgressMeter.finish!(loop_prog)

  if store_trace
    push!(trace,state)
    return trace
  end

  return state
end

function waste_free_smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples;
                        mcmc_kernel::AbstractMCMCKernel = PathDelayedRejection(),
                        cov_estimator::AbstractCovEstimator = _default_cov_estimator(size(initial_samples)...),
                        resampler::AbstractResampler = ResidualResampler(),
                        n_starting = max(2,round(Int,cbrt(size(initial_samples,2)))),
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

    add_incremental_weights!(state.lw,seq,state.seq_state)

    lw_norm_constant = logsumexp(state.lw)
    for i in eachindex(state.W)
      state.W[i] = exp(state.lw[i]-lw_norm_constant)
    end

    state.trcov_reweight = sum(var(state.samples,FrequencyWeights(state.W),2))

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

    state.trcov_mcmc = sum(var(state.samples,FrequencyWeights(state.W),2))

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
        # w += invquad(Σ,δ)
      end
      w /= chain_length - 1

      ker_param_weights[j] = w
    end
    n = sum(ker_param_weights)
    if n == 0
      ker_params = rand(ker_param_prior,n_starting)
      fill!(ker_param_weights,1/length(ker_param_weights))
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

    if islast(seq,state.seq_state)
      fill!(state.W,1/n_samples)
      break
    end
  end


  ProgressMeter.finish!(loop_prog)

  if store_trace
    push!(trace,state)
    return trace
  end

  return state
end
