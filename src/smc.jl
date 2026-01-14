function smc(ref_logdensity,mul_logdensity,initial_samples;
             mcmc_kernel::AbstractMCMCKernel = _default_sampler(ref_logdensity,mul_logdensity),
             cov_estimator::AbstractCovEstimator = ParticleCov(),
             resampler::AbstractResampler = SSPResampler(),
             α = 0.999,
             resampling_α = 0.5,
             mcmc_steps = 1,
             # Reference scale
             ref_cov_scale = 1.,
             # Search scales up to `ϵ` orders of magnitude lower than 
             # the ref scale
             ϵ = 4,
             # Magnitude of the perturbation of the scale estimate
             perturb_scale = 0.015,
             map_func = map,
             maxiters = 1000,
             callback=(_) -> false,
             store_trace = true,
             show_progress = true
             )


  samples = copy(initial_samples)
  n_dims, n_samples = size(samples)
  lN = -log(n_samples)

  loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9,enabled = show_progress)

  ℓ = stabilized_map(
    Base.Fix1(LD.logdensity,mul_logdensity),eachcol(samples),map_func) 

  state = SMCState(
    samples,ℓ,
    ref_cov_scale * 10 .^ (range(-ϵ,ϵ,length=n_samples)),
  )
  trace = typeof(state)[]

  indices_no_resampling = collect(1:n_samples)
  resampled = false

  ProgressMeter.update!(loop_prog,0)
  while state.β < 1 && state.iter < maxiters
    # `state` contains information regarding the previous step in the sequence
    if store_trace
      push!(trace,deepcopy(state))
    end

    if callback(trace)
      @warn "Stopped by callback"
      return store_trace ? trace : state
    end

    # Determines the current distribution in the sequence
    β = _next_β(state,α)

    for i in eachindex(state.lw)
          state.lw[i] += (β-state.β) * state.ℓ[i]
    end

    nw = logsumexp(state.lw)
    for i in eachindex(state.W)
      state.W[i] = exp(state.lw[i]-nw)
    end

    # Evidence estimate and resample
    ess = 1/sum(abs2,state.W)
    indices = if ess < resampling_α*n_samples || isone(β)
      v = resampler(state.W)
      fill!(state.lw,0)
      resampled = true
      state.log_evidence += nw + lN
      v
    else
      resampled = false
      indices_no_resampling
    end

    # Propagate
    starting_x = [view(samples,:,i) for i in indices]
    cov_estimate = estimate_cov(cov_estimator, samples,state.W,starting_x)
    chains = stabilized_map(collect(zip(starting_x,state.scales,cov_estimate)),map_func) do (x,scale,Σ)
      interp_density = TemperedLogDensity(ref_logdensity,mul_logdensity,β,n_dims)
      kernel_state = init_kernel_state(mcmc_kernel,x,scale,Σ)
      mcmc_chain(mcmc_kernel,interp_density,x,kernel_state,mcmc_steps+1)
    end

    # Update the state
    for (i,c) in enumerate(chains)
      state.samples[:,i] .= c.samples[end]
      state.ℓ[i] = c.lps[end].info.mul 
    end

    state.β = β

    # Average acceptance rate of the chains
    state.acceptance_rate = sum(c.n_accepts for c in chains) / (mcmc_steps*n_samples)

    # Resample scale following 10.1214/13-BA814
    for j in 1:n_samples
      c = chains[j]
      Σ = cov_estimate[j]
      # Rao-Blackwellized estimator of the Expected squared jump distance 
      w = 0.
      for i in 1:mcmc_steps
        δ = c.samples[i+1]-c.samples[i]
        w += c.γ[i] * δ'*(Σ\δ)
      end
      w /= mcmc_steps
      state.scale_weights[j] = w
    end
    n = sum(state.scale_weights)
    if n == 0
      # All weights are 0 -> reset weights
      state.scale_weights .= 1 / n_samples
      state.scales ./= 10
    else
      state.scale_weights ./= n
    end
    scale_inds = resampler(state.scale_weights)
    state.scales = state.scales[scale_inds]
    for i in eachindex(state.scales)
      # Instead of just the Mixture model from the paper
      # Do also a mixture with the initial uniform distribution
      # so that if the scale changes abruptly between steps 
      # the distribution of scale parameters is not stuck on the old scale
      if rand() < 0.9 + 0.1*state.β # it is also tempered
        state.scales[i] = exp(log(state.scales[i]) + perturb_scale*randn())
      else
        state.scales[i] = ref_cov_scale * 10 .^ (ϵ*(2*rand()-1))
      end
    end
    state.iter += 1

    ProgressMeter.next!(loop_prog,
                        showvalues=[
                        ("β",state.β),
                        ("Resampled?",resampled),
                        ("Maximum ℓ",maximum(state.ℓ)),
                        ("Log evidence",state.log_evidence),
                        ("Acceptance rate",state.acceptance_rate),
                        ("Median scale",median(state.scales)),
                        ])

  end


  ProgressMeter.finish!(loop_prog)

  if !isone(state.β)
    @warn "Did not reach β=1 in the give limit of iterations"
  end
 
  if store_trace
    push!(trace,state)
    return trace
  end

  return state
end
