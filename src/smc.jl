mutable struct SMCState{T,S<:AbstractSequenceState{T},C}
  # Log Density sequence
  seq_state::S
  # Particles
  samples::Matrix{T}
  # Normalized Weights
  W::Vector{T}
  # unormalized log weights
  lw::Vector{T}
  # MCMCM chain states
  states::Vector{C}
  log_evidence::T
  acceptance_rate::Float64
  trcov_reweight::T
  trcov_mcmc::T
  resampled::Bool
  t_reweight::Float64   # Elapsed time for reweight + ESS + resampling decision
  t_metric::Float64     # Elapsed time for metric/covariance estimation
  t_mcmc::Float64       # Elapsed time for MCMC rejuvenation phase
  t_adapt::Float64      # Elapsed time for kernel parameter adaptation
  n_steps::Int          # Number of MCMC steps run this iteration
end

function SMCState(seq::AbstractDistributionSequence,ref_logdensity,initial_samples,mcmc_kernel,map_func)
  mul_logdensity = initial_logdensity(seq)
  target = FullLogDensity(ref_logdensity,mul_logdensity)
  states = stabilized_map(eachcol(initial_samples),map_func) do s
    init_chain_state(mcmc_kernel,target,s)
  end

  seq_state = init_sequence_state(seq,(s.logp.info.mul for s in states))
  T = eltype(seq_state)
  lw = [convert(T,s.logp.info.mul) for s in states] # Could possibly be a metanumber
  lw_norm_constant = logsumexp(lw)
  W = exp.(lw .- lw_norm_constant)

  n_samples = length(states)

  return SMCState(seq_state,deepcopy(initial_samples),W,lw,
                  states,zero(T),1.,zero(T),zero(T),false,
                  0.,0.,0.,0.,0)
end

_copy_state(state::SMCState) = SMCState(
  deepcopy(state.seq_state), copy(state.samples), copy(state.W),
  copy(state.lw), copy(state.states), state.log_evidence,
  state.acceptance_rate, state.trcov_reweight, state.trcov_mcmc, state.resampled,
  state.t_reweight, state.t_metric, state.t_mcmc, state.t_adapt, state.n_steps)

function smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples::AbstractMatrix;
			 mcmc_kernel::AbstractMCMCKernel = RWMH(),
			 metric_estimator::AbstractMetric = _default_metric_estimator(size(initial_samples)...),
			 ker_parameters::AbstractKernelParameters = ScaleAdaptation(reverse(size(initial_samples))...),
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


	# Normalize kernel to population-based
	pop_kernel = if mcmc_kernel isa AbstractPopulationKernel
		mcmc_kernel  # Already population-based
	else
		PopulationKernel(mcmc_kernel; map_func=map_func)  # Wrap individual kernel
	end

	n_dims, n_samples = size(initial_samples)
	lN = -log(n_samples)

	loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9,enabled = show_progress)

	state = SMCState(seq,ref_logdensity,initial_samples,pop_kernel,map_func)
	trace = typeof(state)[]

	indices_no_resampling = collect(1:n_samples)

	ProgressMeter.update!(loop_prog,0)
	while true
		# `state` contains information regarding the previous step in the sequence
		if store_trace
			push!(trace, _copy_state(state))
		end

		if callback(trace)
			@warn "Stopped by callback"
			return store_trace ? trace : state
		end

		state.t_reweight = @elapsed begin
			mul_logdensity = next_logdensity!(seq,state.seq_state,state.lw,(s.logp.info.mul for s in state.states))
			add_incremental_weights!(state.lw,seq,state.seq_state)

			lw_norm_constant = logsumexp(state.lw)
			for i in eachindex(state.W)
				state.W[i] = exp(state.lw[i]-lw_norm_constant)
			end

			Σg = PDMat(ensure_posdef(cov(
				state.samples,FrequencyWeights(state.W),2
			)))

			state.trcov_reweight = tr(Σg)
		end

		# Evidence estimate and resample
		ess = 1/sum(abs2,state.W)
		indices = if ess < resampling_α*n_samples || islast(seq,state.seq_state)
			v = resampler(state.W)
			state.log_evidence += lw_norm_constant + lN
			fill!(state.lw,0.)
			# Weight cannot be reset here because the metric estimate is based on the weighted particles
			# Moreover, it does not affect anything else because W is calculated from lw at each interation
			# and not accumulated
			# fill!(state.W,1/n_samples)
			state.resampled = true
			v
		else
			state.resampled = false
			indices_no_resampling
		end


		# Propagate
		target = FullLogDensity(ref_logdensity,mul_logdensity)
		starting_x = [state.samples[:,i] for i in indices]
		state.t_metric = @elapsed begin
			metric_estimate = if metric_estimator isa ParticleCov
				Fill(Σg, length(starting_x))
			else
				estimate_metric(metric_estimator, state.samples,state.W,state.states,starting_x,progress_fraction(state.seq_state))
			end
		end
		state.t_mcmc = @elapsed begin
			chains,n_steps = if adapt_mcmc_steps
				# Initialize population state: vectors of chain states and kernel states
				chain_state = init_chain_state(pop_kernel, target, starting_x)
				ker_state   = init_kernel_state(pop_kernel, starting_x, get_parameters(ker_parameters), metric_estimate)

				# Store population state history and acceptance metrics
				history   = [chain_state]
				γ_history = Vector{Vector{Float64}}()
				n_accepts = zeros(Int, length(starting_x))
				n_steps = 0
				prev_msjd = 0.

				# Adaptive step loop: evolve population until MSJD stabilizes
				while true
					# Apply kernel to entire population at once
					chain_state, acc, γi, ker_state = pop_kernel(target, chain_state, ker_state)

					push!(history, chain_state)
					push!(γ_history, γi)
					n_accepts .+= acc
					n_steps += 1

					# Compute mean squared jump distance
					msjd = mean(zip(chain_state, history[1])) do (cs_now, cs_init)
						invquad(Σg, cs_now.x - cs_init.x)
					end

					if abs(msjd - prev_msjd) < adapt_stability * prev_msjd || n_steps >= mcmc_steps
						break
					end
					prev_msjd = msjd
				end

				# Convert population history to per-particle chains
				chains = per_particle_chains((;n_accepts, states=history, kernel_state=ker_state, γ=γ_history))

				chains, n_steps
			else
				# Non-adaptive: fixed steps
				ker_state = init_kernel_state(pop_kernel, starting_x, get_parameters(ker_parameters), metric_estimate)
				result = mcmc_chain(pop_kernel, target, starting_x, ker_state, mcmc_steps+1)
				chains = per_particle_chains(result)

				chains, mcmc_steps
			end
		end
		state.n_steps = n_steps

		# Update particle information
		for (i,c) in enumerate(chains)
			state.samples[:,i] .= c.states[end].x
			state.states[i] = c.states[end]
		end

		Σg = PDMat(ensure_posdef(cov(
			state.samples,FrequencyWeights(state.W),2
		)))

		state.t_adapt = @elapsed begin
			update_parameters!(ker_parameters,chains,Σg)
		end

		state.trcov_mcmc = tr(Σg)

		# Average acceptance rate of the chains
		state.acceptance_rate = sum(c.n_accepts for c in chains) / (n_steps*n_samples)

		ProgressMeter.next!(loop_prog,
					  showvalues=[
					  progress_info(state.seq_state)...,
					  ("Resampled?",state.resampled),
					  ("Log evidence",state.log_evidence),
					  ("Acceptance rate",state.acceptance_rate),
					  ("Rejuvenation steps",n_steps),
					  ("t_reweight",state.t_reweight),
					  ("t_metric",state.t_metric),
					  ("t_mcmc",state.t_mcmc),
					  ("t_adapt",state.t_adapt)
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

function waste_free_smc(seq::AbstractDistributionSequence,ref_logdensity,initial_samples::AbstractMatrix;
						mcmc_kernel::AbstractMCMCKernel = RWMH(),
						metric_estimator::AbstractMetric = _default_metric_estimator(size(initial_samples)...),
						resampler::AbstractResampler = ResidualResampler(),
						n_starting = max(2,round(Int,cbrt(size(initial_samples,2)))),
						ker_parameters::Union{AbstractKernelParameters,Nothing} = nothing,
						map_func = map,
						callback=(_) -> false,
						store_trace = true,
						show_progress = true
						)

	# Normalize kernel to population-based
	pop_kernel = if mcmc_kernel isa AbstractPopulationKernel
		mcmc_kernel  # Already population-based
	else
		PopulationKernel(mcmc_kernel; map_func=map_func)  # Wrap individual kernel
	end

	n_dims, n_samples = size(initial_samples)
	lN = -log(n_samples)

	if mod(n_samples,n_starting) != 0
		n_starting = div(n_samples,round(Int,n_samples/n_starting))
	end
	chain_length = div(n_samples, n_starting)

	if ker_parameters === nothing
		ker_parameters = ScaleAdaptation(n_starting,LD.dimension(ref_logdensity))
	end

	loop_prog = ProgressUnknown(desc="Tempering:",showspeed=true,dt=1e-9,enabled = show_progress)

	state = SMCState(seq,ref_logdensity,initial_samples,pop_kernel,map_func)
	trace = typeof(state)[]

	indices_no_resampling = collect(1:n_samples)

	ProgressMeter.update!(loop_prog,0)
	while true
		# `state` contains information regarding the previous step in the sequence
		if store_trace
			push!(trace, _copy_state(state))
		end

		if callback(trace)
			@warn "Stopped by callback"
			return store_trace ? trace : state
		end

		state.t_reweight = @elapsed begin
			mul_logdensity = next_logdensity!(seq,state.seq_state,state.lw,(s.logp.info.mul for s in state.states))

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
		end

		# Propagate
		target = FullLogDensity(ref_logdensity,mul_logdensity)
		starting_x = [state.samples[:,i] for i in indices]
		state.t_metric = @elapsed begin
			metric_estimate = estimate_metric(metric_estimator, state.samples,state.W,state.states,starting_x,progress_fraction(state.seq_state))
		end

		local chains

		state.t_mcmc = @elapsed begin  # Captures total time across retries
			for _ in 1:10 # TODO: generalize this
				ker_state = init_kernel_state(pop_kernel, starting_x, get_parameters(ker_parameters), metric_estimate)
				result = mcmc_chain(pop_kernel, target, starting_x, ker_state, chain_length)
				chains = per_particle_chains(result)
				# Average acceptance rate of the chains
				state.acceptance_rate = sum(c.n_accepts for c in chains) / ((chain_length-1)*n_starting)

				if !iszero(state.acceptance_rate)
					break
				end
				improve_acceptance!(ker_parameters)
			end
		end
		state.n_steps = chain_length - 1

		# Update particle information
		offset = 0
		for c in chains
			for j in 1:chain_length
				i = offset+j
				state.samples[:,i] .= c.states[j].x
				state.states[i] = c.states[j]
			end
			offset += chain_length
		end

		state.trcov_mcmc = sum(var(state.samples,FrequencyWeights(state.W),2))

		Σg = PDMat(ensure_posdef(cov(
			state.samples,FrequencyWeights(state.W),2
		)))

		state.t_adapt = @elapsed begin
			update_parameters!(ker_parameters,chains,Σg)
		end

		ProgressMeter.next!(loop_prog,
					  showvalues=[
					  progress_info(state.seq_state)...,
					  ("Resampled?",state.resampled),
					  ("Log evidence",state.log_evidence),
					  ("Acceptance rate",state.acceptance_rate),
					  ("t_reweight",state.t_reweight),
					  ("t_metric",state.t_metric),
					  ("t_mcmc",state.t_mcmc),
					  ("t_adapt",state.t_adapt)
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
