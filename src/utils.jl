mutable struct SMCState{T}
  iter::Int
  samples::Matrix{T}
  # Normalized Weights
  W::Vector{T}
  # unormalized log weights
  lw::Vector{T}
  # Target log density
  ℓ::Vector{T}
  # Proposal scale
  scales::Vector{T}
  # Proposal scale weights
  scale_weights::Vector{T}
  # Temperature
  β::Float64
  acceptance_rate::Float64
  # Log evidence estimate
  log_evidence::T
end

function SMCState(samples,ℓ,scales,scale_weights)
  @assert length(ℓ) == size(samples,2)
  T = promote_type(eltype(samples),eltype(ℓ),eltype(scales),eltype(scale_weights))
  return SMCState{T}(0,samples,fill(1/length(ℓ),length(ℓ)),zeros(T,length(ℓ)),ℓ,scales,scale_weights,0.,0.,zero(T))
end


"""
  _next_β(state::SMCstate, metric_target)

Compute the next value for `β` and the nominal weights `w` using bisection.
Uses the conditional effective sample size (https://www.jstor.org/stable/44861887) as a metric.
"""
function _next_β(state::SMCState,metric_target)
  low = state.β
  high = 2one(state.β)

  local x # Declare variables so they are visible outside the loop
  
  lΔw = similar(state.ℓ)
  ϵ = sqrt(eps(zero(state.β)))
  nw = logsumexp(state.lw)
  
  while (high - low) / ((high + low) / 2) > 1e-12 && high > ϵ
    x = (high + low) / 2
    lΔw .= (x - state.β) .* state.ℓ
    cess = exp(2 * logsumexp(state.lw[i]-nw + lΔw[i] for i in eachindex(lΔw)) -
               logsumexp(state.lw[i]-nw + 2*lΔw[i] for i in eachindex(lΔw)))

    if cess == metric_target
      break
    end
  
    if cess < metric_target
      high = x # Reduce high
    else
      low = x # Increase low
    end
  end

  return min(one(x), x)
end

function divisors(n)

    d = Int64[1]

    for (p,e) in factor(n)
        t = Int64[]
        r = 1

        for i in 1:e
            r *= p
            for u in d
                push!(t, u*r)
            end
        end

        append!(d, t)
    end

    return sort(d)
end

function _guess_n_starting(n_samples)
  estimate = 2log10(n_samples)^2
  best = 1
  diff = Inf
  for d in divisors(n_samples)
    if abs(d-estimate) < diff
      best = d
      diff = abs(d-estimate)
    end
  end

  return best
end

norm2(v::AbstractVector) = dot(v,v)

function _default_sampler(ref_logdensity,mul_logdensity)
  lc = _lowest_capability(ref_logdensity,mul_logdensity)
  if lc isa LD.LogDensityOrder{0}()
    return AutoStepRWMH()
  end
  return AutoStepMALA()
end

"""
  stabilized_map(f,x,map_func)

  Uses the `Base.map` infrastructure to infer the return type of the map, using 
  a type assertion to enforce it.
"""
function stabilized_map(f,x,map_func)
  T = only(Base.return_types(f,(typeof(first(x)),)))
  return map_func(f,x)::Vector{T}
end

