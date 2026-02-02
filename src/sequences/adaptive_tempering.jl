struct TemperedState{T} <: AbstractSequenceState{T}
  # Mul log density
  ℓ::Vector{T}
  # Temperature
  β::Vector{Float64}
end

function TemperedState(log_density)
  ℓ = [v.info.logdensity for v in log_density]
  return TemperedState(ℓ,[0.])
end

Base.eltype(::TemperedState{T}) where T = T

function progress_info(state::TemperedState)
  [("β",last(state.β)),("Maximum ℓ",maximum(state.ℓ))]
end

struct TemperedLogDensity{D}
  den::D
  β::Float64
  dim::Int
end

function TemperedLogDensity(logden,β;dim=LD.dimension(logden))
  return TemperedLogDensity(logden,β,dim)
end

LD.dimension(ℓ::TemperedLogDensity) = ℓ.dim
LD.capabilities(::Type{<:TemperedLogDensity}) = LD.LogDensityOrder{1}()

function LD.logdensity(ℓ::TemperedLogDensity,θ)
  den = LD.logdensity(ℓ.den,θ)
  MetaNumber(ℓ.β * den ,(;logdensity=den))
end

function LD.logdensity_and_gradient(ℓ::TemperedLogDensity,θ)
  den,dengrad = LD.logdensity_and_gradient(ℓ.den,θ)
  MetaNumber(ℓ.β * den,(;logdensity=den,logdensity_grad = dengrad)), ℓ.β * dengrad
end

"""
    _next_β(state::SMCstate, metric_target)

Compute the next value for `β` and the nominal weights `w` using bisection.
Uses the conditional effective sample size (https://www.jstor.org/stable/44861887) as a metric.
"""
function _next_β(state::TemperedState,lw,metric_target)
  β = state.β[end]
  low = β
  high = 2one(β)

  local x # Declare variables so they are visible outside the loop
  
  lΔw = similar(lw)
  ϵ = sqrt(eps(zero(β)))
  nw = logsumexp(lw)
  
  while (high - low) / ((high + low) / 2) > 1e-12 && high > ϵ
    x = (high + low) / 2
    lΔw .= (x - β) .* state.ℓ
    cess = exp(2 * logsumexp(lw[i]-nw + lΔw[i] for i in eachindex(lΔw)) -
               logsumexp(lw[i]-nw + 2*lΔw[i] for i in eachindex(lΔw)))

    if cess == metric_target
      break
    end
  
    if cess < metric_target
      high = x # Reduce highstract
    else
      low = x # Increase low
    end
  end

  return min(one(x), x)
end

struct AdaptiveTempering{D} <: AbstractDistributionSequence
  ℓ::D
  α::Float64
end

function AdaptiveTempering(log_density; α = 0.8)
  return AdaptiveTempering(log_density,α)
end

function initial_logdensity(seq::AdaptiveTempering)
  return TemperedLogDensity(seq.ℓ,0.)
end

function next_logdensity!(seq::AdaptiveTempering,state::TemperedState,lw,prev_logdenisity)
  if isone(last(state.β))
    return nothing
  end
  for i in eachindex(state.ℓ,prev_logdenisity)
    state.ℓ[i] = prev_logdenisity[i].info.logdensity
  end
  β = _next_β(state,lw,seq.α)
  push!(state.β,β)
  return TemperedLogDensity(seq.ℓ,β)
end

function update_state!(seq::AdaptiveTempering,state::TemperedState,logdensity)
end

function add_incremental_weights!(lw::AbstractVector,seq::AdaptiveTempering,state::TemperedState)
  Δβ = state.β[end] - state.β[end-1]
  for i in eachindex(lw)
    lw[i] += Δβ * state.ℓ[i]
  end
end

# TODO: define a more general interface that allows the calculation of any γ_i / γ_j instead on only γ_i / γ_i-1
# which would be usefull to allow Persistent Sampling
