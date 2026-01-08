abstract type AbstractMCMCKernel{G <: Val} end

# An mcmc kernel must return:
#  - the next point
#  - the log density of the next point
#  - (the gradient of the logdensity of the next point)
#  - wether the proposal was accepted or not
#  - a factor to weight the expected squared jump distance for parameter
#  adaptation -> usually the acceptance rate for most kernels
#  - the new state for the kernel

function (_::AbstractMCMCKernel{Val{false}})(target,x,logp_x,state) end
function (_::AbstractMCMCKernel{Val{true}})(target,x,logp_x,gradlogp_x,state) end

# Default kernel initialization
function init_kernel_state(_::AbstractMCMCKernel,x,scale,Σ) 
  cholesky(Symmetric(scale*Σ))
end

usesgrad(_::AbstractMCMCKernel{Val{V}}) where {V} = V


# From https://doi.org/10.48550/arXiv.2410.18929
abstract type AbstractAutoStep{G} <: AbstractMCMCKernel{G} end


function init_kernel_state(_::AbstractAutoStep,x,scale,Σ) 
  return (;init_scale=scale,C=cholesky(Symmetric(Σ)))
end

max_iters(::AbstractAutoStep) = 10
factor_base(::AbstractAutoStep) = 4.

# function auto_step_size(type, a, b, θ0, args...)
#   base = factor_base(type)
#
#   θl = zero(θ0)
#   θ = θ0
#
#   info...,logα = involution(type,θ,args...)
#   extra_evals = 0
#   α = min(1.,exp(logα))
#
#   if a < α < b
#     return Normal(θ,0.1*θ), extra_evals
#   end
#
#   # Search for a upper bound
#   while α > a
#     θ *= base
#     info...,logα = involution(type,θ,args...)
#     extra_evals += 1
#     α = min(1.,exp(logα))
#     # @info "Bound search" α θ
#   end
#
#   θu = θ
#
#   for _ in 1:max_iters(type)
#     θ = (θl + θu)/2
#     info...,logα = involution(type,θ,args...)
#     extra_evals += 1
#     α = min(1.,exp(logα))
#     # @info "Bisection" α a b θ θl θu
#     if a < α < b
#       return Normal(θ,0.1*θ), extra_evals
#     end
#
#     if α > b
#       θl = θ 
#     end
#
#     if α < a
#       θu = θ
#     end
#   end
#
#   return Normal(θ,0.05*θ), extra_evals
# end

function auto_step_size(type, a, b, θ0, args...)
  loga = abs(log(a))
  logb = abs(log(b))
  b = factor_base(type)
  θ = θ0

  info...,logα = involution(type,θ,args...)
  extra_evals = 0

  inrange = Int(abs(logα) < logb) - Int(abs(logα) > loga)
  δ = b^inrange
  j = 0

  if iszero(inrange)
    return j, extra_evals
  end
  for _ in 1:max_iters(type)
    θ *= δ
    j += inrange
    info...,logα = involution(type,θ,args...)
    extra_evals += 1

    if inrange == 1 && abs(logα) >= logb
      θ /= δ
      break
    end
    if inrange == -1 && abs(logα) <= loga
      break
    end
  end

  return j, extra_evals
end

function (ker::AbstractAutoStep{Val{true}})(target,x,logp_x,gradlogp_x,state)
  z = randn(length(x))
  a0,b0 = rand(),rand()
  a = min(a0,b0)
  b = max(a0,b0)

  jitter_dist = Normal(0.,0.5)
 
  forward_exp,fevals = auto_step_size(ker,a,b,state.init_scale,
                               target,x,logp_x,gradlogp_x,z,state.C.L)

  forward_jitter = rand(jitter_dist)
  θ = state.init_scale * factor_base(ker)^(forward_exp+forward_jitter)

  y,logp_y,gradlogp_y,w,logα = involution(ker,θ,target,x,logp_x,gradlogp_x,z,state.C.L)

  reverse_exp, revals = auto_step_size(ker,a,b,state.init_scale,
                               target,y,logp_y,gradlogp_y,w,state.C.L)

  reverse_jitter = forward_exp + forward_jitter - reverse_exp
  α = min(1,exp(logα + logpdf(jitter_dist,reverse_jitter) -
                 logpdf(jitter_dist,forward_jitter)))

  N = fevals + revals - 1
  # @info "" α θ logpdf(jitter_dist,reverse_jitter) logpdf(jitter_dist,forward_jitter) logp_y logp_x

  if rand() < α
    return y, logp_y, gradlogp_y,true, α*(1-N/(2max_iters(ker))), state
  end

  return x, logp_x, gradlogp_y, false, α*(1-N/(2max_iters(ker))), state
end

function (ker::AbstractAutoStep{Val{false}})(target,x,logp_x,state)
  z = randn(length(x))
  a0,b0 = rand(),rand()
  a = min(a0,b0)
  b = max(a0,b0)

  jitter_dist = Normal(0.,0.5)
  
  forward_exp,fevals = auto_step_size(ker,a,b,state.init_scale,
                               target,x,logp_x,z,state.C.L)

  forward_jitter = rand(jitter_dist)
  θ = state.init_scale * factor_base(ker)^(forward_exp+forward_jitter)
  
  y,logp_y,w,logα = involution(ker,θ,target,x,logp_x,z,state.C.L)

  reverse_exp,revals = auto_step_size(ker,a,b,state.init_scale,
                               target,y,logp_y,w,state.C.L)

  reverse_jitter = forward_exp + forward_jitter - reverse_exp
  α = min(1,exp(logα + logpdf(jitter_dist,reverse_jitter) -
                 logpdf(jitter_dist,forward_jitter)))

  N = fevals+revals
  # @info "" α θ logpdf(jitter_dist,reverse_jitter) logpdf(jitter_dist,forward_jitter) logp_y logp_x

  if rand() < α
    return y, logp_y, true, α*(1-N/(2max_iters(ker))), state
  end

  return x, logp_x, false, α*(1-N/(2max_iters(ker))), state
end

struct AutoStepMALA <: AbstractAutoStep{Val{true}} end

function involution(::AutoStepMALA,θ,target,x,logp_x,gradlogp_x,z,L)
  # Leapfrog integrator
  zhalf  = z + θ/2*(L * gradlogp_x)
  y = L * zhalf
  y .= x .+ θ .* y
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  w = -(zhalf + θ/2 * L * gradlogp_y)

  logα = logp_y - 0.5 * sum(abs2,w) -
    logp_x + 0.5 * sum(abs2,z)

  return y,logp_y,gradlogp_y, w, logα
end

struct AutoStepRWMH <: AbstractAutoStep{Val{false}} end

function involution(::AutoStepRWMH,θ,target,x,logp_x,z,L)
  y = x + θ*L*z
  logp_y = LD.logdensity(target,y)
  logα = logp_y - logp_x 
  y, logp_y, -z,logα
end

Base.@kwdef @concrete struct FisherMALA <: AbstractMCMCKernel{Val{true}}
  λ = 10.
  ρ = 0.015
  αstar = 0.574
end

function init_kernel_state(_::FisherMALA,x,scale,Σ)
  (;iter=1,σ2 = scale*tr(Σ),R = sqrt(Σ))
end

function (k::FisherMALA)(target,x,logp_x,gradlogp_x,state)
  @unpack iter,σ2,R = state
  @unpack λ,ρ,αstar = k

  ϵ = σ2/(sum(abs2,R)/length(x))

  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵ/2*R*gradlogp_x
  y = x + ϵ * R*velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = velocity_middle + ϵ/2*R*gradlogp_y

  α = min(1.,exp(logp_y - 0.5 * sum(abs2,velocity_end) - 
                  logp_x + 0.5 * sum(abs2,velocity)))

  s = sqrt(α)*(gradlogp_y-gradlogp_x)

  if iter == 1
    ϕ = R'*s
    n = λ + ϕ'*ϕ
    r = 1/(1+sqrt(λ/n))
    nextR = 1/sqrt(λ) * (R - r/n * (R*ϕ)*ϕ')
  else
    ϕ = R'*s
    n = 1 + ϕ'*ϕ
    r = 1/(1+sqrt(1/n))
    nextR = R - r/n * (R*ϕ)*ϕ' 
  end

  nextσ2 = exp(log(σ2) + ρ*(α-αstar))

  next_state = (;iter = iter+1,σ2 = nextσ2, R = nextR)

  if rand() < α
    return y, logp_y, gradlogp_y, true, α, next_state
  end

  return x, logp_x, gradlogp_x, false, α, next_state
end

struct MALA <: AbstractMCMCKernel{Val{true}} end

function init_kernel_state(_::MALA,x,scale,Σ) 
  (;ϵ=scale,Minv = cholesky(Symmetric(Σ)))
end

function (k::MALA)(target,x,logp_x,gradlogp_x,state)
  @unpack ϵ, Minv = state
  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵ/2*Minv.L*gradlogp_x
  y = x + ϵ * Minv.L*velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = velocity_middle + ϵ/2*Minv.L*gradlogp_y

  α = min(1.,exp(logp_y - 0.5 * sum(abs2,velocity_end) - 
                  logp_x + 0.5 * sum(abs2,velocity)))

  if rand() < α
    return y, logp_y, gradlogp_y, true, α, state
  end

  return x, logp_x, gradlogp_x, false, α, state
end

Base.@kwdef @concrete struct PathDelayedRejection <: AbstractMCMCKernel{Val{false}}
  proposal_dist = Normal()
  n_stages = 4
  factor = 0.25
end

_scaled_logpdf(dist,u,scale) = sum(logpdf(dist,u/scale)) - length(u)*log(scale)

function _scaled_logΔ(proposal_dist,logps,us,factor,seq)
  stage = length(seq)-1
  if stage == 1
    return logps[seq[1]] + sum(logpdf(proposal_dist,us[seq[2]]-us[seq[1]]))
  end

  q = _scaled_logpdf(proposal_dist,us[seq[end]]-us[seq[1]],factor^(stage-1))

  next_seq = seq[1:end-1]
  a = _scaled_logΔ(proposal_dist,logps,us,factor,next_seq)

  next_seq[1], next_seq[end] = next_seq[end], next_seq[1]

  b = _scaled_logΔ(proposal_dist,logps,us,factor,next_seq)

  return  q + logsubexp(a,b)
end

function _logα(proposal_dist,logps,us,factor,stage)
  seq = collect(1:stage+1)
  forward_Δ = _scaled_logΔ(proposal_dist,logps,us,factor,seq)
  seq[1],seq[end] = seq[end],seq[1]
  backward_Δ = _scaled_logΔ(proposal_dist,logps,us,factor,seq)

  if isinf(backward_Δ) && isinf(forward_Δ)
    return  forward_Δ
  end
  return backward_Δ - forward_Δ
end

function (k::PathDelayedRejection)(target,x,logp_x,C::Cholesky)
  @unpack proposal_dist,factor,n_stages = k
  n = length(x)

  us = Vector{Vector{eltype(x)}}(undef,n_stages+1)
  logps = Vector{typeof(logp_x)}(undef,n_stages+1)
  us[1] = zeros(n)
  logps[1] = logp_x
  
  local α
  for i in 1:n_stages
    us[i+1] = factor^(i-1) * rand(proposal_dist,n)
    y = x + C.L*us[i+1]
    logps[i+1] =  LD.logdensity(target,y)

    α = min(1.,exp(_logα(
      proposal_dist,
      logps,us,
      factor,i
    )))

    if rand() < α
      return y,logps[i+1], true, α*(1-(i-1)/n_stages),C
    end
  end

  return x, logp_x, false, α/n_stages, C
end

Base.@kwdef @concrete struct RWMH <: AbstractMCMCKernel{Val{false}}
  proposal_dist = Normal()
end

function (k::RWMH)(target,x,logp_x,C::Cholesky)
  @unpack proposal_dist = k
  u = rand(proposal_dist,length(x))
  y = x + C.L*u
  logp_y = LD.logdensity(target,y)

  α = min(1.,exp.(logp_y + sum(logpdf(proposal_dist,-u)) - logp_x - sum(logpdf(proposal_dist,u))))
  if rand() < α
    return y, logp_y, true, α, C
  end

  return x, logp_x, false, α, C
end

Base.@kwdef @concrete struct SliceSampling <: AbstractMCMCKernel{Val{false}}
  m = 20 # Determines the maximum number of log density evaluation in the stepping-out procedure
end

function init_kernel_state(_::SliceSampling,x,scale,Σ)
  (;w=scale/2,Σ = cholesky(Σ))
end

function (k::SliceSampling)(target,x,logp_x,state)
  @unpack w,Σ = state
  @unpack m = k

  counter = 0

  # Slice direction -> From a zero mean multivariate normal 
  d = Σ.L * randn(length(x))

  # Defining the level
  z = logp_x  - rand(Exponential())

  # Fiding the interval that contains the slice using the stepping-out procedure
  u = rand()
  # Interval o length w randomly positioned around x
  L = -w*u
  R = L + w
  v = rand()
  J = floor(Int,m*v) # Maximum number of changes to L
  K = m-1-J # Maximum number of changes to R
  while J > 0 && z < LD.logdensity(target,x + L*d)
    counter += 1
    L -= w
    J -= 1
  end

  while K > 0 && z < LD.logdensity(target,x + R*d)
    counter += 1
    R += w
    K -= 1
  end

  # Sampling uniformly from the interval until a point in the slice if found
  # using the shrinking procedure
  
  while true
    counter += 1
    u = rand()
    y = x + (L + u*(R-L))*d
    logp_y = LD.logdensity(target,y) 
    if z < logp_y
      return y, logp_y, true, 1/counter, state
    end
    if u < 0
      L = u
    else
      R = u
    end
  end
end
