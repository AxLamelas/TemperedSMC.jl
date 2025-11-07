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
struct AutoStepMALA <: AbstractMCMCKernel{Val{true}} end

function init_kernel_state(_::AutoStepMALA,x,scale,Σ) 
  return (;init_scale=scale,C=cholesky(Symmetric(Σ)))
end

function _mala_state_log_α(target,x,logp_x,gradlogp_x,z,θ,L)
  y = x + 0.5*θ^2*L*(L'*gradlogp_x) + θ*L*z
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  hxy = 0.5*(x-y-0.25*θ^2*L*(L'*gradlogp_y))'*gradlogp_y
  hyx = 0.5*(y-x-0.25*θ^2*L*(L'*gradlogp_x))'*gradlogp_x
  logα = logp_y - logp_x + hxy - hyx
  y, logp_y, gradlogp_y, logα
end

function auto_mala_step_size(target,x,logp_x,gradlogp_x,z,loga,logb,θ0,L,init_j=0)
  j = init_j
  θ = θ0*2. ^j

  _...,logα = _mala_state_log_α(target,x,logp_x,gradlogp_x,z,θ,L)

  inrange = Int(abs(logα) < logb) - Int(abs(logα) > loga)

  if inrange !=0 
    while true
      j += inrange
      θ = θ0*2. ^j
      _...,logα = _mala_state_log_α(target,x,logp_x,gradlogp_x,z,θ,L)

      if inrange == 1 && abs(logα) >= logb
        return j-1
      end
      if inrange == -1 && abs(logα) <= loga
        return j
      end
    end
  end

  return init_j
end

function (k::AutoStepMALA)(target,x,logp_x,gradlogp_x,state)
  z = randn(length(x))
  loga,logb = abs.(log.(sort(rand(2))))
  
  forward_exp = auto_mala_step_size(target,x,logp_x,gradlogp_x,z,loga,logb,
                                    state.init_scale,state.C.L)

  
  θ = state.init_scale*2. ^forward_exp
  y,logp_y,gradlogp_y,logα = _mala_state_log_α(target,x,logp_x,gradlogp_x,z,θ,state.C.L)

  # Use the forward_exp as the starting point
  reverse_exp = auto_mala_step_size(target,y,logp_y,gradlogp_y,-z,loga,logb,
                                    state.init_scale,state.C.L,forward_exp)


  # The step size selector is a dirac measure so the exponents have to be the
  # same for both directions
  α = (forward_exp == reverse_exp) * min(1,exp(logα))

  N = abs(forward_exp) + abs(reverse_exp)

  if rand() < α
    return y, logp_y, gradlogp_y,true, 1/(N+1), state
  end

  return x, logp_x, gradlogp_y, false, 1/(N+1), state
end


Base.@kwdef @concrete struct AutoStepRWMH <: AbstractMCMCKernel{Val{false}}
  acceptance_range = (0.,1.)
  proposal_dist = Normal()
end


function init_kernel_state(_::AutoStepRWMH,x,scale,Σ) 
  return (;init_scale=scale,C=cholesky(Symmetric(Σ)))
end

function auto_rwmh_step_size(target,x,logp_x,z,Δaux,loga,logb,θ0,L,init_j=0)
  j = init_j
  θ = θ0*2. ^j
  y = x + θ * L * z
  logp_y = LD.logdensity(target,y)
  logα = logp_y - logp_x + Δaux

  inrange = Int(abs(logα) < logb) - Int(abs(logα) > loga)

  if inrange !=0 
    while true
      j += inrange
      θ = θ0*2. ^j
      y = x + θ * L * z
      logp_y = LD.logdensity(target,y)
      logα = logp_y - logp_x + Δaux

      if inrange == 1 && abs(logα) >= logb
        return j-1
      end
      if inrange == -1 && abs(logα) <= loga
        return j
      end
    end
  end

  return init_j
end

function (k::AutoStepRWMH)(target,x,logp_x,state)
  @unpack proposal_dist, acceptance_range = k
  αlb,αub = acceptance_range
  z = randn(length(x))
  loga,logb = abs.(log.(sort((αub-αlb) * rand(2) .+ αlb)))
  # Step size selector
  
  Δaux = sum(logpdf(proposal_dist,-z)) - sum(logpdf(proposal_dist,z))
  
  forward_exp = auto_rwmh_step_size(target,x,logp_x,z,Δaux,loga,logb,
                                    state.init_scale,state.C.L)

  
  θ = state.init_scale*2. ^forward_exp
  y = x + θ * state.C.L * z
  logp_y = LD.logdensity(target,y)

  # Use the forward_exp as the starting point
  reverse_exp = auto_rwmh_step_size(target,y,logp_y,-z,-Δaux,loga,logb,
                                    state.init_scale,state.C.L,forward_exp)


  # Acceptance rate in the joint space
  logα = logp_y - logp_x + Δaux 

  # The step size selector is a dirac measure so the exponents have to be the
  # same for both directions
  α = (forward_exp == reverse_exp) * min(1,exp(logα))

  N = abs(forward_exp) + abs(reverse_exp)

  if rand() < α
    return y, logp_y, true, 1/(N+1), state
  end

  return x, logp_x, false, 1/(N+1), state
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

  σ2_rel = σ2/(sum(abs2,R)/length(x))

  u = randn(length(x))
  y = x + σ2_rel/2*R*(R'*gradlogp_x) + sqrt(σ2_rel)*R*u

  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  
  α = min(1.,exp(logp_y-logp_x +
                  1/2*(x-y-σ2_rel/4*R*(R'*gradlogp_y))'*gradlogp_y -
                  1/2*(y-x-σ2_rel/4*R*(R'*gradlogp_x))'*gradlogp_x 
                  ))

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

function (k::MALA)(target,x,logp_x,gradlogp_x,C::Cholesky)
  u = randn(length(x))
  y = x + 1/2*C.L*(C.L'*gradlogp_x) + C.L*u

  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  
  α = min(1.,exp(logp_y-logp_x +
                  1/2*(x-y-1/4*C.L*(C.L'*gradlogp_y))'*gradlogp_y -
                  1/2*(y-x-1/4*C.L*(C.L'*gradlogp_x))'*gradlogp_x 
                  ))

  if rand() < α
    return y, logp_y, gradlogp_y, true, α, C
  end

  return x, logp_x, gradlogp_x, false, α, C
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
      return y,logps[i+1], true, α,C
    end
  end

  return x, logp_x, false, α, C
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
  m = 10 # Determines the maximum number of log density evaluation in the stepping-out procedure
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




