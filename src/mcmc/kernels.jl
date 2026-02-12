struct IteratedKernel{G,K<:AbstractMCMCKernel{G}} <: AbstractMCMCKernel{G}
  kernel::K
  n_steps::Int
end

init_kernel_state(k::IteratedKernel,x,scale,Σ) = init_kernel_state(k.kernel,x,scale,Σ)

function (k::IteratedKernel)(target,chain_state::AbstractChainState,state)
  γ = 1.
  accepted = 0
  for i in 1:k.n_steps
    chain_state,acceptedi,γi,state = k.kernel(target,chain_state,state)

    γ *= γi
    accepted += acceptedi
  end

  return chain_state, accepted, γ ^ (1/k.n_steps), state
end


# struct Gibbs{G,K<: Tuple} <: AbstractMCMCKernel{G}
#   ind_blocks::Vector{Vector{Int}}
#   kernels::K
# end
#
# function Gibbs(ind_blocks,kernels)
#   @assert length(ind_blocks) == length(kernels)
#   grad = any(usesgrad.(kernels))
#   k = (kernels...,)
#   return Gibbs{Val{grad},typeof(k)}(ind_blocks,k)
# end
#
#
# function init_kernel_state(g::Gibbs,x,scale,Σ)
#   [init_kernel_state(k,x[inds],scale,Σ[inds,inds]) for (k,inds) in zip(g.kernels,g.ind_blocks)]
# end
#
#
# function (g::Gibbs{Val{false}})(target,x,logp_x,state)
#   order = randperm(length(g.ind_blocks))
#   new_state = copy(state)
#   y = copy(x)
#   logp_y = logp_x
#   γ = 1.
#   accepted = false
#   for i in order
#     ctarget = condition(target,y,vcat((g.ind_blocks[j] for j in eachindex(g.ind_blocks) if j != i)...))
#     z, logp_yi, acceptedi, γi, statei = g.kernels[i](ctarget,view(y,g.ind_blocks[i]),logp_y,state[i])
#     if acceptedi
#       y[g.ind_blocks[i]] .= z
#       logp_y = logp_yi
#     end
#     new_state[i] = statei
#     accepted |= acceptedi
#     γ *= γi
#   end
#   # Geometric mean of the individual γ
#   return y, logp_y, accepted, γ ^ (1/length(order)), new_state
# end
#
#
# # TODO: Implement Gibbs to allow some or all of the kernels to use gradients
# # To do it I would have to implement a BlockLogDensity which is basically a vector
# # of LogDensityies for each bloch and some of the them can be differentiated
# # The state for such a logdensity would have to be a tuple of vectors so that each block
# # can have a different type, i.e., for a combination of discrete and continuous variables
# # It has to have a task local variable of the contitioning values
# # When it is evaluated directly it get the log value and a Tuple of gradients for the components
# # that are differentiable
# function (g::Gibbs{Val{true}})(target,x,logp_x,gradlogp_x,state)
#   throw(error("To be implemented!"))
# end


# From https://doi.org/10.48550/arXiv.2410.18929
abstract type AbstractAutoStep{G} <: AbstractMCMCKernel{G} end

function init_kernel_state(_::AbstractAutoStep,x,scale,Σ)
  return (;init_step=sqrt(scale),Σ=Σ)
end

max_iters(::AbstractAutoStep) = 50
factor_base(::AbstractAutoStep) = 2.

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

function (ker::AbstractAutoStep{Val{true}})(target,chain_state::GradientChainState,state)
  x,logp_x,gradlogp_x = chain_state.x,chain_state.logp, chain_state.gradlogp

  z = randn(length(x))
  a0,b0 = rand(),rand()
  a = min(a0,b0)
  b = max(a0,b0)

  jitter_dist = Normal(0.,0.5)
 
  forward_exp,fevals = auto_step_size(ker,a,b,state.init_step,
                               target,x,logp_x,gradlogp_x,z,state.Σ)

  forward_jitter = rand(jitter_dist)
  θ = state.init_step * factor_base(ker)^(forward_exp+forward_jitter)

  y,logp_y,gradlogp_y,w,logα = involution(ker,θ,target,x,logp_x,gradlogp_x,z,state.Σ)

  reverse_exp, revals = auto_step_size(ker,a,b,state.init_step,
                               target,y,logp_y,gradlogp_y,w,state.Σ)

  reverse_jitter = forward_exp + forward_jitter - reverse_exp
  α = min(1,exp(logα + logpdf(jitter_dist,reverse_jitter) -
                 logpdf(jitter_dist,forward_jitter)))

  N = fevals + revals - 1
  # @info "" α θ logpdf(jitter_dist,reverse_jitter) logpdf(jitter_dist,forward_jitter) logp_y logp_x

  if rand() < α
    return GradientChainState(y, logp_y, gradlogp_y),true, α*(1-N/(2max_iters(ker))), state
  end

  return GradientChainState(x, logp_x, gradlogp_y), false, α*(1-N/(2max_iters(ker))), state
end

function (ker::AbstractAutoStep{Val{false}})(target,chain_state::ChainState,state)
  x,logp_x= chain_state.x,chain_state.logp

  z = randn(length(x))
  a0,b0 = rand(),rand()
  a = min(a0,b0)
  b = max(a0,b0)

  jitter_dist = Normal(0.,0.5)

  forward_exp,fevals = auto_step_size(ker,a,b,state.init_step,
                               target,x,logp_x,z,state.Σ)

  forward_jitter = rand(jitter_dist)
  θ = state.init_step * factor_base(ker)^(forward_exp+forward_jitter)
  
  y,logp_y,w,logα = involution(ker,θ,target,x,logp_x,z,state.Σ)

  reverse_exp,revals = auto_step_size(ker,a,b,state.init_step,
                               target,y,logp_y,w,state.Σ)

  reverse_jitter = forward_exp + forward_jitter - reverse_exp
  α = min(1,exp(logα + logpdf(jitter_dist,reverse_jitter) -
                 logpdf(jitter_dist,forward_jitter)))

  N = fevals+revals
  # @info "" α θ logpdf(jitter_dist,reverse_jitter) logpdf(jitter_dist,forward_jitter) logp_y logp_x

  if rand() < α
    return ChainState(y, logp_y), true, α*(1-N/(2max_iters(ker))), state
  end

  return ChainState(x, logp_x), false, α*(1-N/(2max_iters(ker))), state
end

struct AutoStepMALA <: AbstractAutoStep{Val{true}} end

function involution(::AutoStepMALA,θ,target,x,logp_x,gradlogp_x,z,Σ)
  L = Σ.chol.L
  # Leapfrog integrator
  zhalf  = z + θ/2*(L' * gradlogp_x)
  y = L * zhalf
  y .= x .+ θ .* y
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  w = -(zhalf + θ/2 * (L' * gradlogp_y))

  logα = if isinf(logp_y)
    typemin(logp_x)
  else
    logp_y - 0.5 * sum(abs2,w) -
    logp_x + 0.5 * sum(abs2,z)
  end

  return y,logp_y,gradlogp_y, w, logα
end

struct AutoStepRWMH <: AbstractAutoStep{Val{false}} end

function involution(::AutoStepRWMH,θ,target,x,logp_x,z,Σ)
  y = x + θ*unwhiten(L,z)
  logp_y = LD.logdensity(target,y)
  logα = logp_y - logp_x
  y, logp_y, -z,logα
end

Base.@kwdef @concrete struct FisherMALA <: AbstractMCMCKernel{Val{true}}
  λ = 10.
  αstar = 0.574
end

function init_kernel_state(_::FisherMALA,x,scale,Σ)
  (;iter=1,σ2 = scale,R = Matrix(Σ.chol.L), s=0,prev_err=0.)
end

function (k::FisherMALA)(target,chain_state::GradientChainState,state)
  @unpack iter,σ2,R = state
  @unpack λ,αstar = k

  x,logp_x,gradlogp_x = chain_state.x, chain_state.logp, chain_state.gradlogp

  ϵ = sqrt(σ2/(sum(abs2,R)/length(x)))

  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵ/2*R'*gradlogp_x
  y = x + ϵ * R*velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = -velocity_middle - ϵ/2*R'*gradlogp_y

  α = if isinf(logp_y)
    zero(logp_x)
  else
    min(1.,exp(logp_y - 0.5 * sum(abs2,velocity_end) -
               logp_x + 0.5 * sum(abs2,velocity)))
  end


  s_R = sqrt(α)*(gradlogp_y-gradlogp_x)

  nextR = if iter == 1
    ϕ = R'*s_R
    n = λ + ϕ'*ϕ
    r = 1/(1+sqrt(λ/n))
    1/sqrt(λ) * (R - r/n * (R*ϕ)*ϕ')
  else
    ϕ = R'*s_R
    n = 1 + ϕ'*ϕ
    r = 1/(1+sqrt(1/n))
    R - r/n * (R*ϕ)*ϕ'
  end

  # From 10.1007/s11222-008-9110-y
  err = α-αstar
  s = max(0,state.s+tanh(state.prev_err*err))
  nextσ2 = exp(log(σ2) + s*err)

  next_state = (;iter = iter+1,σ2 = nextσ2, R = nextR, prev_err = err, s )

  if rand() < α
    return GradientChainState(y, logp_y, gradlogp_y), true, α, next_state
  end

  return GradientChainState(x, logp_x, gradlogp_x), false, α, next_state
end

Base.@kwdef @concrete struct FisherULA <: AbstractMCMCKernel{Val{true}}
  λ = 10.
  αstar = 0.574
end

function init_kernel_state(_::FisherULA,x,scale,Σ)
  (;iter=1,σ2 = scale,R = Matrix(Σ.chol.L), s=0,prev_err=0.)
end

function (k::FisherULA)(target,chain_state::GradientChainState,state)
  @unpack iter,σ2,R = state
  @unpack λ,αstar = k

  x,logp_x,gradlogp_x = chain_state.x,chain_state.logp, chain_state.gradlogp

  ϵ = sqrt(σ2/(sum(abs2,R)/length(x)))

  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵx/2*R'*gradlogp_x
  y = x + ϵ * R*velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = -velocity_middle - ϵ/2*R'*gradlogp_y

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

  # From 10.1007/s11222-008-9110-y
  err = α-αstar
  s = max(0,state.s+tanh(state.prev_err*err))
  nextσ2 = exp(log(σ2) + s*err)


  next_state = (;iter = iter+1,σ2 = nextσ2, R = nextR, prev_err = err, s )

  if isinf(logp_y)
    return GradientChainState(x, logp_x, gradlogp_x), false, α, next_state
  end

  return GradientChainState(y, logp_y, gradlogp_y), true, α, next_state
end

struct MALA <: AbstractMCMCKernel{Val{true}} end

function init_kernel_state(_::MALA,x,scale,Σ)
  (;ϵ=scale,R=Σ.chol.L)
end

function (k::MALA)(target,chain_state::GradientChainState,state)
  @unpack ϵ, R = state

  x,logp_x,gradlogp_x = chain_state.x,chain_state.logp, chain_state.gradlogp
  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵ/2*R'*gradlogp_x
  y = x + ϵ * R *velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = -velocity_middle - ϵ/2*R'*gradlogp_y

  α = if isinf(logp_y)
    zero(logp_x)
  else
    min(1.,exp(logp_y - 0.5 * sum(abs2,velocity_end) -
               logp_x + 0.5 * sum(abs2,velocity)))
  end

  if rand() < α
    return GradientChainState(y, logp_y, gradlogp_y), true, α, state
  end

  return GradientChainState(x, logp_x, gradlogp_x), false, α, state
end

struct ULA <: AbstractMCMCKernel{Val{true}} end

function init_kernel_state(_::ULA,x,scale,Σ)
  (;ϵ=sqrt(scale/(tr(Σ)/length(x))),R=Σ.chol.L)
end

function (k::ULA)(target,chain_state::GradientChainState,state)
  @unpack ϵ, R = state

  x,logp_x,gradlogp_x = chain_state.x,chain_state.logp, chain_state.gradlogp
  # Leapfrog integrator
  velocity = randn(length(x))
  velocity_middle  = velocity + ϵ/2*R'*gradlogp_x
  y = x + ϵ * R *velocity_middle
  logp_y,gradlogp_y = LD.logdensity_and_gradient(target,y)
  velocity_end = -velocity_middle - ϵ/2*R'*gradlogp_y

  α = min(1.,exp(logp_y - 0.5 * sum(abs2,velocity_end) -
                  logp_x + 0.5 * sum(abs2,velocity)))

  if isinf(logp_y)
    return GradientChainState(x, logp_x, gradlogp_x), false, α, state
  end

  return GradientChainState(y, logp_y, gradlogp_y), true, α, state
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

function (k::PathDelayedRejection)(target,chain_state::ChainState,C::PDMat)
  @unpack proposal_dist,factor,n_stages = k

  x,logp_x = chain_state.x,chain_state.logp
  n = length(x)

  us = Vector{Vector{eltype(x)}}(undef,n_stages+1)
  logps = Vector{typeof(logp_x)}(undef,n_stages+1)
  us[1] = zeros(n)
  logps[1] = logp_x
 
  local α
  for i in 1:n_stages
    us[i+1] = factor^(i-1) * rand(proposal_dist,n)
    y = x + unwhiten(C,us[i+1])
    logps[i+1] =  LD.logdensity(target,y)

    α = min(1.,exp(_logα(
      proposal_dist,
      logps,us,
      factor,i
    )))

    if rand() < α
      return ChainState(y,logps[i+1]), true, α*(1-(i-1)/n_stages),C
    end
  end

  return chain_state, false, α/n_stages, C
end

Base.@kwdef @concrete struct RWMH <: AbstractMCMCKernel{Val{false}}
  proposal_dist = Normal()
end

function (k::RWMH)(target,chain_state::ChainState,C::PDMat)
  @unpack proposal_dist = k
  x,logp_x = chain_state.x,chain_state.logp
  u = rand(proposal_dist,length(x))
  y = x + unwhiten(C,u)
  logp_y = LD.logdensity(target,y)

  α = min(1.,exp.(logp_y + sum(logpdf(proposal_dist,-u)) - logp_x - sum(logpdf(proposal_dist,u))))
  if rand() < α
    return ChainState(y, logp_y), true, α, C
  end

  return ChainState(x, logp_x), false, α, C
end

Base.@kwdef @concrete struct SliceSampling <: AbstractMCMCKernel{Val{false}}
  m = 20 # Determines the maximum number of log density evaluation in the stepping-out procedure
end

function init_kernel_state(_::SliceSampling,x,scale,Σ)
  (;w=scale/2,Σ)
end

function (k::SliceSampling)(target,chain_state::ChainState,state)
  @unpack w,Σ = state
  @unpack m = k

  x,logp_x = chain_state.x,chain_state.logp
  counter = 0

  # Slice direction -> From a zero mean multivariate normal 
  d = randn(length(x))
  unwhiten!(Σ,d)

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
      return ChainState(y, logp_y), true, 1/counter, state
    end
    if u < 0
      L = u
    else
      R = u
    end
  end
end
