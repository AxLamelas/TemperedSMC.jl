abstract type AbstractChainState{T} end

struct GradientChainState{T,L} <: AbstractChainState{T}
  x::Vector{T}
  logp::L
  gradlogp::Vector{T}
end

struct ChainState{T,L} <: AbstractChainState{T}
  x::Vector{T}
  logp::L
end

abstract type AbstractMCMCKernel{G <: Val} end

# An mcmc kernel must return:
#  - An AbstractChainState
#  - wether the proposal was accepted or not
#  - a factor to weight the expected squared jump distance for parameter
#  adaptation -> usually the acceptance rate for most kernels
#  - the new state for the kernel

function (_::AbstractMCMCKernel)(target,chain_state::AbstractChainState,ker_state) end

@inline init_chain_state(_::AbstractMCMCKernel{Val{false}},target,x::AbstractVector) = ChainState(collect(x),LD.logdensity(target,x))
@inline init_chain_state(_::AbstractMCMCKernel{Val{true}},target,x::AbstractVector) = GradientChainState(collect(x),LD.logdensity_and_gradient(target,x)...)

# If there is no density information in the arguments initialize it
@inline (k::AbstractMCMCKernel)(target,x::AbstractVector,ker_state) = k(target,init_chain_state(k,target,x),ker_state)

# Default kernel initialization: scale is the only parameter in this case
init_kernel_state(_::AbstractMCMCKernel,x,scale,Σ) = scale*Σ

usesgrad(_::AbstractMCMCKernel{Val{V}}) where {V} = V


# Default kernel adaptation -> only scale; custom implementation can have more parameters, i.e., NamedTuple/Vector
function kernel_parameter_prior(_::AbstractMCMCKernel,dim)
  ref_scale = 2.38/dim^2
  return LogNormal(log(ref_scale),2)
end

function kernel_param_perturbative_dist(_::AbstractMCMCKernel,scale)
  pertub_scale = 0.2
  # LogNormal(log(scale)-pertub_scale^2/2,pertub_scale)
  LogNormal(log(scale),pertub_scale)
end


include("chain.jl")
include("kernels.jl")
