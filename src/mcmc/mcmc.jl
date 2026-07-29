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

"""
    PopulationChainState{C<:AbstractChainState,T} <: AbstractChainState{T}

Aggregate chain state representing all particles in a population.
Each particle has its own individual chain state (ChainState or GradientChainState).
Used as a single Markov chain state in the context of PopulationKernel.
"""
struct PopulationChainState{C<:AbstractChainState,T} <: AbstractChainState{T}
  states::Vector{C}   # states[j] = particle j's individual ChainState/GradientChainState
end

function PopulationChainState(states::Vector{C}) where {T,C<:AbstractChainState{T}}
  PopulationChainState{C,T}(states)
end

"""
    PopulationKernelState{S}

Aggregate kernel state representing all particles' individual kernel states.
Each particle has its own kernel state (which may be a NamedTuple, matrix, or any type).
"""
struct PopulationKernelState{S}
  states::Vector{S}   # states[j] = particle j's individual kernel state
end

# Ergonomic forwarding: used by transpose helper, MSJD calc, write-back loop
Base.length(p::Union{PopulationChainState,PopulationKernelState}) = length(p.states)
Base.getindex(p::Union{PopulationChainState,PopulationKernelState}, i) = p.states[i]
Base.iterate(p::Union{PopulationChainState,PopulationKernelState}, args...) = iterate(p.states, args...)
Base.eachindex(p::Union{PopulationChainState,PopulationKernelState}) = eachindex(p.states)

abstract type AbstractMCMCKernel{G <: Val} end

"""
    AbstractIndividualKernel{G} <: AbstractMCMCKernel{G}

Kernel that operates on a single particle x, producing a single proposal.
Signature: (k::AbstractIndividualKernel)(target, chain_state, ker_state)
           → (new_chain_state, accepted::Bool, acceptance_metric, new_ker_state)
"""
abstract type AbstractIndividualKernel{G} <: AbstractMCMCKernel{G} end

"""
    AbstractPopulationKernel{G} <: AbstractMCMCKernel{G}

Kernel that operates on all particles simultaneously, producing proposals for all.
Signature: (k::AbstractPopulationKernel)(target, chain_states::Vector, ker_states::Vector)
           → (new_chain_states::Vector, accepted::Vector{Bool}, metrics::Vector, new_ker_states::Vector)
Internally uses map_func to parallelize over particles.
"""
abstract type AbstractPopulationKernel{G} <: AbstractMCMCKernel{G} end

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


include("chain.jl")
include("kernels.jl")
include("kernel_parameters.jl")
