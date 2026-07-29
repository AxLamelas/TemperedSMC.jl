abstract type AbstractDistributionSequence end

abstract type AbstractSequenceState{T} end

function init_sequence_state(seq::AbstractSequenceState,ℓ) end

function progress_fraction(state::AbstractSequenceState) end

include("adaptive_tempering.jl")

