module TemperedSMC

export waste_free_smc, smc

using CovarianceEstimation
using StatsBase
using LogExpFunctions
using LinearAlgebra
using Distributions
using Random
using SimpleUnPack
using Distances
using ProgressMeter
using ConcreteStructs
using StatsFuns
using PDMats
using FillArrays

import LogDensityProblems as LD

# Sequence interface
# Kernel interface

include("resampling.jl")
include("mcmc/mcmc.jl")
include("sequences/sequences.jl")
include("utils.jl")
include("metric_estimators.jl")
include("smc.jl")


# Adaptive scheduler using conditional cess from
# "Toward Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach"

end # module TemperedSMC
