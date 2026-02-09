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

import LogDensityProblems as LD

# Sequence interface
# Kernel interface

include("resampling.jl")
include("sequences/sequences.jl")
include("mcmc/mcmc.jl")
include("utils.jl")
include("cov_estimators.jl")
include("smc.jl")


# Adaptive scheduler using conditional cess from
# "Toward Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach"

# TODO: Consider adding a AbstractKernelAdapt structure that goes as first argument into the 
# adaptativity methods to allow the specification of different strategies for the same kernel
# TODO: Implement IBIS
# TODO: Maybe move factorized_logdensiy interface to another package

# TODO: Consider changing the state and gradeints to expect named tuples
# Then the Tuple handling for the Gibbs we be already permited

end # module TemperedSMC
