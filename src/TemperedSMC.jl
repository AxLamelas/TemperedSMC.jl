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
using Primes
using ProgressMeter
using ConcreteStructs
using StatsFuns

import LogDensityProblems as LD

include("resampling.jl")
include("tempered_logdensity.jl")
include("mcmc_kernels.jl")
include("mcmc_chain.jl")
include("utils.jl")
include("cov_estimators.jl")
include("wfsmc.jl")
include("smc.jl")

# Adaptive scheduler using conditional ess from 
# "Toward Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach"


end # module TemperedSMC
