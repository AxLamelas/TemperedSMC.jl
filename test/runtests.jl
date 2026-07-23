using Test

include("test_utils.jl")

@testset "TemperedSMC" begin
    include("test_metanumber_fulldensity.jl")
    include("test_resampling.jl")
    include("test_adaptive_tempering.jl")
    include("test_metric_estimators.jl")
    include("test_mcmc_kernels.jl")
    include("test_smc_gaussian.jl")
    include("test_waste_free_smc.jl")
end
