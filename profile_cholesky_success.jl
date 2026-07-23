"""Profile how often Cholesky succeeds vs needs eigenvalue fallback in a real SMC run."""

using LinearAlgebra, Random, Statistics, StatsBase, BenchmarkTools
using LogDensityProblems
LD = LogDensityProblems

include("test/test_utils.jl")
using Distributions: MvNormal

# Instrument the hybrid approach to count success/fallback
const CHOLESKY_SUCCESS = Ref(0)
const CHOLESKY_FALLBACK = Ref(0)
const TOTAL_CALLS = Ref(0)

function ensure_posdef_instrumented(M::Matrix{T}; max_jitter_retries=10) where T <: Real
	ε = sqrt(eps(T))
	invε = inv(ε)
	if !issymmetric(M)
		M = (M + M')/2
	end

	TOTAL_CALLS[] += 1
	
	# Try Cholesky first (fast path for well-conditioned matrices)
	try
		cholesky(Symmetric(M))
		CHOLESKY_SUCCESS[] += 1
		return Symmetric(M)
	catch
		CHOLESKY_FALLBACK[] += 1
		# Fall back to eigenvalue approach for ill-conditioned matrices
		F = eigen(Symmetric(M))
		for i in eachindex(F.values)
			F.values[i] = clamp(F.values[i], ε, invε)
		end
		return Symmetric(Matrix(F))
	end
end

function ensure_posdef_and_invert_instrumented(M::Matrix{T}) where T <: Real
	ε = sqrt(eps(T))
	invε = inv(ε)
	if !issymmetric(M)
		M = (M + M')/2
	end

	TOTAL_CALLS[] += 1
	
	# Try Cholesky first (fast path) — compute inverse via triangular solves
	try
		chol = cholesky(Symmetric(M))
		CHOLESKY_SUCCESS[] += 1
		L = chol.L
		Linv = inv(UpperTriangular(L'))
		return Symmetric(Linv * Linv')
	catch
		CHOLESKY_FALLBACK[] += 1
		# Fall back to eigenvalue approach for ill-conditioned matrices
		F = eigen(Symmetric(M))
		for i in eachindex(F.values)
			F.values[i] = clamp(1/F.values[i], ε, invε)
		end
		return Symmetric(F.vectors * Diagonal(F.values) * F.vectors')
	end
end

# Simulate multiple covariance matrices at different sample sizes
println("=== Profiling Cholesky Success Rate ===\n")

test_configs = [
    ("Very small sample size (5 dims, 5 samples)", 5, 5),
    ("Small sample size (5 dims, 10 samples)", 5, 10),
    ("Moderate (10 dims, 30 samples)", 10, 30),
    ("Well-sampled (20 dims, 200 samples)", 20, 200),
]

for (desc, d, n_samples) in test_configs
    CHOLESKY_SUCCESS[] = 0
    CHOLESKY_FALLBACK[] = 0
    TOTAL_CALLS[] = 0
    
    println("Testing: $desc")
    
    # Generate several covariance matrices
    Random.seed!(42)
    for trial in 1:20
        X = randn(d, n_samples)
        W = ones(n_samples) / n_samples
        M = cov(X, FrequencyWeights(W), 2)
        
        # Add some noise to simulate different conditioning
        if trial > 10
            M = M + (trial - 10) * 0.01 * I(d)  # Better conditioned
        end
        
        # Test both functions
        ensure_posdef_instrumented(M)
        ensure_posdef_and_invert_instrumented(M)
    end
    
    success_rate = CHOLESKY_SUCCESS[] / TOTAL_CALLS[] * 100
    println("  Cholesky success: $(CHOLESKY_SUCCESS[]) / $(TOTAL_CALLS[]) = $(round(success_rate, digits=1))%")
    println("  Cholesky fallback: $(CHOLESKY_FALLBACK[])")
    println()
end

println("\n=== Interpretation ===")
println("If success rate >> 50%, Cholesky fast path is worth the overhead.")
println("If success rate << 50%, eigenvalue-only approach is more efficient.")
