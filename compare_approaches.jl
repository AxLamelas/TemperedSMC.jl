"""Compare eigenvalue clamping vs Cholesky+jitter approaches for ensure_posdef."""

using LinearAlgebra, Random, Statistics, StatsBase, BenchmarkTools

# ============================================================================
# Current approach: Eigenvalue clamping
# ============================================================================
function ensure_posdef_eigen(M::Matrix{T}) where T <: Real
    ε = sqrt(eps(T))
    invε = inv(ε)
    if !issymmetric(M)
        M = (M + M')/2
    end
    F = eigen(Symmetric(M))
    for i in eachindex(F.values)
        F.values[i] = clamp(F.values[i], ε, invε)
    end
    return Symmetric(Matrix(F))
end

function ensure_posdef_and_invert_eigen(M::Matrix{T}) where T <: Real
    ε = sqrt(eps(T))
    invε = inv(ε)
    if !issymmetric(M)
        M = (M + M')/2
    end
    F = eigen(Symmetric(M))
    for i in eachindex(F.values)
        F.values[i] = clamp(1/F.values[i], ε, invε)
    end
    return Symmetric(F.vectors * Diagonal(F.values) * F.vectors')
end

# ============================================================================
# New approach: Cholesky + jitter and retry
# ============================================================================
function ensure_posdef_cholesky(M::Matrix{T}; max_retries=20) where T <: Real
    ε = sqrt(eps(T))

    if !issymmetric(M)
        M = (M + M')/2
    end

    # Try Cholesky directly first
    try
        chol = cholesky(Symmetric(M))
        return Symmetric(M)  # Already positive definite
    catch
        # Add jitter and retry
        d = size(M, 1)
        jitter = ε

        for attempt in 1:max_retries
            try
                M_jittered = M + jitter * I(d)
                chol = cholesky(Symmetric(M_jittered))
                # If Cholesky succeeds, return the jittered matrix (already PD by construction)
                return Symmetric(M_jittered)
            catch
                # Double jitter and retry
                jitter *= 2
                if jitter > 1.0  # Fallback to eigenvalue approach if jitter gets too large
                    return ensure_posdef_eigen(M)
                end
            end
        end

        # Fallback if retries exhausted
        return ensure_posdef_eigen(M)
    end
end

function ensure_posdef_and_invert_cholesky(M::Matrix{T}; max_retries=20) where T <: Real
    ε = sqrt(eps(T))

    if !issymmetric(M)
        M = (M + M')/2
    end

    # Try Cholesky directly first
    try
        chol = cholesky(Symmetric(M))
        # Compute inverse via Cholesky factor: M = L*L', so M^(-1) = (L')^(-1) * L^(-1)
        L = chol.L
        Linv = inv(UpperTriangular(L'))
        return Symmetric(Linv * Linv')
    catch
        # Add jitter and retry
        d = size(M, 1)
        jitter = ε

        for attempt in 1:max_retries
            try
                M_jittered = M + jitter * I(d)
                chol = cholesky(Symmetric(M_jittered))
                L = chol.L
                Linv = inv(UpperTriangular(L'))
                return Symmetric(Linv * Linv')
            catch
                jitter *= 2
                if jitter > 1.0
                    return ensure_posdef_and_invert_eigen(M)
                end
            end
        end

        return ensure_posdef_and_invert_eigen(M)
    end
end

# ============================================================================
# Test suite: Robustness across matrix conditions
# ============================================================================
println("=== Robustness Testing ===\n")

test_cases = []

# Test 1: Well-conditioned matrix
Random.seed!(42)
X = randn(5, 50)
W = ones(50) / 50
M1 = cov(X, FrequencyWeights(W), 2)
push!(test_cases, ("Well-conditioned (cond=$(round(cond(M1), digits=2)))", M1))

# Test 2: Poorly-conditioned matrix (few samples)
X = randn(10, 3)
M2 = cov(X, FrequencyWeights(ones(3)/3), 2)
push!(test_cases, ("Ill-conditioned (cond=$(round(cond(M2), digits=0)))", M2))

# Test 3: Near-singular matrix
M3 = [1.0 1.0-1e-10; 1.0-1e-10 1.0]
push!(test_cases, ("Near-singular (cond=$(round(cond(M3), digits=0)))", M3))

# Test 4: Rank-deficient matrix
M4 = [1.0 1.0; 1.0 1.0]  # rank 1
push!(test_cases, ("Rank-deficient", M4))

for (name, M) in test_cases
    println("Testing: $name")

    # Test ensure_posdef
    try
        R_eigen = ensure_posdef_eigen(M)
        eigvals_r = eigvals(Symmetric(R_eigen))
        println("  Eigen: ✓ (min eigval=$(minimum(eigvals_r)))")
    catch e
        println("  Eigen: ✗ ($e)")
    end

    try
        R_chol = ensure_posdef_cholesky(M)
        eigvals_r = eigvals(Symmetric(R_chol))
        println("  Cholesky: ✓ (min eigval=$(minimum(eigvals_r)))")
    catch e
        println("  Cholesky: ✗ ($e)")
    end

    # Test ensure_posdef_and_invert
    try
        Rinv_eigen = ensure_posdef_and_invert_eigen(M)
        eigvals_r = eigvals(Symmetric(Rinv_eigen))
        println("  Eigen (inv): ✓ (min eigval=$(minimum(eigvals_r)))")
    catch e
        println("  Eigen (inv): ✗ ($e)")
    end

    try
        Rinv_chol = ensure_posdef_and_invert_cholesky(M)
        eigvals_r = eigvals(Symmetric(Rinv_chol))
        println("  Cholesky (inv): ✓ (min eigval=$(minimum(eigvals_r)))")
    catch e
        println("  Cholesky (inv): ✗ ($e)")
    end

    println()
end

# ============================================================================
# Performance testing
# ============================================================================
println("=== Performance Benchmarking ===\n")

# Use a moderately-sized matrix for benchmarking
Random.seed!(42)
X = randn(20, 200)
W = ones(200) / 200
M = cov(X, FrequencyWeights(W), 2)

println("Benchmarking on 20×20 matrix (well-conditioned):")
println()

# Warmup
ensure_posdef_eigen(M)
ensure_posdef_cholesky(M)

# Benchmark ensure_posdef
b_eigen = @benchmark ensure_posdef_eigen($M)
b_chol = @benchmark ensure_posdef_cholesky($M)

println("ensure_posdef:")
println("  Eigen:    $(round(median(b_eigen.times)/1e6, digits=3)) ms/call")
println("  Cholesky: $(round(median(b_chol.times)/1e6, digits=3)) ms/call")
speedup = median(b_eigen.times) / median(b_chol.times)
println("  Speedup:  $(round(speedup, digits=2))x")
println()

# Benchmark ensure_posdef_and_invert
b_eigen_inv = @benchmark ensure_posdef_and_invert_eigen($M)
b_chol_inv = @benchmark ensure_posdef_and_invert_cholesky($M)

println("ensure_posdef_and_invert:")
println("  Eigen:    $(round(median(b_eigen_inv.times)/1e6, digits=3)) ms/call")
println("  Cholesky: $(round(median(b_chol_inv.times)/1e6, digits=3)) ms/call")
speedup_inv = median(b_eigen_inv.times) / median(b_chol_inv.times)
println("  Speedup:  $(round(speedup_inv, digits=2))x")
println()

# ============================================================================
# Numerical quality comparison
# ============================================================================
println("=== Numerical Quality ===\n")

# For a ill-conditioned matrix, check that both approaches produce valid PD matrices
Random.seed!(42)
X = randn(15, 5)
M = cov(X, FrequencyWeights(ones(5)/5), 2)

R_eigen = ensure_posdef_eigen(M)
R_chol = ensure_posdef_cholesky(M)

# Check positive definiteness
eigvals_eigen = eigvals(Symmetric(R_eigen))
eigvals_chol = eigvals(Symmetric(R_chol))

println("Matrix condition: $(round(cond(M), digits=1))")
println("After ensure_posdef_eigen: min eigval = $(minimum(eigvals_eigen))")
println("After ensure_posdef_cholesky: min eigval = $(minimum(eigvals_chol))")
println("Eigen approach preserves more eigenvalue spectrum structure")
println()

# ============================================================================
# Summary
# ============================================================================
println("=== Summary ===")
println("Cholesky + jitter approach:")
println("  ✓ Faster when matrix is already well-conditioned (no eigendecomposition)")
println("  ✓ Simple, stable algorithm")
println("  ✗ May add unnecessary jitter, changing matrix more than needed")
println("  ✗ Fallback to eigenvalue approach needed for very ill-conditioned matrices")
println()
println("Eigenvalue clamping approach:")
println("  ✓ Preserves eigenvalue spectrum structure (clamps individually)")
println("  ✓ Single, deterministic approach (no fallback needed)")
println("  ✗ Always requires full eigendecomposition (slower for large d)")
println()
println("Recommendation: Cholesky is faster but may be less precise for ill-conditioned matrices.")
println("Consider: hybrid approach (try Cholesky first, fall back to eigen if needed)")
