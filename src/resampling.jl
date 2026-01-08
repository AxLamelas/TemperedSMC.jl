# Resampling methods adapted from AdvancedPS.jl

using Random, Distributions

abstract type AbstractResampler end

function (::AbstractResampler)(w::AbstractVector,num_particles=length(w)) end

struct MultinomialResampler <: AbstractResampler end

function (::MultinomialResampler)(
    w::AbstractVector{<:Real}, num_particles::Integer=length(w)
)
    return rand(Distributions.sampler(Distributions.Categorical(w)), num_particles)
end

struct ResidualResampler <: AbstractResampler end

function (::ResidualResampler)(
    w::AbstractVector{<:Real},
    num_particles::Integer=length(w),
)
    # Pre-allocate array for resampled particles
    indices = Vector{Int}(undef, num_particles)

    # deterministic assignment
    residuals = similar(w)
    i = 1
    @inbounds for j in 1:length(w)
        x = num_particles * w[j]
        floor_x = floor(Int, x)
        for k in 1:floor_x
            indices[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end

    # sampling from residuals
    if i <= num_particles
        residuals ./= sum(residuals)
        rand!(Distributions.Categorical(residuals), view(indices, i:num_particles))
    end

    return indices
end

struct StratifiedResampler <: AbstractResampler end

function (::StratifiedResampler)(
    weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]

    # generate all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # sample next `u` (scaled by `n`)
        u = oftype(v, i - 1 + rand())

        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample
    end

    return samples
end

struct SystematicResampler <: AbstractResampler end

function (::SystematicResampler)(
    weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand())

    # find all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample

        # update `u`
        u += one(u)
    end

    return samples
end

struct SSPResampler <: AbstractResampler end

"""
    ssp_sampling

SSP stands for Srinivasan Sampling Process.
This resampling scheme is discussed in Gerber *et al.*[^GCW2019]. Basically, it has similar properties as systematic resampling (number of off-springs is either k or k + 1, with
k <= N W^n < k +1), and in addition is consistent. See that paper for more
details.

# Reference
[^GCW2019]: Gerber M., Chopin N. and Whiteley N. (2019). Negative association, ordering and convergence of resampling methods. *The Annals of Statistics* 47 (2019), no. 4, 2236–2260.
"""
function (::SSPResampler)(
    weights::AbstractVector, n_resample::Int=length(weights)
)
    n       = length(weights)
    m       = n_resample
    mw      = m * weights
    n_child = floor.(Int, mw)
    xi      = mw - n_child
    u       = rand(n - 1)
    i, j    = 1, 2
    for k in 1:(n - 1)
        δi = min(xi[j], 1 - xi[i])
        δj = min(xi[i], 1 - xi[j])
        ∑δ = δi + δj

        pj = (∑δ > 0) ? δi / ∑δ : 0
        if u[k] < pj
            j, i = i, j
            δi   = δj
        end
        if xi[j] < 1 - xi[i]
            xi[i] += δi
            j = k + 2
        else
            xi[j]      -= δi
            n_child[i] += 1
            i          = k + 2
        end
    end

    # due to round-off error accumulation, we may be missing one particle
    if sum(n_child) == m - 1
        last_ij = if j == n + 1
            i
        else
            j
        end
        if xi[last_ij] > 0.99
            n_child[last_ij] += 1
        end
    end
    if sum(n_child) != m
        throw(error("ssp resampling: wrong size for output"))
    end
    return inverse_rle(1:n, n_child)
end
