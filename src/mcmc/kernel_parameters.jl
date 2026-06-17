abstract type AbstractKernelParameters end

function get_parameters(::AbstractKernelParameters) end
function update_parameters!(adpt::AbstractKernelParameters,chains,Σ) end
function improve_acceptance!(adpt::AbstractKernelParameters) end

struct ScaleAdaptation{T<:Real} <: AbstractKernelParameters 
	w::Vector{T}
	scales::Vector{T}
	tmp::Vector{T}
	k::T
end

function ScaleAdaptation(n::Int,dim::Int;k=0.05)
	ref_scale = 2.38/sqrt(dim)
	α = 1/(5k)^2
	θ = ref_scale/α
	return ScaleAdaptation(fill(1/n,n),rand(Gamma(α,θ),n),zeros(n),k)
end

function update_parameters!(adpt::ScaleAdaptation,chains,Σ)
	# Resample scale following 10.1214/13-BA814
	for j in eachindex(chains,adpt.w)
		c = chains[j]
		# Rao-Blackwellized estimator of the Expected squared jump distance
		w = 0.
		for i in 1:length(c.states)-1
			δ = c.states[i+1].x-c.states[i].x
			w += c.γ[i] * invquad(Σ,δ)
		end
		w /= length(c.states)- 1

		adpt.w[j] = w
	end

	n = sum(adpt.w)
	if iszero(n)
		improve_acceptance!(adpt)
		fill!(adpt.w,1/length(adpt.w))
	else
		adpt.w ./=  sum(adpt.w)
	end

	index_dist = Categorical(adpt.w)

	α = 1/adpt.k^2
	for i in eachindex(adpt.tmp)
		ind = rand(index_dist)
		θ = adpt.scales[ind]/α#((1+adpt.k)*α)
		adpt.tmp[i] = rand(Gamma(α,θ))
	end

	copyto!(adpt.scales,adpt.tmp)
	shuffle!(adpt.scales)
end

function improve_acceptance!(adpt)
	for i in eachindex(adpt.scales)
		adpt.scales[i] /= 10
	end
end

get_parameters(adpt::ScaleAdaptation) = adpt.scales

struct FixedScale{T<:Real} <: AbstractKernelParameters 
	scale::T
	n::Int
end

function FixedScale(n::Int,dim::Int)
	ref_scale = 2.38/sqrt(dim)
	return return FixedScale(ref_scale,n)
end

function update_parameters!(adpt::FixedScale,chains,Σ) end
function improve_acceptance!(adpt::FixedScale) end

get_parameters(adpt::FixedScale) = Fill(adpt.scale,adpt.n)

