"""
    MetaNumber(val,info::Tuple)

  Number that has some metadata, but promotes to val
  so that if an operation is performed the metadata is dropped
"""
struct MetaNumber{T<:Real,I<:NamedTuple} <: Real
  val::T
  info::I
end

Base.promote(a::MetaNumber,b,cs...) = Base.promote(a.val,b,cs...)
Base.promote(a,b::MetaNumber,cs...) = Base.promote(a,b.val,cs...)
Base.promote(a::MetaNumber,b::MetaNumber,cs...) = Base.promote(a.val,b.val,cs...)

Base.convert(::Type{T},a::MetaNumber{T}) where T = a.val
Base.convert(::Type{W},a::MetaNumber) where W<:Number = convert(W,a.val)

function Base.convert(::Type{<:MetaNumber{T,<:NamedTuple{Names,V}}},a::MetaNumber) where {T,Names,V}
  new_info = (;(k=> a.info[k] for k in Names)...)
  return MetaNumber(convert(T,a.val),new_info)
end


Base.:(+)(x::T, y::T) where {T<:MetaNumber} = x.val+y.val
Base.:(*)(x::T, y::T) where {T<:MetaNumber} = x.val*y.val
Base.:(-)(x::T, y::T) where {T<:MetaNumber} = x.val-y.val
Base.:(/)(x::T, y::T) where {T<:MetaNumber} = x.val/y.val
Base.:(^)(x::T, y::T) where {T<:MetaNumber} = x.val^y.val
Base.isinf(x::T) where {T<:MetaNumber} = isinf(x.val)
Base.zero(x::T) where {T<:MetaNumber} = zero(x.val)
Base.one(x::T) where {T<:MetaNumber} = one(x.val)

struct FullLogDensity{R,M}
  ref::R
  mul::M
  dim::Int
end

function FullLogDensity(ref_logdensity,mul_logdensity; dim=LD.dimension(ref_logdensity))
  return FullLogDensity(ref_logdensity,mul_logdensity,dim)
end

LD.dimension(ℓ::FullLogDensity) = ℓ.dim
LD.capabilities(::Type{<:FullLogDensity}) = LD.LogDensityOrder{1}()

function LD.logdensity(ℓ::FullLogDensity,θ)
	ref = LD.logdensity(ℓ.ref,θ)
	mul = if ref == -Inf
		oftype(ref, -Inf)             
	else
		LD.logdensity(ℓ.mul, θ)        
	end
	MetaNumber(mul + ref,(;mul,ref))
end

function LD.logdensity_and_gradient(ℓ::FullLogDensity,θ)
	ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
	mul,mulgrad = if ref == -Inf
		oftype(ref,-Inf), -refgrad
	else
		LD.logdensity_and_gradient(ℓ.mul,θ)
	end
	MetaNumber(mul + ref,(;mul,mulgrad,ref,refgrad)), mulgrad + refgrad
end

struct ConditionedLogDensity{L,T}
  ℓ::L
  dim::Int
  inds::Vector{Int}
  not_inds::Vector{Int}
  vals::Vector{T}
end

LD.dimension(c::ConditionedLogDensity) = c.dim
LD.capabilities(::Type{<:ConditionedLogDensity}) = LD.LogDensityOrder{0}()

function LD.logdensity(c::ConditionedLogDensity,x)
  y = Vector{eltype(x)}(undef,c.dim+length(c.vals))
  y[c.inds] .= c.vals
  y[c.not_inds] .= x
  LD.logdensity(c.ℓ,y)
end

function condition(ℓ,x,inds::Vector{Int})
  return ConditionedLogDensity(
    ℓ,
    length(x)-length(inds),
    inds,
    filter(x -> !(x in inds),1:length(x)),
    x[inds]
  )
end

norm2(v::AbstractVector) = dot(v,v)

_order(_::LD.LogDensityOrder{K}) where K = K
function _lowest_capability(ℓ1::L,ℓs::Vararg{L}) where L <: LD.LogDensityOrder
  o = mapreduce(_order,(a,b) -> a < b,ℓs,init=_order(ℓ1))
  return LD.LogDensityOrder{o}()
end

_lowest_capability(ℓs...) = _lowest_capability((LD.capabilities(ℓ) for ℓ in ℓs)...)

function _default_sampler(ref_logdensity,mul_logdensity)
  lc = _lowest_capability(ref_logdensity,mul_logdensity)
  if lc === LD.LogDensityOrder{0}()
    return PathDelayedRejection()
  end
  return MALA()
end

function _default_metric_estimator(n_dim,n_samples)
  if n_samples > 2*n_dim
    return ParticleCov()
  elseif n_samples > n_dim
    return ParticleVar()
  else
    return IdentityCov()
  end
end

"""
    stabilized_map(f,x,map_func)

  Apply an arbitrary mapping function, `map_func`, and stabilize is output type.

"""
function stabilized_map(f, x, map_func)
    raw_result = map_func(f, x)
    
    # 1. Zero-allocation fast path: If the backend already gave us a concrete vector, 
    # just hand it directly to the function barrier. No copies made.
    if isconcretetype(eltype(raw_result))
        return enforce_type(raw_result)
    else
        # 2. Allocating fallback: Only copy if we got a Vector{Any} (e.g. from Distributed)
        tight_result = identity.(raw_result)
        return enforce_type(tight_result)
    end
end

function enforce_type(res::AbstractVector{T}) where T
    return res::Vector{T}
end


# TODO: Switch to a different method after some size
function ensure_posdef(M::Matrix{T}) where T <: Real
	ε = sqrt(eps(T))
	invε = inv(ε)
	if !issymmetric(M)
		M = (M + M')/2
	end
	F = eigen(Symmetric(M))
	for i in eachindex(F.values)
		F.values[i] = clamp(F.values[i],ε,invε)
	end

	# Symmetric must be used due to numerical precision
	return Symmetric(Matrix(F))
end

# As ensure_posdef often involves a matrix decomposition the
# inverse can be computed more efficiently
function ensure_posdef_and_invert(M::Matrix{T}) where T <: Real
	ε = sqrt(eps(T))
	invε = inv(ε)
	if !issymmetric(M)
		M = (M + M')/2
	end
	F = eigen(Symmetric(M))
	for i in eachindex(F.values)
		F.values[i] = clamp(1/F.values[i],ε,invε)
	end

	return Symmetric(F.vectors * Diagonal(F.values) * F.vectors')
end


