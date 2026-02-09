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
  mul = LD.logdensity(ℓ.mul,θ)
  MetaNumber(mul + ref,(;mul,ref))
end

function LD.logdensity_and_gradient(ℓ::FullLogDensity,θ)
  ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
  mul,mulgrad = LD.logdensity_and_gradient(ℓ.mul,θ)
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

function _default_cov_estimator(n_dim,n_samples)
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

  Uses the `Base.map` infrastructure to infer the return type of the map, using 
  a type assertion to enforce it.
"""
function stabilized_map(f,x,map_func)
  T = only(Base.return_types(f,(typeof(first(x)),)))
  return map_func(f,x)::Vector{T}
end

