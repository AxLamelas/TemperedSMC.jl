"""
  Number that has some metadata, but promotes to val
  so that if an operation is performed the metadata is dropped
"""
struct MetaNumber{T<:Real,I} <: Real
  val::T
  info::I
end

Base.promote(a::MetaNumber,b,cs...) = Base.promote(a.val,b,cs...)
Base.promote(a,b::MetaNumber,cs...) = Base.promote(a,b.val,cs...)
Base.promote(a::MetaNumber,b::MetaNumber,cs...) = Base.promote(a.val,b.val,cs...)

Base.:(+)(x::T, y::T) where {T<:MetaNumber} = x.val+y.val
Base.:(*)(x::T, y::T) where {T<:MetaNumber} = x.val*y.val
Base.:(-)(x::T, y::T) where {T<:MetaNumber} = x.val-y.val
Base.:(/)(x::T, y::T) where {T<:MetaNumber} = x.val/y.val
Base.:(^)(x::T, y::T) where {T<:MetaNumber} = x.val^y.val

struct TemperedLogDensity{P,L}
  ref::P
  mul::L
  β::Float64
  dim::Int
end

LD.dimension(ℓ::TemperedLogDensity) = ℓ.dim
LD.capabilities(ℓ::TemperedLogDensity) = _lowest_capability(ℓ.ref,ℓ.mul)

function LD.logdensity(ℓ::TemperedLogDensity,θ)
  ref = LD.logdensity(ℓ.ref,θ) 
  if isinf(ref)
    return MetaNumber(ref,(;mul=ref,ref))
  end
  mul = LD.logdensity(ℓ.mul,θ) 
  MetaNumber(ℓ.β * mul + ref,(;mul,ref))
end

function LD.logdensity_and_gradient(ℓ::TemperedLogDensity,θ)
  ref,refgrad = LD.logdensity_and_gradient(ℓ.ref,θ)
  if isinf(ref)
    return   MetaNumber(ref,(;mul=ref,mulgrad=refgrad,ref,refgrad)), refgrad
  end
  mul,mulgrad = LD.logdensity_and_gradient(ℓ.mul,θ)
  MetaNumber(ℓ.β * mul + ref,(;mul,mulgrad,ref,refgrad)), ℓ.β * mulgrad + refgrad
end

struct ConditionedLogDensity{L,T}
  ℓ::L
  dim::Int
  inds::Vector{Int}
  not_inds::Vector{Int}
  vals::Vector{T}
end

LD.dimension(c::ConditionedLogDensity) = c.dim
LD.capabilities(c::ConditionedLogDensity) = LD.LogDensityOrder{0}()

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

