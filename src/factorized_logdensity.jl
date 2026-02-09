"""
    n_data_points(ℓ)

Number of data points associated with the factorized log density `ℓ`. See `factorized_logdensity`.

"""
function n_data_points(ℓ) end

"""
    factorized_logdensity(ℓ, inds, x)

Evaluate the log density `ℓ` of data points `inds` at `x`, which has length compatible with its
`dimension`.

Return a vector of real number, which may or may not be finite (can also be `NaN`). Non-finite values
other than `-Inf` are invalid but do not error, caller should deal with these appropriately.
Caller must ensure that inds is within the valid range of data points which is from `1` to `n_data_points`

"""
function factorized_logdensity(ℓ,inds,x) end

"""
    logdensity(ℓ, inds, x)

Evaluate the log density `ℓ` of data points `inds` at `x`, which has length compatible with its
`dimension`.

Return a real number, which may or may not be finite (can also be `NaN`). Non-finite values
other than `-Inf` are invalid but do not error, caller should deal with these appropriately.
Caller must ensure that inds is within the valid range of data points which is from `1` to `n_data_points`

"""
function LD.logdensity(ℓ,inds,x) end
