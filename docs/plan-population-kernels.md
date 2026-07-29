# IMPLEMENTATION PLAN: Population-Based Kernels Architecture (Reseeded)

## Executive Summary

This plan refactors the kernel architecture to introduce an explicit distinction between **individual kernels** (operate on single particles) and **population kernels** (operate on all particles in one step). The goal is to:
1. Add abstract subtypes to make this distinction explicit
2. Create a `PopulationKernel` wrapper that applies individual kernels independently via `map_func`
3. Refactor `smc()` and `waste_free_smc()` to use population-based kernels uniformly
4. Make DifferentialEvo automatically work in all contexts (currently requires `ParticleRepresentation()` metric)

---

## Part 1: Current Architecture Analysis

### Current Kernels Classification

**Individual Kernels** (operate on single `x` → single proposal):
- `RWMH`: Random walk Metropolis-Hastings
- `MALA`: Manifold adjusted Langevin algorithm (gradient-based)
- `ULA`: Unadjusted Langevin algorithm
- `FisherMALA`, `FisherULA`: Adaptive variants with Fisher-metric preconditioning
- `PathDelayedRejection`: Multi-stage adaptive proposal
- `AutoStepMALA`, `AutoStepRWMH`: Step-size adaptive variants via involution framework
- `SliceSampling`: Slice sampling (gradient-free)
- `DifferentialEvo`: DE-MCMC using snooker or pairwise jumps (operates on single particle, uses full population in state)

**Native Population Kernels**:
- None currently. The refactor introduces `PopulationKernel{G,K}` as the only native population kernel (a generic wrapper).

### DifferentialEvo: Already Works

DifferentialEvo is an individual kernel with special state management:
- Its `__call__` signature: `(target, chain_state::AbstractChainState, state) → (new_chain_state, accepted, γ, new_state)` (individual convention)
- Its `init_kernel_state`: `(k::DifferentialEvo, x, scale, particles) → (;scale, particles)` captures the full particle matrix in state
- **Today**: Works seamlessly when `metric_estimator=ParticleRepresentation()`, which returns `Fill(samples, length(xs))` and passes the sample matrix as the `Σ` parameter
- **After refactor**: Will work automatically in all contexts because the normalization to `PopulationKernel` ensures particles are always available

No wiring fix needed; the architecture simply makes DifferentialEvo's existing mechanism automatic.

---

## Part 2: Abstract Type Hierarchy

### New Abstract Types

Create **two new intermediate abstract types** under `AbstractMCMCKernel{G}`:

```julia
# In src/mcmc/mcmc.jl (alongside AbstractMCMCKernel)

"""
    AbstractIndividualKernel{G} <: AbstractMCMCKernel{G}
    
Kernel that operates on a single particle x, producing a single proposal.
Signature: (k::AbstractIndividualKernel)(target, chain_state, ker_state) 
           → (new_chain_state, accepted::Bool, acceptance_metric, new_ker_state)
"""
abstract type AbstractIndividualKernel{G} <: AbstractMCMCKernel{G} end

"""
    AbstractPopulationKernel{G} <: AbstractMCMCKernel{G}
    
Kernel that operates on all particles simultaneously, producing proposals for all.
Signature: (k::AbstractPopulationKernel)(target, chain_states::Vector, ker_states::Vector) 
           → (new_chain_states::Vector, accepted::Vector{Bool}, metrics::Vector, new_ker_states::Vector)
Internally uses map_func to parallelize over particles.
"""
abstract type AbstractPopulationKernel{G} <: AbstractMCMCKernel{G} end
```

### Update AbstractAutoStep Inheritance

```julia
# In src/mcmc/kernels.jl, change line 130 from:
abstract type AbstractAutoStep{G} <: AbstractMCMCKernel{G} end

# To:
abstract type AbstractAutoStep{G} <: AbstractIndividualKernel{G} end
```

### Classify Existing Kernels

**Reclassify all concrete kernels to inherit from `AbstractIndividualKernel`:**
- `RWMH <: AbstractIndividualKernel{Val{false}}`
- `MALA <: AbstractIndividualKernel{Val{true}}`
- `ULA <: AbstractIndividualKernel{Val{true}}`
- `FisherMALA <: AbstractIndividualKernel{Val{true}}`
- `FisherULA <: AbstractIndividualKernel{Val{true}}`
- `PathDelayedRejection <: AbstractIndividualKernel{Val{false}}`
- `AutoStepMALA <: AbstractAutoStep{Val{true}} <: AbstractIndividualKernel{Val{true}}`
- `AutoStepRWMH <: AbstractAutoStep{Val{false}} <: AbstractIndividualKernel{Val{false}}`
- `SliceSampling <: AbstractIndividualKernel{Val{false}}`
- `DifferentialEvo <: AbstractIndividualKernel{Val{false}}`

(Note: Most of these already inherit correctly from AbstractMCMCKernel; just change the parent to AbstractIndividualKernel)

---

## Part 3: Population Kernel Wrapper

### PopulationKernel{G,K} Struct

Create a **generic wrapper** that lifts any individual kernel to a population kernel:

```julia
# In src/mcmc/kernels.jl (after all individual kernel definitions)

"""
    PopulationKernel{G,K<:AbstractIndividualKernel{G}} <: AbstractPopulationKernel{G}
    
Wraps an individual kernel to apply it independently to each particle in a population.
Uses map_func for parallel/distributed execution.

The wrapper:
1. Accepts the full population (multiple chain_states via map)
2. Fans out via map_func to apply the individual kernel to each particle
3. Collects results and reconstructs population-level outputs
"""
struct PopulationKernel{G,K<:AbstractIndividualKernel{G}} <: AbstractPopulationKernel{G}
    kernel::K
    map_func::Function  # Captured at construction; default is `map`, can be pmap/distributed
end

# Constructor with default map
PopulationKernel(k::AbstractIndividualKernel{G}; map_func=map) where G = 
    PopulationKernel{G, typeof(k)}(k, map_func)
```

### Single Dispatch Overload for Initialization

Add one clean dispatch for `init_kernel_state` (mirrors existing `init_chain_state` pattern from mcmc.jl:25-26):

```julia
# In src/mcmc/kernels.jl (alongside PopulationKernel definition)

"""
    init_kernel_state(k::PopulationKernel, x, scale, Σ)
    
Delegate per-particle initialization to the wrapped individual kernel.
Returns a single kernel state (not a vector).
"""
init_kernel_state(k::PopulationKernel, x, scale, Σ) = init_kernel_state(k.kernel, x, scale, Σ)
```

### PopulationKernel Calling Convention

```julia
"""
    (k::PopulationKernel)(target, chain_states::Vector, ker_states::Vector)
    
Apply the individual kernel to each chain state independently via map_func.

Returns: (new_chain_states::Vector, accepted::Vector{Bool}, metrics::Vector, new_ker_states::Vector)

Invariant: Input and output have matching cardinality (n particles in → n results out).
"""
function (k::PopulationKernel)(target, chain_states::Vector, ker_states::Vector)
    results = k.map_func(chain_states, ker_states) do cs, ks
        k.kernel(target, cs, ks)
    end
    
    # Unpack results: Vector of (new_cs, acc, metric, new_ks) tuples
    new_chain_states = [r[1] for r in results]
    accepted = [r[2] for r in results]
    metrics = [r[3] for r in results]
    new_ker_states = [r[4] for r in results]
    
    return new_chain_states, accepted, metrics, new_ker_states
end
```

### Key Design Decisions

1. **Map Function Capture**: `PopulationKernel` captures `map_func` at construction, not at call time
   - Rationale: Allows `smc()/waste_free_smc()` to normalize kernel once at start; map_func is "baked in"
   - Simplifies call sites: every iteration just calls `pop_kernel(target, chain_states, ker_states)` uniformly

2. **Per-Particle Initialization**: `init_kernel_state(pop_kernel, x, p, Σ)` returns a single state
   - Matches the existing per-particle initialization pattern already in smc.jl
   - smc.jl builds the vector itself via `map(...) do x, p, Σ; init_kernel_state(...); end`
   - No temporary vector creation or `[1]` indexing — clean dispatch

3. **Signature Consistency**: 
   - `init_kernel_state` stays singular (one call per particle, returns one state)
   - `PopulationKernel.__call__` takes/returns vectors (operates on full population)
   - smc.jl loops over starting particles, collects individual states, passes vectors to pop_kernel

---

## Part 4: Refactoring smc.jl

### Overview of Changes

`smc()` must be modified to:
1. Accept **either** an individual kernel or a population kernel
2. If given an individual kernel, automatically wrap it in `PopulationKernel(kernel; map_func=map_func)`
3. Call the population kernel once per adaptive step loop iteration
4. Collect vector results (accepted, metrics) and process them uniformly

### Detailed Changes: Kernel Normalization (Lines 50-75)

```julia
function smc(seq::AbstractDistributionSequence, ref_logdensity, initial_samples::AbstractMatrix;
             mcmc_kernel::AbstractMCMCKernel = RWMH(),
             metric_estimator::AbstractMetric = _default_metric_estimator(size(initial_samples)...),
             ker_parameters::AbstractKernelParameters = ScaleAdaptation(reverse(size(initial_samples))...),
             resampler::AbstractResampler = ResidualResampler(),
             resampling_α = 0.5,
             mcmc_steps = max(50, 2LD.dimension(ref_logdensity)),
             adapt_mcmc_steps = true,
             adapt_stability = 0.01,
             map_func = map,
             callback = (_) -> false,
             store_trace = true,
             show_progress = true
            )
    
    # NEW: Normalize kernel to population-based
    pop_kernel = if mcmc_kernel isa AbstractPopulationKernel
        mcmc_kernel  # Already population-based
    else
        PopulationKernel(mcmc_kernel; map_func=map_func)  # Wrap individual kernel
    end
    
    # Rest of smc() body uses pop_kernel...
end
```

### Detailed Changes: MCMC Propagation Loop (Lines 135-170)

Replace the existing per-particle loop structure with direct calls to `pop_kernel`, which owns the map internally.

**Adaptive steps branch:**
```julia
if adapt_mcmc_steps
    # Initialize chain states for all particles
    chain_states = stabilized_map(starting_x, map_func) do x
        init_chain_state(pop_kernel, target, x)  # Delegates to wrapped kernel
    end
    
    # Initialize kernel states for all particles (clean dispatch)
    ker_states = map(starting_x, get_parameters(ker_parameters), metric_estimate) do x, p, Σ
        init_kernel_state(pop_kernel, x, p, Σ)  # Single call, returns single state
    end
    
    # Adaptive step loop
    chains = [chain_states]  # Store chain history for all particles
    n_steps = 0
    prev_msjd = 0.
    
    while true
        # Call population kernel once per step (owns the map internally)
        new_chain_states, accepted, γs, new_ker_states = pop_kernel(target, chain_states, ker_states)
        chain_states = new_chain_states
        ker_states = new_ker_states
        
        push!(chains, copy(chain_states))  # Record step for each particle
        n_steps += 1
        
        # Compute mean squared jump distance across population
        msjd = mean(zip(chain_states, chains[1])) do (cs_now, cs_init)
            invquad(Σg, cs_now.x - cs_init.x)
        end
        
        if abs(msjd - prev_msjd) < adapt_stability * prev_msjd || n_steps >= mcmc_steps
            break
        end
        prev_msjd = msjd
    end
    
    # Transpose chains: each entry is one particle's evolution
    chains = [
        (;states=[chains[step][i] for step in eachindex(chains)],
          γ=[...],  # Gather acceptance metrics per step
          n_accepts=...,  # Sum acceptances
          kernel_state=ker_states[i])
        for i in eachindex(chain_states)
    ]
    
else
    # Non-adaptive: fixed steps
    chain_states = stabilized_map(starting_x, map_func) do x
        init_chain_state(pop_kernel, target, x)
    end
    
    ker_states = map(starting_x, get_parameters(ker_parameters), metric_estimate) do x, p, Σ
        init_kernel_state(pop_kernel, x, p, Σ)
    end
    
    # Run pop_kernel for mcmc_steps iterations
    for step in 1:mcmc_steps
        new_chain_states, _, _, new_ker_states = pop_kernel(target, chain_states, ker_states)
        chain_states = new_chain_states
        ker_states = new_ker_states
    end
    
    chains = [
        (;states=[chain_states[i]],  # Only final state
          γ=[],
          n_accepts=0,
          kernel_state=ker_states[i])
        for i in eachindex(chain_states)
    ]
end
```

**Key pattern**: `init_kernel_state(pop_kernel, x, p, Σ)` is called once per particle in a map — clean, no temporary vectors, no indexing tricks.

---

## Part 5: Refactoring waste_free_smc

### Overview

`waste_free_smc()` has different population semantics:
- At each tempering step, it resamples to `n_starting` particles
- Each starting particle generates a chain of `chain_length` steps
- The `n_starting` chains are evolved independently

### Changes Needed

**Signature & Setup (add kernel normalization):**
```julia
function waste_free_smc(seq::AbstractDistributionSequence, ref_logdensity, initial_samples::AbstractMatrix;
                        mcmc_kernel::AbstractMCMCKernel = RWMH(),
                        metric_estimator::AbstractMetric = _default_metric_estimator(size(initial_samples)...),
                        resampler::AbstractResampler = ResidualResampler(),
                        n_starting = max(2, round(Int, cbrt(size(initial_samples, 2)))),
                        ker_parameters::Union{AbstractKernelParameters, Nothing} = nothing,
                        map_func = map,
                        callback = (_) -> false,
                        store_trace = true,
                        show_progress = true
                       )
    
    # NEW: Normalize kernel to population-based
    pop_kernel = if mcmc_kernel isa AbstractPopulationKernel
        mcmc_kernel
    else
        PopulationKernel(mcmc_kernel; map_func=map_func)
    end
    # ... rest unchanged
end
```

**MCMC Propagation Loop (replace mcmc_chain calls):**

Replace the current `mcmc_chain(mcmc_kernel, ...)` call with iteration via `pop_kernel`:

```julia
for _ in 1:10  # Retry loop for acceptance
    # Initialize chain states for n_starting particles
    initial_chain_states = stabilized_map(starting_x, map_func) do x
        init_chain_state(pop_kernel, target, x)
    end
    
    # Initialize kernel states (clean dispatch, no vector wrapping)
    initial_ker_states = map(starting_x, get_parameters(ker_parameters), metric_estimate) do x, p, Σ
        init_kernel_state(pop_kernel, x, p, Σ)
    end
    
    # Evolve chain_length steps via population kernel
    chain_states = initial_chain_states
    ker_states = initial_ker_states
    chain_history = [copy(chain_states)]  # Track states for each step
    
    for step in 1:chain_length - 1
        new_chain_states, accepted, _, new_ker_states = pop_kernel(target, chain_states, ker_states)
        chain_states = new_chain_states
        ker_states = new_ker_states
        push!(chain_history, copy(chain_states))
    end
    
    # Transpose to per-particle chains (same structure mcmc_chain produces)
    chains = [
        (;states=[chain_history[step][i] for step in eachindex(chain_history)],
          n_accepts=...,
          kernel_state=ker_states[i],
          γ=[...])
        for i in eachindex(initial_chain_states)
    ]
    
    # Check if acceptance rate is non-zero; break if so
    state.acceptance_rate = sum(c.n_accepts for c in chains) / ((chain_length - 1) * n_starting)
    if !iszero(state.acceptance_rate)
        break
    end
    improve_acceptance!(ker_parameters)
end
```

---

## Part 6: Changes to Supporting Files

### src/mcmc/mcmc.jl

1. Add `AbstractIndividualKernel{G}` and `AbstractPopulationKernel{G}` abstract type definitions
2. Update `AbstractAutoStep` inheritance (change parent from `AbstractMCMCKernel{G}` to `AbstractIndividualKernel{G}`)

No other changes needed — `init_kernel_state` and `init_chain_state` already dispatch correctly.

### src/mcmc/kernels.jl

1. Update `AbstractAutoStep` inheritance declaration
2. Add `PopulationKernel` struct and its methods (Part 3)
3. Update individual kernel inheritance declarations to use `AbstractIndividualKernel` instead of `AbstractMCMCKernel` (optional for clarity, but `AbstractIndividualKernel` is more precise)
4. All existing kernel definitions (RWMH, MALA, etc.) stay as-is

### src/utils.jl

No changes needed; `stabilized_map` already supports the generic interface.

### src/mcmc/kernel_parameters.jl

No changes needed; kernel parameter adaptation operates on chains post-hoc.

### src/metric_estimators.jl

No changes needed; `ParticleRepresentation` continues to work as before.

---

## Part 7: Architectural Trade-offs & Decisions

### Trade-off 1: State Management in PopulationKernel

**Decision**: PopulationKernel applies the individual kernel to each particle independently; each particle maintains its own kernel state.

**Rationale**: 
- Matches the per-particle state initialization pattern already in smc.jl
- Kernels with adaptive step-sizes need per-particle state (each may need different scales)
- Thread-safe and re-entrant if map_func is distributed

---

### Trade-off 2: Map Function Capture vs. Pass-Through

**Decision**: PopulationKernel captures `map_func` at construction: `PopulationKernel(kernel; map_func=map_func)`.

**Alternatives Rejected**:
- Pass map_func to every call: `pop_kernel(target, ...; map_func=...)`
  - Would require changing every kernel call site in smc.jl / waste_free_smc
  - Verbose and error-prone (easy to forget or pass inconsistently)

**Rationale**: 
- Simplifies smc.jl/waste_free_smc
- `map_func` is algorithm-level configuration, chosen once at smc() entry, not varied per-iteration

---

### Trade-off 3: Particle Access in DifferentialEvo

**Decision**: DifferentialEvo receives particles through the standard `Σ` parameter of `init_kernel_state`, captured in its kernel state.

**Rationale**:
- Existing mechanism already works; no special-casing needed
- When `ParticleRepresentation()` metric is used, it passes `Fill(samples, n)`, making particles available
- After refactor, PopulationKernel ensures particles are automatically available via metric estimation
- Keeps the architecture simple: DifferentialEvo needs no different treatment than other kernels

---

## Part 8: Testing & Validation Strategy

### Unit Tests

1. **PopulationKernel wrapper identity**:
   - Verify that `PopulationKernel(RWMH())` produces same chains as calling RWMH per-particle
   - Test with various `map_func` (map, identity, dummy implementations)

2. **Dispatch correctness**:
   - Verify `init_kernel_state(pop_kernel, x, p, Σ)` returns a single state (not a vector)
   - Verify `pop_kernel(target, [cs1, cs2], [ks1, ks2])` returns vectors of results

### Integration Tests

1. **smc() equivalence**:
   - Run smc with individual kernel vs. auto-wrapped population kernel
   - Verify identical chains and evidence estimates (up to randomness)

2. **waste_free_smc() equivalence**:
   - Same as above for waste_free_smc

3. **DifferentialEvo automatic wiring**:
   - Verify DifferentialEvo works in smc() with default metric estimators
   - Verify particles are correctly passed to DifferentialEvo's state

### Performance Tests

- Verify no regression in runtime or allocations vs. current smc.jl
- PopulationKernel overhead should be minimal (just dispatch + map)

---

## Part 9: Implementation Order & Critical Dependencies

**Phase 1: Type Hierarchy & PopulationKernel Struct** (1-2 hours)
- Add `AbstractIndividualKernel` and `AbstractPopulationKernel` to mcmc.jl
- Update `AbstractAutoStep` inheritance
- Implement `PopulationKernel` struct and methods in kernels.jl
- Update kernel inheritance declarations as needed
- Files: `src/mcmc/mcmc.jl`, `src/mcmc/kernels.jl`
- **Test locally**: Verify types load and dispatch works

**Phase 2: smc.jl Refactoring** (2-3 hours)
- Add kernel normalization at start of smc()
- Refactor MCMC propagation loop (both adaptive and fixed-step paths)
- Replace per-particle iteration patterns with direct pop_kernel calls
- Files: `src/smc.jl`
- **Test**: Run `test_smc_gaussian.jl`, verify identical results to main

**Phase 3: waste_free_smc Refactoring** (1-2 hours)
- Add kernel normalization
- Replace mcmc_chain calls with PopulationKernel iteration
- Files: `src/smc.jl` (waste_free_smc is in the same file)
- **Test**: Run `test_waste_free_smc.jl`, verify identical results

**Phase 4: Validation** (1 hour)
- Run full test suite (fast + slow)
- Verify DifferentialEvo works automatically
- Type stability check (`test_type_stability.jl`)
- Performance regression check (`benchmark/`)

---

## Part 10: Files & Dependencies Map

### Critical Files for Implementation

1. **src/mcmc/mcmc.jl** (~10 line changes)
   - Add `AbstractIndividualKernel{G}`, `AbstractPopulationKernel{G}` abstract types
   - Update `AbstractAutoStep` to inherit from `AbstractIndividualKernel`
   - **Dependencies**: None (defines base abstractions)

2. **src/mcmc/kernels.jl** (~50 lines added)
   - Add `PopulationKernel` struct definition
   - Implement `init_kernel_state(k::PopulationKernel, x, scale, Σ)`
   - Implement `(k::PopulationKernel)(target, chain_states, ker_states)` call operator
   - Update `AbstractAutoStep` inheritance declaration
   - **Dependencies**: mcmc.jl (for abstract types)

3. **src/smc.jl** (~50-80 lines changed)
   - Add kernel normalization to `pop_kernel` at start of `smc()`
   - Refactor MCMC propagation loop in both `smc()` and `waste_free_smc()`
   - Update kernel initialization and iteration blocks (both adaptive and fixed-step)
   - **Dependencies**: kernels.jl (for PopulationKernel)

### Test Files (Validate, Do Not Modify)

- `test/test_smc_gaussian.jl`: Verify smc() produces identical results
- `test/test_waste_free_smc.jl`: Verify waste_free_smc() produces identical results
- `test/test_mcmc_kernels.jl`: Verify individual kernel stationarity unchanged
- `test/test_type_stability.jl`: Verify no type regressions

---

## Part 11: Example Usage After Refactoring

```julia
# Backward compatible: Individual kernel auto-wrapped
result = smc(seq, ref_ld, samples; mcmc_kernel=MALA(), map_func=map)
# Internally: PopulationKernel(MALA(); map_func=map)

# Explicit population kernel with custom map
result = smc(seq, ref_ld, samples; mcmc_kernel=PopulationKernel(MALA(); map_func=pmap), ...)

# DifferentialEvo now works with any metric (auto-wired)
result = smc(seq, ref_ld, samples; mcmc_kernel=DifferentialEvo(), metric_estimator=ScalarAdaptation(), ...)

# Still works with explicit ParticleRepresentation
result = smc(seq, ref_ld, samples; mcmc_kernel=DifferentialEvo(), metric_estimator=ParticleRepresentation(), ...)
```

---

## Implementation Checklist

- [ ] Phase 1: Type hierarchy and PopulationKernel struct
- [ ] Phase 2: smc.jl refactoring
- [ ] Phase 3: waste_free_smc refactoring
- [ ] Phase 4: Full validation and testing

---

## Notes

- **Kernel state staleness**: In smc.jl, `kernel_state` is initialized once per tempering step (before the inner adaptive-steps loop), then reused across all steps. For DifferentialEvo, this means the particle snapshot is fixed for the whole inner loop — this is existing behavior and remains unchanged by the refactor.

- **No breaking changes**: All existing code using individual kernels continues to work. PopulationKernel is transparent to users — they can pass individual kernels and smc() wraps them automatically.

- **DifferentialEvo improvement**: After refactor, DifferentialEvo receives particle data automatically from any metric estimator (not just ParticleRepresentation), making it more flexible and intuitive to use.
