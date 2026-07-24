# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TemperedSMC.jl is a Julia package implementing Sequential Monte Carlo (SMC) algorithms with support for:
- **Traditional SMC**: Standard tempered particle filter
- **Waste-Free SMC**: Dau and Chopin's waste-free variant for improved sample efficiency
- **Adaptive MCMC kernels**: Delayed rejection and auto-scaling step-size adaptation
- **Tempering sequences**: Automatic temperature scheduling using conditional effective sample size (CESS)

The package is primarily a research implementation for PhD work, focusing on sampling from complex target distributions and Bayesian model comparison.
No need to make pull request as this is solo work. Merge directly into main.

## Core Architecture

### Modular Design Pattern

The package uses three orthogonal, pluggable interfaces:

1. **Distribution Sequences** (`src/sequences/`)
   - `AbstractDistributionSequence`: Defines a sequence of distributions (e.g., tempered sequence from β=0 to 1)
   - `AbstractSequenceState{T}`: Tracks state during sequence traversal (e.g., `TemperedState` with β values and log-densities)
   - Key implementation: `AdaptiveTempering` with bisection-based CESS metric for computing next β
   - The sequence can be any ordered family of distributions; tempering is just one example

2. **MCMC Kernels** (`src/mcmc/`)
   - `AbstractMCMCKernel{G}`: Parametrized by `Val{G}` indicating gradient requirement (true/false)
   - Kernel calling convention: `(kernel::AbstractMCMCKernel)(target, chain_state, ker_state) → (new_chain_state, accepted::Bool, acceptance_metric, new_ker_state)`
   - Chain states: `ChainState{T,L}` (position + log-density) or `GradientChainState{T,L}` (adds gradients)
   - Kernel types: `RWMH`, `MALA`, `ULA`, `DifferentialEvo`, `DelayedRejection`, `AutoStepSize` (via `kernel_parameters.jl`)
   - Kernels manage their own state (e.g., step-size, metric) via `ker_state` parameter

3. **Metric Estimators** (`src/metric_estimators.jl`)
   - `AbstractMetric`: Estimates covariance for adaptive MCMC from particle samples
   - Used to initialize kernel state: `init_kernel_state(kernel, x, scale, Σ)`
   - Implementations: `DiagonalMetric`, `FullMetric`, with optional `FillArray` support for constant-across-starting-points metrics

### SMC Workflow

The main `smc()` function orchestrates these three interfaces:

1. **Initialization**: Compute initial log-densities and resampling weights from tempering sequence
2. **Loop**: For each temperature β in the sequence:
   - Update weights based on density change
   - Resampling if ESS falls below threshold
   - Run MCMC on each particle (in parallel via `map_func`)
   - Accumulate evidence for log-marginal-likelihood estimate
   - Adapt kernel parameters and metric based on acceptance/jump distance
3. **Output**: `SMCState` containing samples, weights, evidence, acceptance rates, and MCMC chain states

Key parameters:
- `mcmc_steps`: Number of MCMC steps per particle (adaptive if `adapt_mcmc_steps=true`)
- `resampling_α`: ESS threshold as fraction of N_particles for resampling
- `metric_estimator`: How to estimate covariance for adaptive kernels
- `ker_parameters`: Strategy for adapting kernel (e.g., `ScaleAdaptation`, `RMAdaptation`)

### Supporting Components

- **`utils.jl`**: Helper functions including `FullLogDensity` (product of reference × muliplying density) and `stabilized_map` (wraps failures)
- **`resampling.jl`**: Resampling methods (`ResidualResampler`, `SystematicResampler`, `MultinomialResampler`)
- **`factorized_logdensity.jl`**: Interface for product densities used in tempering
- **`MetaNumber`**: Wraps scalar values with metadata (e.g., original log-density before tempering), used to track information throughout the algorithm

## Commands

### Development Setup
```bash
julia --project
]up          # Update all dependencies
```

### Running the Package
```julia
using TemperedSMC
# Example: smc(seq, ref_logdensity, initial_samples; mcmc_kernel=..., metric_estimator=..., ...)
result = smc(adaptive_tempering, target, samples; mcmc_kernel=MALA(), ...)
```

### Testing
No formal test suite yet. Validation is typically done via:
- Example scripts in PhD notebooks / separate experiments
- Checking that MCMC chain diagnostics match theoretical expectations
- Comparing SMC evidence estimates against known benchmarks

To verify basic functionality after changes:
```julia
using TemperedSMC, Distributions, LogDensityProblems
# Create a simple target and run SMC with default settings
```

## Key Design Decisions & TODOs

### Current TODOs 

1. **Remove samples from SMCState** (`src/smc.jl:1`)
   - Samples are redundant with chain states; consider consolidating

2. **Add timmings to the state and return information**

3. **Add benchmark of the cost of the implementation per logdensity call**

4. **Design and implement factorized logdensity interface**
    - method to get all logdensity
    - method to get one log density value
    - method to get multiply, specified logdensity values
    - implement mutating version of the methods that return vectors

5. **Design and implement Population-based kernels**
    - Add abstract subtypes of `AbstractMCMCKernel` for individual and population based kernels
    - Make a common interface so that they can be used seemly within smc and waste_free_smc

6. **Design and implement collective and individual implementations of Gibbs**

7. **Generalize the ad hoc handling of 0 acceptance rate**

7. **Design and Implement adaptive steps**
    - Compare current implementation with WFSMC paper and the Particles.py implementation
    - Design it to be compatible with smc and waste_free_smc

8. **Implement IBIS** (`src/TemperedSMC.jl:37`)
   - Importance-Batch-Importance-Sampling for sequential inference without tempering

9. **Implement Stein variational gradient descent as a collective kernel**

10. **Look into transport maps**
    - eg normalizing flows 

### Historical Context

Recent major work (git log):
- **Removed internal Base.result_types reliance** (f51a39c): No longer depends on undocumented Base internals
- **Chain state initialization fixes** (bd9de95): Proper setup for gradient vs. non-gradient kernels
- **Bug fixes in MCMC kernels** (b998b6a): Corrected acceptance logic and parameter updates
- **Kernel parameter refactoring** (9bc0005): Separated parameters from kernel logic to enable strategy pluggability

### Key Patterns

- **Dual-mode kernels**: All kernels support `Val{false}` (no gradients, cheaper) and `Val{true}` (with gradients, more efficient)
- **Lazy adaptation**: Kernel and metric adaptation happen post-hoc on acceptance/jump distance; no online preconditioning during kernel initialization
- **MetaNumber tracking**: Log-densities wrapped to preserve information (e.g., unnormalized density before tempering) for later analysis
- **Resampling callbacks**: Via optional `callback` parameter in `smc()`; allows external logging/monitoring
- **CESS/ESS Normalization**: CESS and ESS metrics are normalized relative to N (in [0, 1] range), not absolute (in [1, N] range). This is by design—the adaptive tempering and resampling use relative values (fractions of sample size), which is cleaner for algorithms that should be invariant to N. The bisection target `α` (default 0.8) operates on this normalized scale.

## Dependencies

Key scientific packages:
- **LogDensityProblems.jl**: Interface for providing log-densities and gradients (supports autodiff + hand-written)
- **CovarianceEstimation.jl**: Regularized covariance estimation for metric initialization
- **Distributions.jl, StatsBase.jl**: Standard statistical types and methods
- **LinearAlgebra, Random**: Standard library for numerical ops and RNG

See `Project.toml` for full dependency tree and versions.

## Code Navigation

### Entry Points
- `smc()`: Main algorithm (allocates and runs SMC iterations)
- `waste_free_smc()`: Waste-free variant (exported but not yet implemented in source)

### By Responsibility
- **Sequence logic**: `src/sequences/adaptive_tempering.jl` (tempering schedule), `src/sequences/sequences.jl` (interface)
- **Kernel logic**: `src/mcmc/kernels.jl` (RWMH, MALA, ULA, DE), `src/mcmc/kernel_parameters.jl` (adaptation rules)
- **Chain tracking**: `src/mcmc/chain.jl` (chain iterations and state management)
- **Metric estimation**: `src/metric_estimators.jl` (covariance from samples)
- **Algorithm**: `src/smc.jl` (main loop and orchestration)

## Common Edits

### Adding a New MCMC Kernel
1. Create a new struct `MyKernel <: AbstractMCMCKernel{Val{use_grad}}`
2. Implement `init_kernel_state(::MyKernel, x, scale, Σ)` → kernel-specific state
3. Implement `(k::MyKernel)(target, chain_state, ker_state)` → `(new_chain_state, accepted, metric, new_ker_state)`
4. Optionally override `usesgrad()` if different from default
5. Export in `src/TemperedSMC.jl`

### Changing Kernel Adaptation Strategy
- Modify `src/mcmc/kernel_parameters.jl` (adaptation rules live here)
- Update `ker_parameters` argument handling in `src/smc.jl` to apply the new strategy
- Future: Use AbstractKernelAdapt interface (currently TODO)

### Modifying Temperature Schedule
- Edit `_next_β()` in `src/sequences/adaptive_tempering.jl` or create a new `AbstractDistributionSequence` subtype
- The algorithm is generalized; any monotonic sequence of densities works (not just tempering)

## Evidence Accumulation

The log-evidence (log marginal likelihood) is accumulated only at **resampling steps**. This follows equation (10.3) in Chopin 2020, specifically the case when resampling occurs:

```
log(t_t^N) = log((1/N) ∑_n w_n^t) = logsumexp(lw) - log(N)
```

where `lw` contains the unnormalized incremental log-weights. The non-resampling case (a more complex formula in Chopin 2020) is not accumulated in the current implementation; this is a deliberate design choice that simplifies the algorithm while remaining mathematically sound.

## Debugging Tips

- **NaN/Inf in chain**: Check that log-densities are finite for the target; added NaN checking in `iterate_mcmc` (85c6247)
- **Acceptance too low**: Metric estimator may be poor in high dimensions; try increasing initial samples or using a cheaper kernel (RWMH vs. MALA)
- **ESS collapse**: Indicates resampling threshold is too loose; lower `resampling_α` (currently 0.5 by default)
- **Evidence estimates unstable**: Common in misspecified models; ensure proposal (initial samples) has reasonable overlap with target. Note: evidence is only accumulated at resampling steps.

## References

Academic foundation (papers in `references/`):
- **Chopin (2020)**: Introduction to SMC; covers adaptive tempering and CESS metric
- **Dau & Chopin (2022)**: Waste-free SMC; key motivation for package design
- **Zhou et al. (2016)**: Adaptive model comparison; CESS metric derivation
- **Fearnhead (2013)**: Adaptive MCMC within SMC; foundation for kernel adaptation strategies
