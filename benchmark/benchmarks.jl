# Benchmark suite for TemperedSMC implementation performance
# Measures per-step wall time and cost comparison for adaptive vs. fixed MCMC step counts
#
# Setup (one-time):
#   julia --project=benchmark -e 'import Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
#
# Run:
#   julia --project=benchmark benchmark/benchmarks.jl
#
# Outputs a table comparing all kernels (RWMH, MALA, ULA, DifferentialEvo, PathDelayedRejection, AutoStepMALA, AutoStepRWMH)
# in both fixed (adapt_mcmc_steps=false) and adaptive (adapt_mcmc_steps=true) modes on a fixed 2D Gaussian problem.

using BenchmarkTools
using TemperedSMC
using Distributions
using LinearAlgebra, Statistics, Random
using LogDensityProblems
using Printf

LD = LogDensityProblems

include(joinpath(@__DIR__, "..", "test", "test_utils.jl"))

# Fixed test configuration
const PROBLEM = get_test_problem()
const GROUND_TRUTH = get_test_ground_truth()
const N_SAMPLES = 600
const MCMC_STEPS = 25
const ADAPT_STABILITY = 0.01

# Kernel configurations: (name, kernel, metric_estimator_override)
# metric_estimator_override is used if non-nothing
KERNELS = [
    ("RWMH", TemperedSMC.RWMH(), nothing),
    ("MALA", TemperedSMC.MALA(), nothing),
    ("ULA", TemperedSMC.ULA(), nothing),
    ("DifferentialEvo", TemperedSMC.DifferentialEvo(), TemperedSMC.ParticleRepresentation()),
    ("PathDelayedRejection", TemperedSMC.PathDelayedRejection(), nothing),
    ("AutoStepMALA", TemperedSMC.AutoStepMALA(), nothing),
    ("AutoStepRWMH", TemperedSMC.AutoStepRWMH(), nothing),
]

function run_benchmark(kernel_name, kernel, metric_est, adapt_steps)
    prior_ld, likelihood_ld, prior_dist = PROBLEM
    seq = TemperedSMC.AdaptiveTempering(likelihood_ld)

    # Benchmark with fresh samples each trial
    result = @benchmark begin
        initial_samples = rand($prior_dist, $N_SAMPLES)
        smc_result = if $metric_est !== nothing
            TemperedSMC.smc(
                $seq, $prior_ld, initial_samples;
                mcmc_kernel=$kernel,
                metric_estimator=$metric_est,
                mcmc_steps=$MCMC_STEPS,
                adapt_mcmc_steps=$adapt_steps,
                adapt_stability=$ADAPT_STABILITY,
                show_progress=false,
                store_trace=true
            )
        else
            TemperedSMC.smc(
                $seq, $prior_ld, initial_samples;
                mcmc_kernel=$kernel,
                mcmc_steps=$MCMC_STEPS,
                adapt_mcmc_steps=$adapt_steps,
                adapt_stability=$ADAPT_STABILITY,
                show_progress=false,
                store_trace=true
            )
        end
        smc_result
    end evals=1 samples=5 seconds=60

    # Single reference run to extract detailed instrumentation
    initial_samples = rand(prior_dist, N_SAMPLES)
    trace = if metric_est !== nothing
        TemperedSMC.smc(
            seq, prior_ld, initial_samples;
            mcmc_kernel=kernel,
            metric_estimator=metric_est,
            mcmc_steps=MCMC_STEPS,
            adapt_mcmc_steps=adapt_steps,
            adapt_stability=ADAPT_STABILITY,
            show_progress=false,
            store_trace=true
        )
    else
        TemperedSMC.smc(
            seq, prior_ld, initial_samples;
            mcmc_kernel=kernel,
            mcmc_steps=MCMC_STEPS,
            adapt_mcmc_steps=adapt_steps,
            adapt_stability=ADAPT_STABILITY,
            show_progress=false,
            store_trace=true
        )
    end

    # Compute per-step metrics from trace
    total_t_mcmc = sum(s.t_mcmc for s in trace)
    total_n_steps = sum(s.n_steps for s in trace)
    mean_n_steps = mean(s.n_steps for s in trace)
    total_t_metric = sum(s.t_metric for s in trace)
    total_t_adapt = sum(s.t_adapt for s in trace)
    total_t_reweight = sum(s.t_reweight for s in trace)
    total_wall_time = total_t_mcmc + total_t_metric + total_t_adapt + total_t_reweight

    seconds_per_step = total_n_steps > 0 ? total_t_mcmc / total_n_steps : 0.0

    pct_mcmc = total_wall_time > 0 ? 100 * total_t_mcmc / total_wall_time : 0.0
    pct_metric = total_wall_time > 0 ? 100 * total_t_metric / total_wall_time : 0.0
    pct_adapt = total_wall_time > 0 ? 100 * total_t_adapt / total_wall_time : 0.0

    # Extract accuracy metrics from final state
    final_state = trace[end]
    acceptance_rate = final_state.acceptance_rate
    log_evidence_error = abs(final_state.log_evidence - GROUND_TRUTH[3])

    return (
        result=result,
        seconds_per_step=seconds_per_step,
        mean_n_steps=mean_n_steps,
        acceptance_rate=acceptance_rate,
        log_evidence_error=log_evidence_error,
        pct_mcmc=pct_mcmc,
        pct_metric=pct_metric,
        pct_adapt=pct_adapt
    )
end

println("\n" * "="^150)
println("Benchmark: TemperedSMC Implementation Performance")
println("Problem: 2D conjugate Gaussian | Particles: $N_SAMPLES | Fixed MCMC steps cap: $MCMC_STEPS")
println("="^150 * "\n")

# Results storage: kernel_name -> Dict(:fixed => results, :adaptive => results)
all_results = Dict()

# Run all kernels in both modes
for (kernel_name, kernel, metric_est) in KERNELS
    println("Benchmarking $kernel_name...")

    all_results[kernel_name] = Dict()

    try
        # Fixed steps
        fixed_results = run_benchmark(kernel_name, kernel, metric_est, false)
        all_results[kernel_name][:fixed] = fixed_results
        println("  Fixed steps: done")
    catch e
        println("  Fixed steps: ERROR - $(e)")
        continue
    end

    try
        # Adaptive steps
        adaptive_results = run_benchmark(kernel_name, kernel, metric_est, true)
        all_results[kernel_name][:adaptive] = adaptive_results
        println("  Adaptive steps: done\n")
    catch e
        println("  Adaptive steps: ERROR - $(e)\n")
        # Still save fixed result even if adaptive fails
    end
end

# Print results table
println("\n" * "="^150)
println("DETAILED RESULTS")
println("="^150 * "\n")

println("Format: kernel | mode | median time (ms) | allocations | mean n_steps | s/step | %MCMC | %metric | %adapt | accept rate | log-evid error\n")

for (kernel_name, kernel, metric_est) in KERNELS
    if !haskey(all_results, kernel_name) || !haskey(all_results[kernel_name], :fixed)
        println("$kernel_name: SKIPPED (benchmark failed)\n")
        continue
    end

    results = all_results[kernel_name]

    # Fixed mode
    fixed = results[:fixed]
    btime_ms = median(fixed[:result]).time / 1e6
    allocs = median(fixed[:result]).allocs
    println(
        @sprintf(
            "%-20s | %-7s | %10.3f ms | %12d | %10.2f | %8.6f | %6.1f%% | %7.1f%% | %6.1f%% | %8.4f | %8.4f",
            kernel_name, "fixed", btime_ms, allocs,
            fixed[:mean_n_steps], fixed[:seconds_per_step],
            fixed[:pct_mcmc], fixed[:pct_metric], fixed[:pct_adapt],
            fixed[:acceptance_rate], fixed[:log_evidence_error]
        )
    )

    # Adaptive mode (if available)
    if haskey(results, :adaptive)
        adaptive = results[:adaptive]
        btime_ms = median(adaptive[:result]).time / 1e6
        allocs = median(adaptive[:result]).allocs
        println(
            @sprintf(
                "%-20s | %-7s | %10.3f ms | %12d | %10.2f | %8.6f | %6.1f%% | %7.1f%% | %6.1f%% | %8.4f | %8.4f",
                kernel_name, "adaptive", btime_ms, allocs,
                adaptive[:mean_n_steps], adaptive[:seconds_per_step],
                adaptive[:pct_mcmc], adaptive[:pct_metric], adaptive[:pct_adapt],
                adaptive[:acceptance_rate], adaptive[:log_evidence_error]
            )
        )

        # Per-step comparison summary
        fixed_sps = results[:fixed][:seconds_per_step]
        adaptive_sps = results[:adaptive][:seconds_per_step]
        ratio = fixed_sps > 0 ? adaptive_sps / fixed_sps : 1.0
        diff = adaptive_sps - fixed_sps

        println(
            @sprintf(
                "%-20s | %-7s | Per-step ratio: %.3f (adaptive slower by %.6f s/step)",
                kernel_name, "ratio", ratio, diff
            )
        )
    else
        println("%-20s | %-7s | (adaptive mode failed)", kernel_name)
    end
    println()
end

println("="^150)
