"""Learn an identified one-factor model using Gaussian channels and FFBS.

This is the intermediate test between a purely structural Gaussian model and
the mixed Gaussian/GEV PGAS model.  It demonstrates that a channel consists of

1. a pure reference channel defining the factor;
2. a smooth common factor driven by stochastic slope innovations;
3. horseshoe-regularized channel deviations;
4. baseline normalization of factor location;
5. separate baseline, shared, and dynamic-deviation output.

Run with

    python bucex_gaussian_identified_decomposition.py

No particle filter is used: conditional state paths are sampled exactly with
Gaussian forward-filtering backward-sampling (FFBS).
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_MONTHS = 300
RUN_FIT = True
QUICK_RUN = True
BASELINE_MONTHS = 60

# A loading anchor fixes the factor's scale and sign, but it does not by itself
# separate the factor path from an idiosyncratic random walk in the same anchor
# channel.  For this first channel lesson we therefore use a "pure reference":
# its LocalLevel remains in the model grammar required by FS-NCP, while its
# innovation SD is fixed at zero.  Set this to False to put all three channel
# innovations under the regularized horseshoe and explore the weaker dynamic
# identification problem.
IDENTIFY_WITH_PURE_REFERENCE = True

# The common factor is meant to be a smooth low-frequency climate trajectory.
# Fixing its direct level shock at zero leaves stochastic slope innovations:
#
#   f[t] = f[t-1] + beta[t-1]
#   beta[t] = beta[t-1] + q_beta * z[t].
#
# The idiosyncratic channel deviations remain local levels.  This difference
# in smoothness materially improves shared-versus-individual identification.
SMOOTH_COMMON_FACTOR = True

MCMC = (
    bx.MCMC(
        draws=300,
        warmup=500,
        chains=2,
        seed=2026,
        progress=True
    )
    if QUICK_RUN
    else bx.MCMC(
        draws=2000,
        warmup=2000,
        chains=4,
        seed=2026,
        progress=True,
    )
)


def scalar_summary(name, draws, truth):
    values = np.asarray(draws, dtype=float).reshape(-1)
    lower, median, upper = np.quantile(values, (0.05, 0.50, 0.95))
    return {
        "parameter": name,
        "truth": float(truth),
        "mean": float(np.mean(values)),
        "median": float(median),
        "lower90": float(lower),
        "upper90": float(upper),
        "median_error": float(median - truth),
    }


# ---------------------------------------------------------------------------
# 1. Define three channels
# ---------------------------------------------------------------------------
#
# reference:
#   anchors the factor loading at one and has no idiosyncratic innovation.
#   With IDENTIFY_WITH_PURE_REFERENCE=True, its innovation SD is also fixed at
#   zero during fitting; the channel then directly identifies the factor path.
#
# amplified:
#   responds 1.35 times as strongly to the common factor and also has a clear
#   channel-specific random walk.
#
# damped:
#   responds only 0.65 times as strongly and has a small idiosyncratic drift.
channels = (
    bx.Channel(
        "reference",
        bx.Gaussian(),
        components=(bx.LocalLevel(),),
        description="pure loading anchor for the common path",
    ),
    bx.Channel(
        "amplified",
        bx.Gaussian(),
        components=(bx.LocalLevel(),),
        description="strong common response plus individual drift",
    ),
    bx.Channel(
        "damped",
        bx.Gaussian(),
        components=(bx.LocalLevel(),),
        description="weak common response plus small individual drift",
    ),
)

common_factor = bx.Factor(
    "common",
    components=(
        bx.LocalLinearTrend(
            initial_level=0.0,
            initial_level_sd=0.0,  # identification: f[0] = 0
            initial_slope=0.0,
        ),
    ),
    loadings={
        "reference": 1.0,
        "amplified": bx.Loading.estimated(1.20),
        "damped": bx.Loading.estimated(0.80),
    },
    description="smooth shared signal driven by slope innovations",
)

model = bx.FactorModel(
    channels=channels,
    factors=(common_factor,),
    name="all-Gaussian channel-learning model",
)

assert model.all_gaussian
assert model.supports_fs_parameterization


# ---------------------------------------------------------------------------
# 2. Choose the data-generating truth
# ---------------------------------------------------------------------------
truth = {
    # Shared local-linear factor
    "sd.factor.common.level": 0.000,
    "sd.factor.common.slope": 0.00030,
    # Channel-specific local levels
    "sd.channel.reference.level": 0.000,
    "sd.channel.amplified.level": 0.006,
    "sd.channel.damped.level": 0.002,
    # Gaussian observation noise
    "sigma.reference": 0.10,
    "sigma.amplified": 0.14,
    "sigma.damped": 0.12,
    # Estimated loadings. The fixed reference loading is not a parameter.
    "loading.common.amplified": 1.35,
    "loading.common.damped": 0.65,
}

compiled = bx.compile_model(
    model,
    np.zeros((N_MONTHS, len(model.channel_names))),
)
state_index = {name: j for j, name in enumerate(compiled.state_names)}
initial_state = np.zeros(compiled.state_dim)
initial_state[state_index["factor.common.slope"]] = 0.0

simulation = bx.simulate(
    model,
    N_MONTHS,
    truth,
    initial_state=initial_state,
    seed=2025,
)

dates = pd.date_range("1995-01-01", periods=N_MONTHS, freq="MS")
data = pd.DataFrame(
    simulation.y,
    index=dates,
    columns=simulation.channel_names,
)

true_factor = simulation.states[1:, state_index["factor.common.level"]]
true_channel_levels = {
    channel: simulation.states[1:, state_index[f"channel.{channel}.level"]]
    for channel in model.channel_names
}
true_loadings = {
    "reference": 1.0,
    "amplified": truth["loading.common.amplified"],
    "damped": truth["loading.common.damped"],
}

# Factor location is normalized, not estimated scientifically: subtracting a
# draw-specific constant from f and adding lambda times that constant to c
# leaves every predictor exactly unchanged.  We define f=0 on average during
# the first BASELINE_MONTHS observations.
baseline_stop = min(BASELINE_MONTHS, N_MONTHS)
baseline_slice = slice(0, baseline_stop)
true_factor_shift = float(np.mean(true_factor[baseline_slice]))
true_factor_centered = true_factor - true_factor_shift
true_centered_intercepts = {
    channel: true_loadings[channel] * true_factor_shift
    for channel in model.channel_names
}

channel_table = pd.DataFrame(
    [
        {
            "channel": channel.name,
            "observation": channel.family,
            "loading": true_loadings[channel.name],
            "loading_status": (
                "fixed anchor" if channel.name == "reference" else "estimated"
            ),
            "idio_sd": truth[f"sd.channel.{channel.name}.level"],
            "idio_prior": (
                "fixed zero"
                if channel.name == "reference" and IDENTIFY_WITH_PURE_REFERENCE
                else "regularized horseshoe"
            ),
            "observation_sd": truth[f"sigma.{channel.name}"],
            "role": channel.description,
        }
        for channel in model.channels
    ]
).set_index("channel")

print("\nbucex loaded from:", bx.__file__)
print("\nCHANNEL DEFINITIONS\n", channel_table.to_string())
print("\nFIRST FIVE OBSERVATIONS\n", data.head().round(3).to_string())
print("\nSIMULATION SUMMARY\n", data.describe().round(3).to_string())
print(
    "\nFACTOR NORMALIZATION\n"
    f"The common factor averages zero over the first {baseline_stop} months."
)


# Plot the simulated series before fitting.
figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
data.plot(ax=axes[0], title="Three Gaussian observation channels")
axes[0].set_ylabel("observation")
axes[1].plot(
    dates,
    true_factor_centered,
    color="black",
    label="true baseline-normalized common factor",
)
axes[1].set(title="Data-generating shared factor", ylabel="factor change")
axes[1].legend()
figure.tight_layout()


# ---------------------------------------------------------------------------
# 3. Fit with FS-NCP and exact Gaussian FFBS
# ---------------------------------------------------------------------------
if RUN_FIT:
    if QUICK_RUN:
        print(
            "\nEXPLORATORY-RUN WARNING\n"
            "Two chains with 300 retained draws are suitable for learning and "
            "code testing, not final inference. Treat parameters with R-hat > "
            "1.01 or low ESS as unconverged. Set QUICK_RUN=False before "
            "assessing loading or variance recovery."
        )
    else:
        print(
            "\nLONG-RUN MODE\n"
            "Using four chains with 2000 warmup and 2000 retained draws."
        )
    fit_compiled = bx.compile_model(model, data)
    priors = bx.default_factor_priors(
        fit_compiled,
        profile="regularized_horseshoe",
    )

    process_priors = dict(priors.process)
    horseshoe_processes = list(priors.horseshoe_processes)
    horseshoe_scales = dict(priors.horseshoe.coefficient_scale)

    if IDENTIFY_WITH_PURE_REFERENCE:
        reference_process = "channel.reference.level"
        process_priors[reference_process] = bx.FixedSD(0.0)
        horseshoe_processes = [
            name for name in horseshoe_processes if name != reference_process
        ]
        horseshoe_scales.pop(reference_process, None)

    if SMOOTH_COMMON_FACTOR:
        process_priors["factor.common.level"] = bx.FixedSD(0.0)

    priors = replace(
        priors,
        process=process_priors,
        horseshoe=replace(
            priors.horseshoe,
            coefficient_scale=horseshoe_scales,
        )
        if horseshoe_scales
        else None,
        horseshoe_processes=tuple(horseshoe_processes),
        profile="identified_smooth_regularized_horseshoe",
    )

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        particles=bx.Particles(n=64),
        priors=priors,
        asis=True,
        mcmc=MCMC,
    )

    print("\nINFERENCE PLAN\n", fit.plan)
    assert fit.plan.engine == "ffbs"
    assert fit.plan.targets_exact_posterior

    diagnostics = fit.diagnostics()
    diagnostic_table = diagnostics["parameters"][[
        "rhat",
        "ess_bulk",
        "acceptance",
    ]]
    
    all_scalar_parameters = pd.DataFrame.from_dict(
        fit.static_summary(),
        orient="index",
    )

    print(
        "\nALL STORED SCALAR PARAMETERS\n",
        all_scalar_parameters.round(5).to_string(),
    )
    # Scientific parameters only: the signed FS coefficients and horseshoe
    # hyperparameters remain available through fit.static_summary(), but are
    # excluded here to keep the first Gaussian lesson readable.
    parameter_rows = [
        scalar_summary(
            "initial.factor.common.slope",
            fit.parameter("initial.factor.common.slope"),
            0.0,
        )
    ]
    for process_name in compiled.noise_names:
        name = f"sd.{process_name}"
        parameter_rows.append(
            scalar_summary(name, fit.parameter(name), truth[name])
        )
    for channel in model.channel_names:
        name = f"sigma.{channel}"
        parameter_rows.append(
            scalar_summary(name, fit.parameter(name), truth[name])
        )
        parameter_rows.append(
            scalar_summary(
                f"loading.common.{channel}",
                fit.loading_draws("common", channel, original_scale=True),
                true_loadings[channel],
            )
        )

    parameter_table = pd.DataFrame(parameter_rows).set_index("parameter")
    parameter_table = parameter_table.join(diagnostic_table, how="left")
    print(
        "\nSCIENTIFIC PARAMETER RECOVERY\n",
        parameter_table.round(5).to_string(),
    )

    horseshoe_targets = ", ".join(priors.horseshoe_processes)
    print(
        "\nDYNAMIC IDENTIFICATION AND SHRINKAGE\n"
        f"pure reference constraint = {IDENTIFY_WITH_PURE_REFERENCE}\n"
        f"smooth common factor (no direct level shock) = "
        f"{SMOOTH_COMMON_FACTOR}\n"
        f"reference idiosyncratic SD = "
        f"{np.median(fit.parameter('sd.channel.reference.level')):.6f}\n"
        f"regularized-horseshoe targets = {horseshoe_targets}"
    )

    raw_factor_draws = fit.factor("common")
    factor_shift_draws = raw_factor_draws[:, baseline_slice].mean(axis=1)
    factor_draws = raw_factor_draws - factor_shift_draws[:, None]
    factor_mean = factor_draws.mean(axis=0)
    factor_lower, factor_upper = np.quantile(
        factor_draws, (0.05, 0.95), axis=0
    )
    print(
        "\nBASELINE-NORMALIZED FACTOR RECOVERY\n"
        f"correlation = "
        f"{np.corrcoef(factor_mean, true_factor_centered)[0, 1]:.3f}\n"
        f"RMSE = "
        f"{np.sqrt(np.mean((factor_mean - true_factor_centered) ** 2)):.4f}\n"
        f"90% path coverage = "
        f"{np.mean((true_factor_centered >= factor_lower) & (true_factor_centered <= factor_upper)):.3f}"
    )

    # -----------------------------------------------------------------------
    # 4. Decompose each fitted channel correctly
    # -----------------------------------------------------------------------
    # fit.state("channel.<name>.level") is c[i] + alpha[i,t], not alpha[i,t].
    # After baseline normalization the exactly equivalent predictor is
    #
    #   eta[i,t] = c_baseline[i] + loading[i] * f_centered[t] + alpha[i,t],
    #
    # with
    #
    #   c_baseline[i] = c_raw[i] + loading[i] * factor_shift.
    #
    # The code below keeps these three scientific contributions separate and
    # checks their sum against fit.reconstructed_state().
    decomposition_rows = []
    decomposition_figure, decomposition_axes = plt.subplots(
        len(model.channel_names),
        3,
        figsize=(17, 9),
        sharex=True,
    )

    for row, channel in enumerate(model.channel_names):
        loading_draws = fit.loading_draws(
            "common", channel, original_scale=True
        )
        raw_intercept_draws = fit.parameter(f"intercept.{channel}")
        centered_intercept_draws = (
            raw_intercept_draws
            + loading_draws * factor_shift_draws
        )

        channel_level_draws = fit.state(f"channel.{channel}.level")
        alpha_draws = (
            channel_level_draws
            - raw_intercept_draws[:, None]
        )
        shared_draws = loading_draws[:, None] * factor_draws
        eta_from_parts = (
            centered_intercept_draws[:, None]
            + shared_draws
            + alpha_draws
        )
        eta_draws = fit.reconstructed_state(channel)
        reconstruction_error = float(
            np.max(np.abs(eta_from_parts - eta_draws))
        )
        if reconstruction_error > 1e-8:
            raise RuntimeError(
                f"Decomposition failed for {channel}: "
                f"maximum error={reconstruction_error:.3e}."
            )

        shared_mean = shared_draws.mean(axis=0)
        alpha_mean = alpha_draws.mean(axis=0)
        eta_mean = eta_draws.mean(axis=0)
        eta_lower, eta_upper = np.quantile(
            eta_draws,
            (0.05, 0.95),
            axis=0,
        )

        true_shared = true_loadings[channel] * true_factor_centered
        true_alpha = true_channel_levels[channel]
        true_intercept = true_centered_intercepts[channel]
        true_eta = simulation.eta[:, model.channel_names.index(channel)]

        intercept_lower, intercept_median, intercept_upper = np.quantile(
            centered_intercept_draws,
            (0.05, 0.50, 0.95),
        )
        decomposition_rows.append(
            {
                "channel": channel,
                "loading_truth": true_loadings[channel],
                "loading_median": np.median(loading_draws),
                "baseline_truth": true_intercept,
                "baseline_median": intercept_median,
                "baseline_lower90": intercept_lower,
                "baseline_upper90": intercept_upper,
                "shared_RMSE": np.sqrt(
                    np.mean((shared_mean - true_shared) ** 2)
                ),
                "deviation_RMSE": np.sqrt(
                    np.mean((alpha_mean - true_alpha) ** 2)
                ),
                "predictor_RMSE": np.sqrt(
                    np.mean((eta_mean - true_eta) ** 2)
                ),
                "predictor_correlation": np.corrcoef(
                    eta_mean,
                    true_eta,
                )[0, 1],
                "reconstruction_error": reconstruction_error,
            }
        )

        predictor_axis = decomposition_axes[row, 0]
        predictor_axis.scatter(
            dates,
            data[channel],
            s=5,
            alpha=0.22,
            label="observed",
        )
        predictor_axis.plot(
            dates,
            true_eta,
            color="black",
            label="true predictor",
        )
        predictor_axis.plot(
            dates,
            eta_mean,
            color="tab:red",
            label="posterior mean",
        )
        predictor_axis.fill_between(
            dates,
            eta_lower,
            eta_upper,
            color="tab:red",
            alpha=0.15,
            label="posterior 90% interval",
        )
        predictor_axis.set_title(f"{channel}: complete predictor")
        predictor_axis.set_ylabel("eta")

        shared_axis = decomposition_axes[row, 1]
        shared_axis.plot(
            dates,
            true_shared,
            color="black",
            label="true shared contribution",
        )
        shared_axis.plot(
            dates,
            shared_mean,
            color="tab:blue",
            label="estimated shared contribution",
        )
        shared_axis.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
        shared_axis.set_title(f"{channel}: loading x common factor")
        shared_axis.set_ylabel("shared change")

        alpha_axis = decomposition_axes[row, 2]
        alpha_axis.plot(
            dates,
            true_alpha,
            color="grey",
            linestyle="--",
            label="true dynamic deviation",
        )
        alpha_axis.plot(
            dates,
            alpha_mean,
            color="tab:orange",
            linestyle="--",
            label="estimated dynamic deviation",
        )
        alpha_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        alpha_axis.set_title(f"{channel}: alpha deviation only")
        alpha_axis.set_ylabel("idiosyncratic change")

    decomposition_axes[0, 0].legend(fontsize=8)
    decomposition_axes[0, 1].legend(fontsize=8)
    decomposition_axes[0, 2].legend(fontsize=8)
    decomposition_figure.tight_layout()

    decomposition_table = pd.DataFrame(decomposition_rows).set_index("channel")
    print(
        "\nBASELINE, SHARED AND DEVIATION RECOVERY\n",
        decomposition_table.round(5).to_string(),
    )

    # Fixed parameters legitimately have zero posterior SD and do not need an
    # ESS/R-hat warning.  Restrict the convergence screen to sampled scalars.
    sampled_names = all_scalar_parameters.index[
        all_scalar_parameters["sd"] > 1e-12
    ]
    sampled_diagnostics = diagnostic_table.loc[
        diagnostic_table.index.intersection(sampled_names)
    ]
    convergence_flags = sampled_diagnostics[
        (sampled_diagnostics["rhat"] > 1.01)
        | (sampled_diagnostics["ess_bulk"] < 100)
    ]
    print(
        "\nPARAMETERS REQUIRING LONGER CHAINS\n",
        "none"
        if convergence_flags.empty
        else convergence_flags.round(3).to_string(),
    )


plt.show()
