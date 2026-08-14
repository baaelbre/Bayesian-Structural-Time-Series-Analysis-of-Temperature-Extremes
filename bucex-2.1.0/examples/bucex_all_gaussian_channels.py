"""Learn a one-factor model using Gaussian channels and exact FFBS.

This is the intermediate test between a purely structural Gaussian model and
the mixed Gaussian/GEV PGAS model.  It demonstrates that a channel consists of

1. an observation distribution;
2. channel-specific latent components;
3. one loading for each shared factor.

Run with

    python examples/bucex_all_gaussian_channels.py

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

# A loading anchor fixes the factor's scale and sign, but it does not by itself
# separate the factor path from an idiosyncratic random walk in the same anchor
# channel.  For this first channel lesson we therefore use a "pure reference":
# its LocalLevel remains in the model grammar required by FS-NCP, while its
# innovation SD is fixed at zero.  Set this to False to put all three channel
# innovations under the regularized horseshoe and explore the weaker dynamic
# identification problem.
IDENTIFY_WITH_PURE_REFERENCE = True

# This is still an exploratory run.  For a formal recovery experiment use
# roughly draws=2000, warmup=2000, chains=4.
MCMC = bx.MCMC(
    draws=300,
    warmup=500,
    chains=2,
    seed=2026,
    progress=True,
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
    description="shared smooth signal",
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
    "sd.factor.common.level": 0.025,
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


# Plot the simulated series before fitting.
figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
data.plot(ax=axes[0], title="Three Gaussian observation channels")
axes[0].set_ylabel("observation")
axes[1].plot(dates, true_factor, color="black", label="true common factor")
axes[1].set(title="Data-generating shared factor", ylabel="factor")
axes[1].legend()
figure.tight_layout()


# ---------------------------------------------------------------------------
# 3. Fit with FS-NCP and exact Gaussian FFBS
# ---------------------------------------------------------------------------
if RUN_FIT:
    print(
        "\nEXPLORATORY-RUN WARNING\n"
        "Two chains with 300 retained draws are suitable for learning and "
        "code testing, not final inference. Treat parameters with R-hat > "
        "1.01 or low ESS as unconverged; increase warmup, draws and chains "
        "before scientific reporting."
    )
    fit_compiled = bx.compile_model(model, data)
    priors = bx.default_factor_priors(
        fit_compiled,
        profile="regularized_horseshoe",
    )
    if IDENTIFY_WITH_PURE_REFERENCE:
        reference_process = "channel.reference.level"
        process_priors = dict(priors.process)
        process_priors[reference_process] = bx.FixedSD(0.0)
        horseshoe_processes = tuple(
            name for name in priors.horseshoe_processes
            if name != reference_process
        )
        horseshoe = replace(
            priors.horseshoe,
            coefficient_scale={
                name: scale
                for name, scale in priors.horseshoe.coefficient_scale.items()
                if name != reference_process
            },
        )
        priors = replace(
            priors,
            process=process_priors,
            horseshoe=horseshoe,
            horseshoe_processes=horseshoe_processes,
            profile="identified_regularized_horseshoe",
        )

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
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

    print(
        "\nALL STORED SCALAR PARAMETERS\n",
        fit.static_summary().round(5).to_string(),
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
        f"reference idiosyncratic SD = "
        f"{np.median(fit.parameter('sd.channel.reference.level')):.6f}\n"
        f"regularized-horseshoe targets = {horseshoe_targets}"
    )

    factor_draws = fit.factor("common")
    factor_mean = factor_draws.mean(axis=0)
    factor_lower, factor_upper = np.quantile(
        factor_draws, (0.05, 0.95), axis=0
    )
    print(
        "\nFACTOR RECOVERY\n"
        f"correlation = {np.corrcoef(factor_mean, true_factor)[0, 1]:.3f}\n"
        f"RMSE = {np.sqrt(np.mean((factor_mean - true_factor) ** 2)):.4f}\n"
        f"90% path coverage = "
        f"{np.mean((true_factor >= factor_lower) & (true_factor <= factor_upper)):.3f}"
    )

    # -----------------------------------------------------------------------
    # 4. Decompose each fitted channel
    # -----------------------------------------------------------------------
    # For channel i:
    #
    #   eta[i,t] = loading[i] * factor[t] + channel_level[i,t].
    #
    # The channel level is the semantic centered state c[i] + alpha[i,t].
    decomposition_rows = []
    decomposition_figure, decomposition_axes = plt.subplots(
        len(model.channel_names),
        2,
        figsize=(13, 9),
        sharex=True,
    )

    for row, channel in enumerate(model.channel_names):
        loading_draws = fit.loading_draws(
            "common", channel, original_scale=True
        )
        shared_draws = loading_draws[:, None] * factor_draws
        channel_draws = fit.state(f"channel.{channel}.level")
        eta_draws = fit.reconstructed_state(channel)

        shared_mean = shared_draws.mean(axis=0)
        channel_mean = channel_draws.mean(axis=0)
        eta_mean = eta_draws.mean(axis=0)
        eta_lower, eta_upper = np.quantile(eta_draws, (0.05, 0.95), axis=0)

        true_shared = true_loadings[channel] * true_factor
        true_individual = true_channel_levels[channel]
        true_eta = simulation.eta[:, model.channel_names.index(channel)]

        decomposition_rows.append(
            {
                "channel": channel,
                "loading_truth": true_loadings[channel],
                "loading_median": np.median(loading_draws),
                "shared_RMSE": np.sqrt(
                    np.mean((shared_mean - true_shared) ** 2)
                ),
                "individual_RMSE": np.sqrt(
                    np.mean((channel_mean - true_individual) ** 2)
                ),
                "predictor_RMSE": np.sqrt(np.mean((eta_mean - true_eta) ** 2)),
                "predictor_correlation": np.corrcoef(eta_mean, true_eta)[0, 1],
            }
        )

        left = decomposition_axes[row, 0]
        left.scatter(dates, data[channel], s=5, alpha=0.22, label="observed")
        left.plot(dates, true_eta, color="black", label="true predictor")
        left.plot(dates, eta_mean, color="tab:red", label="posterior mean")
        left.fill_between(
            dates,
            eta_lower,
            eta_upper,
            color="tab:red",
            alpha=0.15,
            label="posterior 90% interval",
        )
        left.set_title(f"{channel}: complete predictor")
        left.set_ylabel("eta")

        right = decomposition_axes[row, 1]
        right.plot(
            dates,
            true_shared,
            color="black",
            label="true shared contribution",
        )
        right.plot(
            dates,
            shared_mean,
            color="tab:blue",
            label="estimated shared contribution",
        )
        right.plot(
            dates,
            true_individual,
            color="grey",
            linestyle="--",
            label="true individual contribution",
        )
        right.plot(
            dates,
            channel_mean,
            color="tab:orange",
            linestyle="--",
            label="estimated individual contribution",
        )
        right.set_title(f"{channel}: shared versus individual")

    decomposition_axes[0, 0].legend(fontsize=8)
    decomposition_axes[0, 1].legend(fontsize=8)
    decomposition_figure.tight_layout()

    decomposition_table = pd.DataFrame(decomposition_rows).set_index("channel")
    print(
        "\nCHANNEL DECOMPOSITION RECOVERY\n",
        decomposition_table.round(4).to_string(),
    )


plt.show()
