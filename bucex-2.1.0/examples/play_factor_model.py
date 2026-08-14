"""A small bucex 2.1 factor-model sandbox.

Run from the project directory with

    python examples/play_factor_model.py

The script simulates three centred temperature summaries and, by default,
runs a deliberately short PGAS fit.  Set RUN_FIT to False for simulation only.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


RUN_FIT = True
N_MONTHS = 120


def posterior_row(parameter, draws, true_value=np.nan):
    """One truth-versus-posterior row for a scalar parameter."""

    values = np.asarray(draws, dtype=float).reshape(-1)
    lower, median, upper = np.quantile(values, (0.05, 0.50, 0.95))
    known = np.isfinite(true_value)
    return {
        "parameter": parameter,
        "truth": float(true_value) if known else np.nan,
        "mean": float(np.mean(values)),
        "median": float(median),
        "lower90": float(lower),
        "upper90": float(upper),
        "median_error": float(median - true_value) if known else np.nan,
        "covers_truth": bool(lower <= true_value <= upper) if known else np.nan,
    }


# One Gaussian mean, one upper extreme and one lower extreme.  Each series has
# its own random-walk deviation.  Seasonality is omitted in this first test.
model = bx.FactorModel(
    channels=(
        bx.Channel("mean", bx.Gaussian(), components=(bx.LocalLevel(),)),
        bx.Channel("maximum", bx.GEV(), components=(bx.LocalLevel(),)),
        bx.Channel(
            "minimum",
            bx.GEV(),
            components=(bx.LocalLevel(),),
            tail="lower",
        ),
    ),
    factors=(
        bx.Factor(
            "common",
            components=(
                bx.LocalLinearTrend(
                    initial_level=0.0,
                    initial_level_sd=0.0,
                    initial_slope=0.0,
                ),
            ),
            loadings={
                "mean": 1.0,  # fixed anchor
                "maximum": bx.Loading.estimated(1.2),
                # Lower tails are sign-reversed internally.
                "minimum": bx.Loading.estimated(-0.8),
            },
        ),
    ),
    name="three-channel toy climate model",
)

assert model.supports_fs_parameterization


# These are the data-generating values.  Innovation scales are standard
# deviations per month.  The minimum loading is -0.8 internally and +0.8 on
# the original temperature orientation.
truth = {
    "sd.factor.common.level": 0.025,
    "sd.factor.common.slope": 0.0003,
    "sd.channel.mean.level": 0.010,
    "sd.channel.maximum.level": 0.025,
    "sd.channel.minimum.level": 0.005,
    "sigma.mean": 0.20,
    "sigma.maximum": 0.35,
    "xi.maximum": -0.10,
    "sigma.minimum": 0.30,
    "xi.minimum": -0.05,
    "loading.common.maximum": 1.20,
    "loading.common.minimum": -0.80,
}


# simulate() starts at zero unless an initial state is supplied.  Give the
# common factor a visible initial warming rate of 0.01 units per month.
compiled = bx.compile_model(
    model,
    np.zeros((N_MONTHS, len(model.channel_names))),
)
initial_state = np.zeros(compiled.state_dim)
factor_level_index = compiled.state_names.index("factor.common.level")
factor_slope_index = compiled.state_names.index("factor.common.slope")
initial_state[factor_slope_index] = 0.01

simulation = bx.simulate(
    model,
    N_MONTHS,
    truth,
    initial_state=initial_state,
    seed=123,
)

dates = pd.date_range("2000-01-01", periods=N_MONTHS, freq="MS")
data = pd.DataFrame(
    simulation.y,
    index=dates,
    columns=simulation.channel_names,
)
true_factor = simulation.states[1:, factor_level_index]

print("\nFirst 10 simulated observations:\n", data.head(10).round(3))
print("\nSimulation summary:\n", data.describe().round(3))
print("\nModel supports FS-NCP:", model.supports_fs_parameterization)
print("True original-scale loadings: mean=1.0, maximum=1.2, minimum=0.8")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
data.plot(ax=axes[0], title="Simulated observations")
axes[0].set_ylabel("centred temperature")
axes[1].plot(dates, true_factor, color="black", label="true common factor")
axes[1].set(title="Latent shared local-linear trend", ylabel="factor")
axes[1].legend()
fig.tight_layout()


if RUN_FIT:
    # This is a smoke test, not a publication-quality run.  Increase draws,
    # warmup, chains and particles after the code behaves as expected.
    print(
        "\nQUICK-RUN WARNING: 1 chain and 100 retained draws test the code, "
        "not posterior convergence. Expect noisy recovery, low ESS and "
        "possibly large R-hat values."
    )
    print(
        "Progress fields: ESSmin is the lowest particle ESS in the current "
        "PGAS update; ancestors is the mean number of distinct particle "
        "ancestors. Values should not repeatedly collapse toward 1."
    )
    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors="regularized_horseshoe",
        asis=True,
        mcmc=bx.MCMC(
            draws=100,
            warmup=100,
            chains=1,
            seed=456,
            progress=True,
        ),
        particles=bx.Particles(n=64, proposal="guided"),
    )

    print("\nInference plan:\n", fit.plan)

    # Every stored scalar, including the FS signed scales and regularized-
    # horseshoe hierarchy.  The non-negative sd.* rows are the scientifically
    # interpretable innovation scales; signed_sd.* are their FS auxiliaries.
    diagnostics = fit.diagnostics()
    all_parameters = pd.DataFrame.from_dict(
        fit.static_summary(), orient="index"
    ).join(
        diagnostics["parameters"][["rhat", "ess_bulk", "acceptance"]],
        how="left",
    )
    # The sampler proposes the signed FS coefficient.  Copy its acceptance
    # rate to the corresponding non-negative scientific SD row as well.
    for process_name in compiled.noise_names:
        sd_name = f"sd.{process_name}"
        signed_name = f"signed_sd.{process_name}"
        if signed_name in all_parameters.index:
            all_parameters.loc[sd_name, "acceptance"] = all_parameters.loc[
                signed_name, "acceptance"
            ]
    print(
        "\nAll stored scalar parameters:\n",
        all_parameters.round(4).to_string(),
    )

    # A cleaner table containing the scientific parameters and their known
    # simulation truths.  Loadings are mapped back to the original temperature
    # orientation, so the minimum loading is displayed as +0.8 rather than its
    # internal value -0.8.
    scientific_truth = {
        "initial.factor.common.level": 0.0,
        "initial.factor.common.slope": 0.01,
        "intercept.mean": 0.0,
        "intercept.maximum": 0.0,
        "intercept.minimum": 0.0,
        **truth,
    }
    scientific_names = [
        "initial.factor.common.level",
        "initial.factor.common.slope",
        "intercept.mean",
        "intercept.maximum",
        "intercept.minimum",
        *[f"sd.{name}" for name in compiled.noise_names],
        *compiled.observation_parameter_names,
    ]
    comparison_rows = [
        posterior_row(name, fit.parameter(name), scientific_truth[name])
        for name in scientific_names
    ]
    for channel in model.channel_names:
        internal_truth = (
            1.0
            if channel == "mean"
            else truth[f"loading.common.{channel}"]
        )
        sign = model.transform_signs[model.channel_names.index(channel)]
        comparison_rows.append(
            posterior_row(
                f"loading.common.{channel} [original]",
                fit.loading_draws("common", channel, original_scale=True),
                sign * internal_truth,
            )
        )
    comparison = pd.DataFrame(comparison_rows).set_index("parameter")
    print(
        "\nScientific parameters: truth versus posterior:\n",
        comparison.round(4).to_string(),
    )

    print("\nPGAS diagnostics:")
    for name, value in diagnostics["engine"].items():
        print(f"  {name}: {value:.3f}")

    factor_draws = fit.factor_draws("common")
    estimated_factor = factor_draws.mean(axis=0)
    factor_lower, factor_upper = np.quantile(factor_draws, (0.05, 0.95), axis=0)
    factor_recovery = {
        "RMSE": float(np.sqrt(np.mean((estimated_factor - true_factor) ** 2))),
        "correlation": float(np.corrcoef(estimated_factor, true_factor)[0, 1]),
        "90% path coverage": float(
            np.mean((true_factor >= factor_lower) & (true_factor <= factor_upper))
        ),
    }
    print("\nCommon-factor recovery:")
    for name, value in factor_recovery.items():
        print(f"  {name}: {value:.3f}")

    predictor_rows = []
    for channel_index, channel in enumerate(model.channel_names):
        estimated_eta = fit.channel_eta_draws(
            channel, original_scale=True
        ).mean(axis=0)
        true_eta = simulation.eta[:, channel_index]
        predictor_rows.append(
            {
                "channel": channel,
                "eta_RMSE": np.sqrt(np.mean((estimated_eta - true_eta) ** 2)),
                "eta_correlation": np.corrcoef(estimated_eta, true_eta)[0, 1],
            }
        )
    print(
        "\nReconstructed predictor recovery:\n",
        pd.DataFrame(predictor_rows).set_index("channel").round(3).to_string(),
    )

    axes[1].fill_between(
        dates,
        factor_lower,
        factor_upper,
        color="tab:red",
        alpha=0.18,
        label="posterior 90% interval",
    )
    axes[1].plot(
        dates,
        estimated_factor,
        color="tab:red",
        linewidth=1.5,
        label="posterior mean (short run)",
    )
    axes[1].legend()
    fig.tight_layout()


plt.show()
