"""Multi-chain and engine-specific diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


Array = np.ndarray


def _finite_mean(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_median(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _split_chains(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (chains, draws).")
    half = values.shape[1] // 2
    if half < 2:
        return values
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)


def _rank_normalize(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1)
    ranks = rankdata(flat, method="average")
    return norm.ppf((ranks - 0.375) / (flat.size + 0.25)).reshape(values.shape)


def _basic_rhat(values: Array) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.nan
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(values, axis=1), ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else np.inf
    variance = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance / within))


def rhat(values: Array) -> float:
    """Rank-normalized split R-hat with the folded-tail diagnostic."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # R-hat compares within- and between-chain variation.  It is not one
        # when both are zero; it is mathematically undefined.
        return np.nan
    split = _split_chains(values)
    if split.shape[0] < 2 or split.shape[1] < 2:
        return np.nan
    bulk = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - float(np.median(split)))
    tail = _basic_rhat(_rank_normalize(folded))
    return float(max(bulk, tail))


def _autocovariance(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    n = values.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n]
    return covariance / np.arange(n, 0, -1)


def ess_bulk(values: Array) -> float:
    """Rank-normalized split-chain bulk effective sample size."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # A constant sampled indicator may represent posterior structural
        # certainty, but it supplies no autocorrelation information.
        return np.nan
    values = _rank_normalize(_split_chains(values))
    chains, draws = values.shape
    if draws < 3:
        return float(chains * draws)
    autocov = np.vstack([_autocovariance(chain) for chain in values])
    within = float(np.mean(autocov[:, 0]))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1)) if chains > 1 else 0.0
    variance_plus = (draws - 1.0) / draws * within + between / draws
    if variance_plus <= 0.0:
        return float(chains * draws)
    rho = np.ones(draws)
    for lag in range(1, draws):
        rho[lag] = 1.0 - (within - np.mean(autocov[:, lag])) / variance_plus
    pair_sums: list[float] = []
    for lag in range(1, draws - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0.0:
            break
        pair_sums.append(float(pair))
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    positive_sum = float(np.sum(pair_sums))
    return float(
        min(
            chains * draws,
            chains * draws / max(1.0 + 2.0 * positive_sum, 1e-12),
        )
    )


def posterior_pit(fit) -> Array:
    """In-sample posterior PIT values.

    These values reuse observations to estimate the latent state and should be
    treated as a posterior-predictive check, not as forecast calibration.
    Prefer ``bx.leave_future_out(...).pit_diagnostics()`` for validation.
    """
    if getattr(fit, "is_multiseries_model", False):
        eta = fit.eta_draws(original_scale=False)
        output = np.full((fit.n_time, len(fit.channel_names)), np.nan)
        for index, channel in enumerate(fit.model.channels):
            sigma = fit.parameter(f"sigma.{channel.name}")[:, None]
            xi = (
                fit.parameter(f"xi.{channel.name}")[:, None]
                if channel.family == "gev"
                else None
            )
            cdf = channel.observation.cdf(
                fit.y[None, :, index],
                eta[:, :, index],
                sigma=sigma,
                xi=xi,
            )
            if float(fit.transform_sign[index]) < 0.0:
                cdf = 1.0 - cdf
            output[:, index] = np.nanmean(cdf, axis=0)
        return output
    eta = fit.eta_draws(original_scale=False)
    sigma = fit.parameter("sigma")[:, None]
    xi = fit.parameter("xi")[:, None] if fit.family == "gev" else None
    cdf = fit.model.observation.cdf(fit.y[None, :], eta, sigma=sigma, xi=xi)
    if float(fit.transform_sign) < 0.0:
        cdf = 1.0 - cdf
    return np.nanmean(cdf, axis=0)


def fit_diagnostics(fit):
    acceptance = fit.sampler_diagnostics.get("acceptance", {})
    rows = []
    for name, values in fit.parameter_draws.items():
        if values.ndim != 2:
            continue
        constant = bool(
            values.size > 0 and np.all(values == values.reshape(-1)[0])
        )
        row = {
            "parameter": name,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "rhat": rhat(values),
            "ess_bulk": ess_bulk(values),
            "acceptance": _finite_mean(acceptance[name]) if name in acceptance else np.nan,
            "constant": constant,
            "diagnostic": (
                "constant draw; R-hat and ESS undefined"
                if constant
                else "sampled"
            ),
        }
        rows.append(row)
    try:
        import pandas as pd

        table = pd.DataFrame(rows).set_index("parameter")
    except ImportError:
        table = rows
    metrics = fit.sampler_diagnostics.get("draw_metrics", {})
    engine: dict[str, float] = {}
    if fit.plan.engine == "laplace":
        engine = {
            "convergence_rate": _finite_mean(metrics["laplace_converged"]),
            "median_iterations": _finite_median(metrics["laplace_iterations"]),
            "median_relative_change": _finite_median(
                metrics["laplace_relative_change"]
            ),
            "mean_support_rejections": _finite_mean(
                metrics["laplace_support_rejections"]
            ),
        }
    elif fit.plan.engine == "pgas":
        engine = {
            "median_min_particle_ess": _finite_median(
                metrics["particle_min_ess"]
            ),
            "mean_unique_ancestors": _finite_mean(
                metrics["particle_mean_unique_ancestors"]
            ),
            "path_change_rate": _finite_mean(
                metrics["particle_path_changed"]
            ),
            "mean_changed_fraction": _finite_mean(
                metrics["particle_changed_fraction"]
            ),
        }
        if bool(fit.metadata.get("structural_ssvs", False)):
            engine.update(
                ssvs_model_move_acceptance=_finite_mean(
                    metrics.get("ssvs_model_move_accepted", np.asarray([]))
                ),
                ssvs_proposed_change_rate=_finite_mean(
                    metrics.get("ssvs_model_proposed_change", np.asarray([]))
                ),
                ssvs_median_log_acceptance_ratio=_finite_median(
                    metrics.get(
                        "ssvs_model_log_acceptance_ratio", np.asarray([])
                    )
                ),
                ssvs_mean_elliptical_slice_steps=_finite_mean(
                    metrics.get("fs_elliptical_slice_steps", np.asarray([]))
                ),
            )
    return {
        "parameters": table,
        "engine": engine,
        "pit": posterior_pit(fit),
        "warnings": list(fit.plan.warnings),
    }
