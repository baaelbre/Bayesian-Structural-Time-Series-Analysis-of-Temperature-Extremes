"""Multi-chain and engine-specific diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


Array = np.ndarray


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
    eta = fit.eta_draws()
    sigma = fit.parameter("sigma")[:, None]
    xi = fit.parameter("xi")[:, None] if fit.family == "gev" else None
    cdf = fit.model.observation.cdf(fit.y[None, :], eta, sigma=sigma, xi=xi)
    return np.nanmean(cdf, axis=0)


def fit_diagnostics(fit):
    acceptance = fit.sampler_diagnostics.get("acceptance", {})
    rows = []
    for name, values in fit.parameter_draws.items():
        if values.ndim != 2:
            continue
        row = {
            "parameter": name,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "rhat": rhat(values),
            "ess_bulk": ess_bulk(values),
            "acceptance": float(np.nanmean(acceptance[name])) if name in acceptance else np.nan,
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
            "convergence_rate": float(np.nanmean(metrics["laplace_converged"])),
            "median_iterations": float(np.nanmedian(metrics["laplace_iterations"])),
            "median_relative_change": float(np.nanmedian(metrics["laplace_relative_change"])),
            "mean_support_rejections": float(np.nanmean(metrics["laplace_support_rejections"])),
        }
    elif fit.plan.engine == "pgas":
        engine = {
            "median_min_particle_ess": float(np.nanmedian(metrics["particle_min_ess"])),
            "mean_unique_ancestors": float(np.nanmean(metrics["particle_mean_unique_ancestors"])),
            "path_change_rate": float(np.nanmean(metrics["particle_path_changed"])),
            "mean_changed_fraction": float(np.nanmean(metrics["particle_changed_fraction"])),
        }
    return {
        "parameters": table,
        "engine": engine,
        "pit": posterior_pit(fit),
        "warnings": list(fit.plan.warnings),
    }
