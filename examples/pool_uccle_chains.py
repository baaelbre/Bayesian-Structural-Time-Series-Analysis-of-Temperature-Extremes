#!/usr/bin/env python3
"""Pool completed Uccle chains and write basic split-Rhat diagnostics.

This script concatenates already post-burn posterior draws. It does not alter or
rerun the chains. Review the Rhat table and restoration summaries before using
the pooled fits for poster or manuscript figures.

Example
-------
python examples/pool_uccle_chains.py \
    --chains-dir results/uccle_v033_regularized/chains \
    --out-dir results/uccle_v033_regularized
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bucex import PosteriorBundle, __version__

SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chains-dir",
        type=Path,
        default=Path("results/uccle_v033_regularized/chains"),
        help="Directory containing chain_1, chain_2, ... subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/uccle_v033_regularized"),
        help="Output root for pooled fits and diagnostics.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Chain numbers to pool. Default: 1 2 3.",
    )
    parser.add_argument(
        "--pool-thin",
        type=int,
        default=3,
        help=(
            "Keep every k-th post-burn draw in the pooled fit. Default: 3. "
            "Rhat is still computed from the unthinned chains."
        ),
    )
    parser.add_argument(
        "--rhat-warning",
        type=float,
        default=1.05,
        help="Print a warning for split Rhat above this value.",
    )
    return parser.parse_args()


def find_fit(chains_dir: Path, chain: int, series: str) -> Path:
    fit_dir = chains_dir / f"chain_{chain}" / "fits"
    preferred = fit_dir / f"{series}_bucex_v{__version__}.pkl"
    if preferred.exists():
        return preferred
    matches = sorted(fit_dir.glob(f"{series}_bucex_v*.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"No saved fit found for {series}, chain {chain}, under {fit_dir}."
        )
    return matches[-1]


def check_compatible(fits: list[PosteriorBundle], series: str) -> None:
    first = fits[0]
    for i, fit in enumerate(fits[1:], start=2):
        if fit.series_name != first.series_name:
            raise ValueError(f"Chain {i} has a different series name for {series}.")
        if fit.state_names != first.state_names:
            raise ValueError(f"Chain {i} has different state names for {series}.")
        if fit.draws_states is None or first.draws_states is None:
            raise ValueError(f"All chains must contain state draws for {series}.")
        if fit.draws_states.shape[1:] != first.draws_states.shape[1:]:
            raise ValueError(f"Chain {i} has an incompatible state shape for {series}.")
        if set(fit.draws_static) != set(first.draws_static):
            raise ValueError(f"Chain {i} has different static parameters for {series}.")
        if fit.dates is not None and first.dates is not None:
            if not np.array_equal(np.asarray(fit.dates), np.asarray(first.dates)):
                raise ValueError(f"Chain {i} has different dates for {series}.")


def split_rhat(chains: list[np.ndarray]) -> float:
    """Classical split Rhat for equal-length one-dimensional chains."""
    arrays = [np.asarray(values, dtype=float).reshape(-1) for values in chains]
    half = min(len(values) for values in arrays) // 2
    if half < 2:
        return float("nan")
    split = []
    for values in arrays:
        values = values[-2 * half :]
        split.extend((values[:half], values[half:]))
    x = np.stack(split)
    n = x.shape[1]
    within = np.var(x, axis=1, ddof=1)
    W = float(np.mean(within))
    chain_means = np.mean(x, axis=1)
    B = float(n * np.var(chain_means, ddof=1))
    if W <= 0.0:
        return 1.0 if B <= 0.0 else float("inf")
    var_hat = ((n - 1.0) / n) * W + B / n
    return float(np.sqrt(max(var_hat / W, 0.0)))


def pool_one(
    series: str,
    fits: list[PosteriorBundle],
    *,
    pool_thin: int = 3,
) -> PosteriorBundle:
    check_compatible(fits, series)
    first = fits[0]
    if int(pool_thin) < 1:
        raise ValueError("pool_thin must be at least 1.")
    selector = slice(None, None, int(pool_thin))

    draws_static = {
        key: np.concatenate(
            [np.asarray(fit.draws_static[key])[selector] for fit in fits], axis=0
        )
        for key in first.draws_static
    }
    draws_states = np.concatenate(
        [np.asarray(fit.draws_states)[selector] for fit in fits], axis=0
    )
    logpost = None
    if all(fit.logpost is not None for fit in fits):
        logpost = np.concatenate(
            [np.asarray(fit.logpost)[selector] for fit in fits], axis=0
        )

    weights = np.asarray([fit.n_draws for fit in fits], dtype=float)
    acceptance_keys = set().union(*(fit.acceptance for fit in fits))
    acceptance = {}
    for key in acceptance_keys:
        values = np.asarray([fit.acceptance.get(key, np.nan) for fit in fits], dtype=float)
        keep = np.isfinite(values)
        acceptance[key] = float(np.average(values[keep], weights=weights[keep])) if np.any(keep) else np.nan

    meta = dict(first.meta)
    # The non-centred trajectories duplicate most of the large state storage and
    # are not needed for the poster/risk API. Deliberately omit them from the
    # pooled fit to keep memory and file size manageable.
    meta.pop("draws_states_ncp", None)

    meta.update(
        {
            "bucex_version": __version__,
            "pooled_chains": [int(fit.meta.get("chain_id", i + 1)) for i, fit in enumerate(fits)],
            "chain_seeds": [fit.meta.get("chain_seed") for fit in fits],
            "chain_n_draws": [fit.n_draws for fit in fits],
            "pooled_chain_n_draws": [len(range(0, fit.n_draws, int(pool_thin))) for fit in fits],
            "pool_thin": int(pool_thin),
            "n_chains": len(fits),
            "pooled_posterior": True,
            "pooled_omits_ncp_draws": True,
            "restored_iterations_by_chain": [
                int(fit.meta.get("restored_iterations", 0)) for fit in fits
            ],
            "restored_fraction_by_chain": [
                float(fit.meta.get("restored_fraction", 0.0)) for fit in fits
            ],
            "restore_failure_counts_by_chain": [
                dict(fit.meta.get("restore_failure_counts", {})) for fit in fits
            ],
        }
    )

    return PosteriorBundle(
        draws_static=draws_static,
        draws_states=draws_states,
        logpost=logpost,
        acceptance=acceptance,
        meta=meta,
        y=None if first.y is None else np.asarray(first.y).copy(),
        dates=None if first.dates is None else np.asarray(first.dates).copy(),
        model=first.model,
        state_names=first.state_names,
        series_name=first.series_name,
        transform_sign=first.transform_sign,
    )


def main() -> None:
    args = parse_args()
    fit_out = args.out_dir / "fits"
    diag_out = args.out_dir / "diagnostics"
    summary_out = args.out_dir / "summaries"
    fit_out.mkdir(parents=True, exist_ok=True)
    diag_out.mkdir(parents=True, exist_ok=True)
    summary_out.mkdir(parents=True, exist_ok=True)

    all_rhat_rows: list[dict[str, object]] = []
    restore_rows: list[dict[str, object]] = []

    for series in SERIES:
        paths = [find_fit(args.chains_dir, chain, series) for chain in args.chains]
        fits = [PosteriorBundle.load(path) for path in paths]

        common_scalar_keys = [
            key
            for key in fits[0].draws_static
            if all(np.asarray(fit.draws_static[key]).ndim == 1 for fit in fits)
        ]
        for key in common_scalar_keys:
            value = split_rhat([fit.draws_static[key] for fit in fits])
            all_rhat_rows.append({"series": series, "parameter": key, "split_rhat": value})

        for chain, path, fit in zip(args.chains, paths, fits):
            restore_rows.append(
                {
                    "series": series,
                    "chain": chain,
                    "path": str(path),
                    "n_draws": fit.n_draws,
                    "restored_iterations": int(fit.meta.get("restored_iterations", 0)),
                    "restored_fraction": float(fit.meta.get("restored_fraction", 0.0)),
                    "restore_failure_counts": json.dumps(
                        fit.meta.get("restore_failure_counts", {}), sort_keys=True
                    ),
                }
            )

        pooled = pool_one(series, fits, pool_thin=args.pool_thin)
        pooled_path = fit_out / f"{series}_bucex_v{__version__}.pkl"
        pooled.save(pooled_path)
        pd.DataFrame(pooled.static_summary()).T.to_csv(
            summary_out / f"{series}_pooled_posterior_summary.csv"
        )
        print(
            f"Pooled {series}: {sum(fit.n_draws for fit in fits)} original draws, "
            f"{pooled.n_draws} retained after pool thinning -> {pooled_path}"
        )

    rhat = pd.DataFrame(all_rhat_rows)
    rhat.to_csv(diag_out / "split_rhat.csv", index=False)
    restores = pd.DataFrame(restore_rows)
    restores.to_csv(diag_out / "restoration_summary.csv", index=False)

    bad = rhat[np.isfinite(rhat["split_rhat"]) & (rhat["split_rhat"] > args.rhat_warning)]
    if len(bad):
        warnings.warn(
            f"{len(bad)} scalar series/parameter combinations have split Rhat "
            f"> {args.rhat_warning:.3f}. Inspect diagnostics/split_rhat.csv before interpretation.",
            RuntimeWarning,
        )
    print(f"Wrote diagnostics to: {diag_out}")


if __name__ == "__main__":
    main()
