from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kstest


@dataclass(frozen=True)
class PITResult:
    """Probability-integral-transform values and compact diagnostics."""

    values: np.ndarray
    bin_edges: np.ndarray
    counts: np.ndarray
    summary: dict[str, float | int]

    def histogram(self):
        rows = [
            {
                "lower": float(self.bin_edges[index]),
                "upper": float(self.bin_edges[index + 1]),
                "count": int(self.counts[index]),
                "expected": float(self.values.size / self.counts.size),
            }
            for index in range(self.counts.size)
        ]
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def plot(self, *, ax=None, save=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        widths = np.diff(self.bin_edges)
        ax.bar(
            self.bin_edges[:-1],
            self.counts,
            width=widths,
            align="edge",
            color="C0",
            alpha=0.55,
            edgecolor="white",
            label="PIT",
        )
        ax.axhline(
            self.values.size / self.counts.size,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="uniform expectation",
        )
        ax.set(xlim=(0.0, 1.0), xlabel="PIT", ylabel="count", title="PIT histogram")
        ax.legend()
        if save is not None:
            options = {}
            if isinstance(save, dict):
                options = dict(save)
                if "path" not in options:
                    raise ValueError("A save mapping requires a 'path' entry.")
                path = options.pop("path")
            else:
                path = save
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            options.setdefault("bbox_inches", "tight")
            ax.figure.savefig(path, **options)
        return ax


def empirical_coverage(samples: np.ndarray, truth: np.ndarray, alpha: float = 0.1) -> float:
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def pit_diagnostics(values: np.ndarray, *, bins: int = 10) -> PITResult:
    """Summarize PIT uniformity for genuinely held-out predictions.

    A nearly uniform histogram is necessary but not sufficient for calibrated
    forecasts.  The lag-one correlation additionally flags temporal structure
    left in the PIT sequence.  The Kolmogorov--Smirnov p-value is descriptive;
    rolling-origin PIT values overlap when ``horizon > 1`` and are then not
    independent, so the p-value should not be treated as a formal test.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("PIT values contain no finite observations.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PIT values must lie in [0, 1].")
    bins = int(bins)
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    statistic, pvalue = kstest(values, "uniform")
    if values.size > 1 and float(np.std(values)) > 0.0:
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    else:
        lag1 = np.nan
    return PITResult(
        values=values,
        bin_edges=edges,
        counts=counts,
        summary={
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
            "ks_statistic": float(statistic),
            "ks_pvalue": float(pvalue),
            "lag1_correlation": lag1,
        },
    )
