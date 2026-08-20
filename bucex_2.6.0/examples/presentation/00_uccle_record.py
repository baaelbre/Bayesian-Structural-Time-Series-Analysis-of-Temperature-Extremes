"""Load and describe the Uccle temperature-extreme record with the public API.

Run from the package root with

    python examples/presentation/00_uccle_record.py

All constants that a user may want to change are collected immediately below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
DATA_DIR: Path | None = None  # None uses the Uccle data shipped with bucex.
START = "1892-01-01"
END: str | None = None
SERIES = ("TXx", "TXn", "TNx", "TNn")
FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
OVERWRITE = False


def main() -> None:
    # Each call returns a monthly pandas Series with its dates attached.
    values = [
        bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        for name in SERIES
    ]
    uccle = pd.concat(values, axis=1, join="inner")
    if list(uccle.columns) != list(SERIES) or uccle.isna().any().any():
        raise ValueError("The four Uccle series are not completely aligned.")

    table_dir = OUTPUT_DIR / "tables" / "00_data"
    figure_dir = OUTPUT_DIR / "figures" / "00_data"
    data_path = table_dir / "uccle_extremes.csv"
    summary_path = table_dir / "uccle_integrity.csv"
    if not OVERWRITE:
        existing = [path for path in (data_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite {existing[0]}; set OVERWRITE=True to rerun."
            )
    table_dir.mkdir(parents=True, exist_ok=True)

    uccle.rename_axis("date").reset_index().to_csv(data_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "series": name,
                "description": bx.UCCLE_INFO[name]["description"],
                "tail": bx.UCCLE_INFO[name]["tail"],
                "n": int(uccle[name].size),
                "start": uccle[name].index.min(),
                "end": uccle[name].index.max(),
                "minimum": float(uccle[name].min()),
                "maximum": float(uccle[name].max()),
                "mean": float(uccle[name].mean()),
                "sd": float(uccle[name].std()),
            }
            for name in SERIES
        ]
    )
    summary.to_csv(summary_path, index=False)

    # The plotting helper accepts the same aligned DataFrame used above.
    bx.plot_uccle_record_figures(
        uccle,
        figure_dir,
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )
    print(f"Wrote the Uccle record to {data_path}")


if __name__ == "__main__":
    main()
