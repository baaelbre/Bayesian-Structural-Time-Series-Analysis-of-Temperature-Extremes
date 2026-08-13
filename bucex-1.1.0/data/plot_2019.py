from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    # global base font
    "font.size": 18,

    # titles + axis labels
    "axes.titlesize": 16,
    "axes.labelsize": 16,

    # tick labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # legends
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})

YEAR = 2019
INFILE = Path(__file__).resolve().parent / "Uccle_24_10_23.csv"
OUTFILE = Path(__file__).resolve().parent / f"uccle_TX_TN_{YEAR}_daily_with_monthly_means_and_extremes.png"

# ----------------------------
# Load daily data
# ----------------------------
df = pd.read_csv(INFILE)

if not all(c in df.columns for c in ["DAY", "TX", "TN"]):
    raise ValueError(f"Expected columns DAY, TX, TN in {INFILE.name}. Found: {list(df.columns)}")

df["DAY"] = pd.to_datetime(df["DAY"], errors="coerce")
df["TX"] = pd.to_numeric(df["TX"], errors="coerce")
df["TN"] = pd.to_numeric(df["TN"], errors="coerce")

df = df.dropna(subset=["DAY", "TX", "TN"]).sort_values("DAY").set_index("DAY")

# heuristic: if temps look like 0.1°C units (e.g. 150), scale down
med_abs = np.nanmedian(np.abs(np.r_[df["TX"].to_numpy(), df["TN"].to_numpy()]))
if np.isfinite(med_abs) and med_abs > 80:
    df["TX"] /= 10.0
    df["TN"] /= 10.0

dfy = df[df.index.year == YEAR].copy()
if dfy.empty:
    raise ValueError(f"No data for year={YEAR} in {INFILE.name}")

# month key for grouping
dfy["month"] = dfy.index.to_period("M")
months = pd.period_range(f"{YEAR}-01", f"{YEAR}-12", freq="M")

# ----------------------------
# Monthly summaries from daily:
#   - means (TXm/TNm): horizontal bars spanning entire month
#   - extremes (TXx/TXn/TNx/TNn): crosses placed at the ACTUAL day they occur
# ----------------------------
summ = []
for m in months:
    g = dfy[dfy["month"] == m]
    if g.empty:
        raise ValueError(f"Missing daily data for month {m}.")

    tx_max = float(g["TX"].max())
    tx_min = float(g["TX"].min())
    tx_mean = float(g["TX"].mean())
    tx_argmax_day = g["TX"].idxmax()  # first occurrence if ties
    tx_argmin_day = g["TX"].idxmin()

    tn_max = float(g["TN"].max())
    tn_min = float(g["TN"].min())
    tn_mean = float(g["TN"].mean())
    tn_argmax_day = g["TN"].idxmax()
    tn_argmin_day = g["TN"].idxmin()

    m_start = m.to_timestamp(how="start")
    m_end_excl = (m + 1).to_timestamp(how="start")

    summ.append(
        dict(
            m_start=m_start,
            m_end_excl=m_end_excl,
            tx_mean=tx_mean,
            tn_mean=tn_mean,
            tx_max=tx_max,
            tx_min=tx_min,
            tn_max=tn_max,
            tn_min=tn_min,
            tx_argmax_day=tx_argmax_day,
            tx_argmin_day=tx_argmin_day,
            tn_argmax_day=tn_argmax_day,
            tn_argmin_day=tn_argmin_day,
        )
    )

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Daily time series
ax.plot(dfy.index, dfy["TX"], color="red", lw=0.9, alpha=0.55)
ax.plot(dfy.index, dfy["TN"], color="blue", lw=0.9, alpha=0.55)

# Summary glyph styling
lw_h = 3.2
ms = 8

# Means spanning whole month + crosses at actual min/max days
for s in summ:
    ax.hlines(s["tx_mean"], s["m_start"], s["m_end_excl"], color="red", linewidth=lw_h, alpha=0.98)
    ax.hlines(s["tn_mean"], s["m_start"], s["m_end_excl"], color="blue", linewidth=lw_h, alpha=0.98)

    ax.plot(s["tx_argmax_day"], s["tx_max"], marker="x", linestyle="None", color="red", markersize=ms)
    ax.plot(s["tx_argmin_day"], s["tx_min"], marker="x", linestyle="None", color="red", markersize=ms)
    ax.plot(s["tn_argmax_day"], s["tn_max"], marker="x", linestyle="None", color="blue", markersize=ms)
    ax.plot(s["tn_argmin_day"], s["tn_min"], marker="x", linestyle="None", color="blue", markersize=ms)

# X-axis: Jan–Dec labels
month_ticks = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-01", freq="MS")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels)

ax.set_xlim(pd.Timestamp(f"{YEAR}-01-01"), pd.Timestamp(f"{YEAR}-12-31"))
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUTFILE, dpi=220)
plt.close(fig)

print(f"[saved] {OUTFILE}")
