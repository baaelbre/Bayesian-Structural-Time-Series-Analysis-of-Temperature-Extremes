# make_rr_seasonal_monthly.py
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
INPUT_CSV = Path("data/Uccle_24_10_23.csv")  # adjust if needed
OUTDIR = Path("data")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=["DAY"])

# Ensure RR is numeric and drop rows with missing RR
df["RR"] = pd.to_numeric(df["RR"], errors="coerce")
df = df.dropna(subset=["RR"]).copy()

# ---------------------------------------------------------------------
# Helper: assign meteorological season + "season year"
# Season year i: DJF(i) = Dec(i-1), Jan(i), Feb(i)
# ---------------------------------------------------------------------
def assign_season_and_year(dt):
    m = dt.month
    y = dt.year

    if m == 12:
        season = "DJF"
        season_year = y + 1      # Dec 'y' belongs to winter of year y+1
    elif m in (1, 2):
        season = "DJF"
        season_year = y
    elif m in (3, 4, 5):
        season = "MAM"
        season_year = y
    elif m in (6, 7, 8):
        season = "JJA"
        season_year = y
    else:  # 9, 10, 11
        season = "SON"
        season_year = y

    return season, season_year

seasons, season_years = zip(*df["DAY"].map(assign_season_and_year))
df["season"] = seasons
df["season_year"] = season_years

# Make season a categorical so ordering is DJF, MAM, JJA, SON
season_order = ["DJF", "MAM", "JJA", "SON"]
df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)

# ---------------------------------------------------------------------
# 1) Seasonal mean RR  -> Precm_seasonal.csv
#    date = first day of season's first month (Jan/Apr/Jul/Oct)
# ---------------------------------------------------------------------
seasonal_mean_stats = (
    df.groupby(["season_year", "season"])["RR"]
      .mean()
      .reset_index(name="Prec")
      .sort_values(["season_year", "season"])
)

def season_to_date(row):
    sy = int(row["season_year"])
    s = row["season"]
    if s == "DJF":
        month = 1
    elif s == "MAM":
        month = 4
    elif s == "JJA":
        month = 7
    else:  # "SON"
        month = 10
    return pd.Timestamp(year=sy, month=month, day=1)

seasonal_mean_stats["date"] = seasonal_mean_stats.apply(season_to_date, axis=1)
Precm_seasonal = seasonal_mean_stats[["date", "Prec"]]
Precm_seasonal.to_csv(OUTDIR / "Precm_seasonal.csv", index=False)

# ---------------------------------------------------------------------
# 2) Seasonal max RR -> Precx_seasonal.csv
#    date = exact date within the season where max RR occurs
# ---------------------------------------------------------------------
seasonal_max_idx = df.groupby(["season_year", "season"])["RR"].idxmax()

seasonal_max = (
    df.loc[seasonal_max_idx, ["DAY", "RR", "season_year", "season"]]
      .sort_values(["season_year", "season"])
)

Precx_seasonal = seasonal_max.rename(columns={"DAY": "date", "RR": "Prec"})[
    ["date", "Prec"]
]
Precx_seasonal.to_csv(OUTDIR / "Precx_seasonal.csv", index=False)

# ---------------------------------------------------------------------
# 3) Monthly mean RR (calendar months) -> Precm_monthly.csv
#    date = first of the month
# ---------------------------------------------------------------------
df["year"] = df["DAY"].dt.year
df["month"] = df["DAY"].dt.month

monthly_mean_stats = (
    df.groupby(["year", "month"])["RR"]
      .mean()
      .reset_index(name="Prec")
      .sort_values(["year", "month"])
)

monthly_mean_stats["date"] = pd.to_datetime(
    dict(year=monthly_mean_stats["year"],
         month=monthly_mean_stats["month"],
         day=1)
)

Precm_monthly = monthly_mean_stats[["date", "Prec"]]
Precm_monthly.to_csv(OUTDIR / "Prec_monthly.csv", index=False)

# ---------------------------------------------------------------------
# 4) Monthly max RR (calendar months) -> Precx_monthly.csv
#    date = exact date within month where max RR occurs
# ---------------------------------------------------------------------
monthly_max_idx = df.groupby(["year", "month"])["RR"].idxmax()

monthly_max = (
    df.loc[monthly_max_idx, ["DAY", "RR", "year", "month"]]
      .sort_values(["year", "month"])
)

Precx_monthly = monthly_max.rename(columns={"DAY": "date", "RR": "Prec"})[
    ["date", "Prec"]
]
Precx_monthly.to_csv(OUTDIR / "Precx.csv", index=False)

print("Written:")
print(f"  {OUTDIR / 'Precm_seasonal.csv'}")
print(f"  {OUTDIR / 'Precx_seasonal.csv'}")
print(f"  {OUTDIR / 'Precm.csv'}")
print(f"  {OUTDIR / 'Precx.csv'}")
