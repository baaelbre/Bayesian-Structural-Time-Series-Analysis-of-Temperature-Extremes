import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

src = "data/Uccle_24_10_23.csv"   # input daily file in current dir
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

# Load and filter
df = pd.read_csv(src, parse_dates=["DAY"])
start, end = pd.Timestamp("1892-01-01"), pd.Timestamp("2022-12-31")
df = df.loc[(df["DAY"] >= start) & (df["DAY"] <= end)].copy()

# Define meteorological seasons via quarters ending in February:
# Q1=DJF, Q2=MAM, Q3=JJA, Q4=SON. DJF is labeled by the year of Jan/Feb.
season_idx = df["DAY"].dt.to_period("Q-FEB")
df["Season"] = season_idx

# Optional: human-readable season name (DJF/MAM/JJA/SON) if you want to inspect/merge later
quarter_to_name = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}
df["SeasonName"] = df["Season"].dt.quarter.map(quarter_to_name)
df["SeasonYear"] = df["Season"].dt.year  # e.g., DJF 2022 spans Dec 2021–Feb 2022

def seasonal_agg(col):
    g = df.groupby("Season")[col]
    return {
        "max": g.max().astype(float),   # seasonal maximum of daily values
        "min": g.min().astype(float),   # seasonal minimum of daily values
        "avg": g.mean().astype(float),  # seasonal mean of daily values
    }

tx = seasonal_agg("TX")
tn = seasonal_agg("TN")

def save_series(s, fname, how="start"):
    # Use the season's start (Dec-01 for DJF) or end (last day of season) as timestamp.
    # For most climatological plots, 'start' is intuitive; switch to 'end' if you prefer.
    s = s.sort_index()
    idx = s.index.to_timestamp(how=how)  # PeriodIndex('Q-FEB') -> Timestamp
    out = pd.DataFrame({fname.split(".")[0]: s.values}, index=idx)
    out.index.name = "date"
    out.to_csv(os.path.join(out_dir, fname))

# Save 6 seasonal CSV series; index dates mark the season start (e.g., DJF starts Dec 1)
save_series(tx["max"], "TXx_seasonal.csv", how="start")
save_series(tx["min"], "TXn_seasonal.csv", how="start")
save_series(tx["avg"], "TXm_seasonal.csv", how="start")
save_series(tn["max"], "TNx_seasonal.csv", how="start")
save_series(tn["min"], "TNn_seasonal.csv", how="start")
save_series(tn["avg"], "TNm_seasonal.csv", how="start")

print("Saved 6 seasonal CSV files (DJF/MAM/JJA/SON) to", out_dir)
print("Note: DJF is labeled by the year of Jan/Feb (e.g., Dec 2021–Feb 2022 -> DJF 2022).")
