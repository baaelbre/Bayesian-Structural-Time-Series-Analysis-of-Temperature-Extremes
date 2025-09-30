import os
import pandas as pd


def main():
    src = "data/Uccle_24_10_23.csv"   # input daily file in current dir
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Load and filter
    df = pd.read_csv(src, parse_dates=["DAY"])
    start, end = pd.Timestamp("1892-01-01"), pd.Timestamp("2022-12-31")
    df = df.loc[(df["DAY"] >= start) & (df["DAY"] <= end)].copy()
    df["YearMonth"] = df["DAY"].dt.to_period("M")

    # Aggregation helper
    def monthly_agg(col):
        g = df.groupby("YearMonth")[col]
        return {
            "max": g.max().astype(float),
            "min": g.min().astype(float),
            "avg": g.mean().astype(float),
        }

    tx = monthly_agg("TX")
    tn = monthly_agg("TN")

    def save_series(s, fname):
        s = s.sort_index()
        idx = s.index.to_timestamp(how="start")
        out = pd.DataFrame({fname.split(".")[0]: s.values}, index=idx)
        out.index.name = "date"
        out.to_csv(os.path.join(out_dir, fname))

    save_series(tx["max"], "TXx.csv")
    save_series(tx["min"], "TXn.csv")
    save_series(tx["avg"], "TXm.csv")
    save_series(tn["max"], "TNx.csv")
    save_series(tn["min"], "TNn.csv")
    save_series(tn["avg"], "TNm.csv")

    print("Saved 6 monthly CSV files to", out_dir)


if __name__ == "__main__":
    main()
