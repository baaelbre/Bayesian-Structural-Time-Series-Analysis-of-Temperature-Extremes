# plot_covid_weekly.py
import sys
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/weekly_cases.csv" 
COUNTRY  = "Belgium"
START_DATE = "2020-03-01"   # or None
END_DATE   = "2022-12-31"   # or None

df = pd.read_csv(CSV_PATH)

s = df[["date", COUNTRY]].copy()
s["date"] = pd.to_datetime(s["date"], errors="coerce")
s = s.dropna(subset=["date"]).sort_values("date")

if START_DATE is not None:
    s = s[s["date"] >= START_DATE]
if END_DATE is not None:
    s = s[s["date"] <= END_DATE]

plt.figure(figsize=(9, 4.5))
plt.plot(s["date"], s[COUNTRY])  
plt.title(f"COVID-19 weekly cases — {COUNTRY}")
plt.xlabel("Date")
plt.ylabel("Cases per week")
plt.tight_layout()
plt.show(block=True)
