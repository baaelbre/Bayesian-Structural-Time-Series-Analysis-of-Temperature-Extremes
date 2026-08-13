# Uccle data in this source release

`Uccle_24_10_23.csv` is the supplied daily source table with columns `DAY`,
`TX`, `TN`, and `RR`. The six monthly files cover January 1892 through December
2022:

| File | Aggregation |
| --- | --- |
| `TXm.csv` | monthly mean of daily maximum temperature |
| `TNm.csv` | monthly mean of daily minimum temperature |
| `TXx.csv` | monthly maximum of daily maximum temperature |
| `TXn.csv` | monthly minimum of daily maximum temperature |
| `TNx.csv` | monthly maximum of daily minimum temperature |
| `TNn.csv` | monthly minimum of daily minimum temperature |

They can be regenerated and checked with:

```python
import bucex as bx

monthly = bx.derive_uccle_monthly("data")
print(bx.validate_uccle_data("data", check_daily=True))
```

All six supplied monthly files reproduce the corresponding daily aggregation
to floating-point precision (maximum absolute discrepancy below `4e-15`).

## Provenance and redistribution gate

The uploaded package did not include the daily file's original download URL,
dataset identifier, access date, citation, or license. The station name and
variables are consistent with Royal Meteorological Institute of Belgium
(RMI/KMI/IRM) observations, but that inference is not sufficient provenance
for public redistribution.

Before publishing the CSVs, record the exact source dataset and verify that its
terms cover this historical extract and derived monthly files. The official
RMI open-data portal and terms are:

- <https://opendata.meteo.be/download>
- <https://opendata.meteo.be/termsandconditions>

The portal states that qualifying high-value datasets may be reused under CC
BY 4.0, but the exact supplied file has not been tied to such a dataset record.
Do not apply CC BY 4.0 merely by assumption. The repository's MIT license covers
software, not the observations.
