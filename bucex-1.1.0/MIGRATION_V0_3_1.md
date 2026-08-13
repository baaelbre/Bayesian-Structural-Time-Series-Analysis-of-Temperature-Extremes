# Migration to bucex v0.3.1

Existing v0.2 manuscript calls continue to work:

```python
fit_bayes(..., priors="manuscript")
```

New profiles:

```python
fit_bayes(..., priors="normal")
fit_bayes(..., priors="ssvs")
```

Saved Uccle files now use names such as:

```text
TXx_bucex_v0.3.1.pkl
```

The poster script searches for the newest matching version and can therefore
still read v0.2 files.

SSVS fits contain additional static draws:

```text
state_level
state_trend
state_season
model_index
```

State codes are:

```text
0 = zero
1 = fixed
2 = dynamic
```

`state_level` never equals zero.
