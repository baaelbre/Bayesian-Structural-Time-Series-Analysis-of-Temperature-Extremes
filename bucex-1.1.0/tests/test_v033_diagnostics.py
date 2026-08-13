from __future__ import annotations

from collections import Counter

from bucex import __version__, fit_uccle_series
from bucex.inference.fit.noncentered_gev import _compact_counter


def test_v033_diagnostics_remain_available_in_v11():
    assert __version__ == "1.1.0"


def test_restore_failure_counter_format_is_compact_and_sorted():
    counter = Counter({"support_after_state_parameters": 25, "laplace_ffbs:ValueError": 2})
    text = _compact_counter(counter)
    assert text.startswith("support_after_state_parameters:25")
    assert "laplace_ffbs:ValueError:2" in text


def test_gev_fit_saves_restoration_diagnostics():
    fit = fit_uccle_series(
        "TXx",
        data_dir="data",
        start="2000-01-01",
        end="2002-12-31",
        priors="regularized",
        n_iter=6,
        burn=2,
        progress=False,
        seed=333,
    )
    for key in (
        "restored_iterations",
        "restored_fraction",
        "attempt_failure_counts",
        "restore_failure_counts",
        "max_state_tries",
    ):
        assert key in fit.meta
    assert fit.meta["max_state_tries"] == 25
    assert 0.0 <= fit.meta["restored_fraction"] <= 1.0


def test_restored_progress_line_prints_reason(monkeypatch, capsys):
    import numpy as np
    import bucex.inference.fit.noncentered_gev as module
    from bucex import fit_bayes

    calls = {"n": 0}

    def fake_support(*args, **kwargs):
        calls["n"] += 1
        return calls["n"] == 1  # initial support check only

    monkeypatch.setattr(module, "_gev_support_ok", fake_support)
    y = np.array(
        [
            10.0, 11.0, 9.5, 10.5, 11.2, 10.1,
            9.8, 10.7, 11.1, 10.2, 9.9, 10.4,
            10.3, 11.3, 9.7, 10.6, 11.4, 10.2,
            9.9, 10.8, 11.2, 10.3, 10.0, 10.5,
        ]
    )
    fit = fit_bayes(
        y,
        family="gev",
        period=12,
        priors="regularized",
        n_iter=2,
        burn=1,
        seed=1,
        progress=True,
        state_kwargs={"max_state_tries": 2},
    )
    output = capsys.readouterr().out
    assert "[restored attempts=2" in output
    assert "support_after_state_parameters:2" in output
    assert fit.meta["restored_iterations"] == 2
    assert fit.meta["restore_failure_counts"] == {
        "support_after_state_parameters": 4
    }
