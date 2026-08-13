import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx


class TestParallelRegressionAndCompatibilityAPI(unittest.TestCase):
    def test_parallel_bulk_tail_object_is_explicitly_independent(self):
        rng = np.random.default_rng(40)
        bulk = rng.normal(size=18)
        tail = rng.gumbel(size=18)
        quick = bx.MCMC(draws=2, warmup=2, chains=1, seed=41)
        pair = bx.fit_bulk_tail(
            bulk,
            tail,
            period=None,
            bulk_mcmc=quick,
            tail_mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=42),
            tail_engine="laplace",
        )
        self.assertFalse(pair.metadata["joint_likelihood"])
        self.assertEqual(pair.bulk.family, "gaussian")
        self.assertEqual(pair.tail.family, "gev")

    def test_static_and_dynamic_regression_fit(self):
        rng = np.random.default_rng(43)
        exog = rng.normal(size=(25, 1))
        y = 2.0 * exog[:, 0] + rng.normal(scale=0.2, size=25)
        for dynamic in (False, True):
            model = bx.Model(
                bx.Gaussian(),
                [bx.LocalLevel(), bx.Regression(1, dynamic=dynamic, name="x")],
            )
            fit = bx.fit(
                y,
                model=model,
                exog=exog,
                mcmc=bx.MCMC(draws=3, warmup=3, chains=1, seed=44),
            )
            self.assertIn("x[1]", fit.state_names)
            if dynamic:
                self.assertIn("sd.x[1]", fit.parameter_draws)

    def test_fit_bayes_aliases_old_profiles_and_particle_name(self):
        y = np.random.default_rng(45).normal(size=16)
        fit = bx.fit_bayes(
            y,
            family="gaussian",
            period=None,
            priors="normal",
            n_iter=6,
            burn=2,
            thin=2,
            chains=1,
            seed=46,
        )
        self.assertEqual(fit.draws_per_chain, 2)
        self.assertEqual(fit.priors.profile, "half_normal")

    def test_family_specific_wrappers_select_the_right_observation(self):
        y = np.random.default_rng(48).normal(size=12)
        gaussian = bx.fit_gaussian_structural(
            y,
            trend="local_level",
            mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=49),
        )
        gev = bx.fit_gev_structural(
            y,
            trend="local_level",
            engine="laplace",
            mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=50),
            laplace=bx.Laplace(max_iterations=8),
        )
        self.assertEqual(gaussian.family, "gaussian")
        self.assertEqual(gev.family, "gev")

    def test_uccle_loader_and_fit_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            dates = pd.date_range("2000-01-01", periods=24, freq="MS")
            frame = pd.DataFrame({"date": dates, "TXm": np.linspace(1, 2, 24)})
            frame.to_csv(Path(directory) / "TXm.csv", index=False)
            series = bx.load_uccle_series("TXm", directory)
            self.assertEqual(series.size, 24)
            fit = bx.fit_uccle_series(
                "TXm",
                directory,
                mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=47),
                asis=False,
            )
            self.assertEqual(fit.series_name, "TXm")

    def test_pandas_series_metadata_is_preserved(self):
        series = pd.Series(
            np.random.default_rng(51).normal(size=12),
            index=pd.date_range("2010-01-01", periods=12, freq="MS"),
            name="bulk_temperature",
        )
        fit = bx.fit(
            series,
            trend="local_level",
            mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=52),
        )
        self.assertEqual(fit.series_name, "bulk_temperature")
        np.testing.assert_array_equal(fit.dates, series.index.to_numpy())


if __name__ == "__main__":
    unittest.main()
