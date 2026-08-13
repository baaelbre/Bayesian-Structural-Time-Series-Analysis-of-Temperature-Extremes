import tempfile
import unittest
from pathlib import Path

import numpy as np

import bucex as bx


class TestForecastScoresAndIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20)
        cls.y = np.cumsum(rng.normal(0, 0.08, 30)) + rng.normal(0, 0.2, 30)
        cls.fit = bx.fit(
            cls.y,
            model=bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
            mcmc=bx.MCMC(draws=8, warmup=8, chains=2, seed=21),
            asis=True,
        )

    def test_forecast_propagates_and_widens_process_uncertainty(self):
        forecast = self.fit.forecast(20, draws=2000, seed=22)
        self.assertEqual(forecast.observations.shape, (2000, 20))
        variance = np.var(forecast.eta, axis=0)
        self.assertGreater(variance[-1], variance[0])

    def test_scores_have_expected_zero_for_perfect_constant_ensemble(self):
        observed = np.asarray([1.0, 2.0])
        samples = np.tile(observed, (10, 1))
        np.testing.assert_allclose(bx.crps_ensemble(samples, observed), 0.0)
        np.testing.assert_allclose(bx.threshold_weighted_crps(samples, observed, 1.5), 0.0)

    def test_forecast_score_includes_tail_metrics(self):
        forecast = bx.forecast(self.fit, 3, draws=20, seed=23)
        table = bx.score(
            forecast,
            np.zeros(3),
            thresholds=[0.2],
            quantiles=[0.9],
        )
        self.assertEqual(set(table["score"]), {"crps", "twcrps", "exceedance_brier", "exceedance_log", "quantile"})

    def test_top_level_score_accepts_a_raw_ensemble(self):
        samples = np.asarray([[0.0, 1.0], [0.2, 1.2]])
        table = bx.score(samples, np.asarray([0.1, 1.1]), quantiles=[])
        self.assertEqual(table.iloc[0]["score"], "crps")

    def test_safe_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.bucex"
            self.fit.save(path)
            restored = bx.FitResult.load(path)
            np.testing.assert_allclose(restored.state_draws, self.fit.state_draws)
            np.testing.assert_allclose(restored.parameter("sigma"), self.fit.parameter("sigma"))
            self.assertEqual(restored.model, self.fit.model)
            self.assertEqual(restored.plan, self.fit.plan)

    def test_prior_posterior_sd_plot(self):
        import matplotlib

        matplotlib.use("Agg")
        figure, axes = self.fit.plot("process_sd")
        self.assertEqual(len(axes), 1)
        figure.canvas.draw()

    def test_minimum_tail_transform_is_returned_on_original_scale(self):
        values = 5.0 - np.random.default_rng(30).gumbel(size=18)
        fit = bx.fit(
            values,
            family="gev",
            trend="local_level",
            tail="min",
            engine="laplace",
            mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=31),
            laplace=bx.Laplace(max_iterations=10),
        )
        forecast = fit.forecast(2, draws=2, seed=32)
        self.assertEqual(forecast.tail, "lower")
        self.assertTrue(np.all(np.isfinite(forecast.observations)))
        np.testing.assert_allclose(fit.observed, values)


if __name__ == "__main__":
    unittest.main()
