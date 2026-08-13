import unittest

import numpy as np

import bucex as bx
from bucex.diagnostics import ess_bulk, rhat


class TestPriorsAndUnifiedSampler(unittest.TestCase):
    def test_rank_diagnostics_detect_a_shifted_chain(self):
        rng = np.random.default_rng(400)
        well_mixed = rng.normal(size=(4, 400))
        shifted = well_mixed.copy()
        shifted[-1] += 2.0
        self.assertLess(rhat(well_mixed), 1.05)
        self.assertGreater(rhat(shifted), 1.1)
        self.assertGreater(ess_bulk(well_mixed), 100.0)
        self.assertLessEqual(ess_bulk(well_mixed), well_mixed.size)

    def test_mcmc_iteration_count_matches_retained_thinning_grid(self):
        self.assertEqual(bx.MCMC(draws=4, warmup=3, thin=2).iterations, 10)

    def test_pc_prior_calibration(self):
        prior = bx.PCSD(upper=0.4, alpha=0.05)
        self.assertAlmostEqual(np.exp(-prior.rate * 0.4), 0.05, places=12)

    def test_spike_slab_indicator_probability_moves_with_sd(self):
        prior = bx.SpikeSlabSD(0.01, 0.2, 0.5)
        self.assertLess(prior.indicator_probability(0.001), prior.indicator_probability(0.2))

    def test_convenience_prior_does_not_tighten_with_record_length(self):
        short = np.tile([0.0, 1.0], 20)
        long = np.tile([0.0, 1.0], 200)
        model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
        short_prior = bx.default_priors(bx.compile_model(model, short))
        long_prior = bx.default_priors(bx.compile_model(model, long))
        ratio = (
            long_prior.process["level"].upper
            / short_prior.process["level"].upper
        )
        self.assertGreater(ratio, 0.95)
        self.assertLess(ratio, 1.05)

    def test_observation_reference_removes_large_fixed_seasonal_cycle(self):
        rng = np.random.default_rng(40)
        seasonal = np.tile(np.linspace(-12.0, 12.0, 12), 20)
        y = seasonal + rng.normal(0.0, 0.4, seasonal.size)
        model = bx.Model(
            bx.Gaussian(),
            [bx.LocalLevel(), bx.DummySeasonal(12)],
        )
        compiled = bx.compile_model(model, y)
        priors = bx.default_priors(compiled)
        self.assertLess(compiled.observation_scale, compiled.y_scale / 4.0)
        self.assertAlmostEqual(priors.observation_sd.scale, compiled.observation_scale)

    def test_gaussian_centered_noncentered_and_asis(self):
        rng = np.random.default_rng(1)
        y = np.cumsum(rng.normal(0, 0.05, 25)) + rng.normal(0, 0.2, 25)
        model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
        for parameterization, asis in (("centered", False), ("noncentered", False), ("auto", True)):
            fit = bx.fit(
                y,
                model=model,
                parameterization=parameterization,
                asis=asis,
                mcmc=bx.MCMC(draws=4, warmup=4, chains=2, seed=3),
            )
            self.assertEqual(fit.state_draws.shape, (2, 4, 26, 1))
            self.assertTrue(fit.plan.targets_exact_posterior)
            self.assertEqual(fit.plan.asis, asis)

    def test_gev_engines_report_truthful_targets(self):
        model = bx.Model(bx.GEV(), [bx.LocalLevel(initial_mean=5.0, initial_sd=2.0)])
        simulation = bx.simulate(
            model,
            18,
            {"sd.level": 0.04, "sigma": 0.7, "xi": -0.1},
            initial_state=[5.0],
            seed=4,
        )
        laplace = bx.fit(
            simulation.y,
            model=model,
            engine="laplace",
            mcmc=bx.MCMC(draws=3, warmup=3, chains=1, seed=5),
            laplace=bx.Laplace(max_iterations=12),
        )
        exact = bx.fit(
            simulation.y,
            model=model,
            engine="pgas",
            mcmc=bx.MCMC(draws=3, warmup=3, chains=1, seed=6),
            particles=bx.Particles(n=24),
            laplace=bx.Laplace(max_iterations=12),
        )
        self.assertFalse(laplace.plan.targets_exact_posterior)
        self.assertIn("Laplace", laplace.plan.approximation)
        self.assertTrue(exact.plan.targets_exact_posterior)
        self.assertGreaterEqual(exact.diagnostics()["engine"]["path_change_rate"], 0.0)

    def test_spike_slab_draws_are_stored(self):
        y = np.random.default_rng(10).normal(size=20)
        fit = bx.fit(
            y,
            model=bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
            priors="spike_slab",
            mcmc=bx.MCMC(draws=5, warmup=5, chains=1, seed=11),
        )
        self.assertIn("slab.level", fit.parameter_draws)
        self.assertIn("level", fit.inclusion_probabilities())

    def test_invalid_engine_is_rejected_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            bx.fit(
                np.arange(10.0),
                model=bx.Model(bx.GEV(), [bx.LocalLevel()]),
                engine="ffbs",
                mcmc=bx.MCMC(draws=1, warmup=0, chains=1),
            )


if __name__ == "__main__":
    unittest.main()
