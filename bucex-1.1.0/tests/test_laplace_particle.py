import unittest

import numpy as np

import bucex as bx
from bucex.kalman import kalman_filter, kalman_smoother
from bucex.laplace import iterated_laplace, observation_log_likelihood
from bucex.particle import particle_filter, pgas
from bucex.numerics import gaussian_support


class TestLaplaceAndParticleInference(unittest.TestCase):
    def test_iterated_laplace_converges_and_respects_support(self):
        model = bx.Model(bx.GEV(), [bx.LocalLevel(initial_mean=10.0, initial_sd=2.0)])
        params = {"sd.level": 0.08, "sigma": 0.9, "xi": -0.15}
        simulation = bx.simulate(model, 30, params, initial_state=[10.0], seed=4)
        compiled = bx.compile_model(model, simulation.y)
        result = iterated_laplace(
            simulation.y,
            compiled,
            params,
            np.random.default_rng(5),
            max_iterations=30,
            tolerance=1e-5,
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertTrue(np.isfinite(observation_log_likelihood(simulation.y, compiled.eta(result.path), compiled, params)))

    def test_particle_likelihood_agrees_with_kalman(self):
        rng = np.random.default_rng(6)
        y = np.cumsum(rng.normal(0, 0.1, 8)) + rng.normal(0, 0.3, 8)
        compiled = bx.compile_model(
            bx.Model(bx.Gaussian(), [bx.LocalLevel(initial_mean=0.0, initial_sd=1.0)]),
            y,
        )
        params = {"sd.level": 0.1, "sigma": 0.3}
        exact = kalman_filter(y, compiled, params).log_likelihood
        for proposal, seed in (("bootstrap", 7), ("guided", 8)):
            estimate = particle_filter(
                y,
                compiled,
                params,
                particles=bx.Particles(
                    n=5000, ess_threshold=1.0, proposal=proposal
                ),
                rng=np.random.default_rng(seed),
            ).log_likelihood
            self.assertLess(abs(estimate - exact), 0.35)

    def test_pgas_gaussian_moments_match_ffbs_benchmark(self):
        y = np.asarray([0.2, 0.0, 0.3, 0.1, 0.4, 0.2])
        compiled = bx.compile_model(
            bx.Model(bx.Gaussian(), [bx.LocalLevel(initial_mean=0.0, initial_sd=1.0)]),
            y,
        )
        params = {"sd.level": 0.2, "sigma": 0.35}
        smoother = kalman_smoother(kalman_filter(y, compiled, params), compiled)
        reference = smoother.mean.copy()
        rng = np.random.default_rng(9)
        retained = []
        for iteration in range(500):
            reference = pgas(
                y,
                compiled,
                params,
                reference,
                particles=bx.Particles(n=48),
                rng=rng,
            ).path
            if iteration >= 100:
                retained.append(reference[-1, 0])
        self.assertAlmostEqual(float(np.mean(retained)), float(smoother.mean[-1, 0]), delta=0.06)

    def test_pgas_handles_singular_dummy_seasonal_transition(self):
        model = bx.Model(bx.GEV(), [bx.LocalLinearTrend(), bx.DummySeasonal(4)])
        params = {
            "sd.level": 0.05,
            "sd.slope": 0.004,
            "sd.seasonal": 0.03,
            "sigma": 0.8,
            "xi": -0.1,
        }
        simulation = bx.simulate(model, 16, params, initial_state=np.zeros(5), seed=10)
        compiled = bx.compile_model(model, simulation.y)
        reference = iterated_laplace(
            simulation.y, compiled, params, np.random.default_rng(11), max_iterations=15
        ).path
        result = pgas(
            simulation.y,
            compiled,
            params,
            reference,
            particles=bx.Particles(n=24),
            rng=np.random.default_rng(12),
        )
        self.assertEqual(result.path.shape, reference.shape)
        self.assertTrue(result.exact_invariant)
        compiled.to_noncentered(result.path, params)

    def test_tiny_slope_variance_is_not_misclassified_as_zero(self):
        model = bx.Model(bx.Gaussian(), [bx.LocalLinearTrend(), bx.DummySeasonal(4)])
        compiled = bx.compile_model(model, np.arange(20.0))
        params = {
            "sd.level": 0.01,
            "sd.slope": 0.0002,
            "sd.seasonal": 0.02,
            "sigma": 1.0,
        }
        factor = gaussian_support(compiled.transition_cov(params))
        self.assertEqual(factor.active_variances.size, 3)


if __name__ == "__main__":
    unittest.main()
