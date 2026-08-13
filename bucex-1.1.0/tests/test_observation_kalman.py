import unittest

import numpy as np
from scipy.stats import multivariate_normal

import bucex as bx
from bucex.numerics import psd_eigh
from bucex.kalman import ffbs, kalman_filter, kalman_smoother


class TestObservationAndKalman(unittest.TestCase):
    def test_psd_projection_only_accepts_roundoff_sized_negative_eigenvalues(self):
        values, _ = psd_eigh(np.diag([1.0, -5e-10]))
        np.testing.assert_allclose(values, [0.0, 1.0])
        with self.assertRaises(np.linalg.LinAlgError):
            psd_eigh(np.diag([1.0, -1e-5]))

    def test_gev_derivatives_match_finite_differences(self):
        observation = bx.GEV()
        sigma = 1.3
        eta = 0.4
        step = 1e-5
        for xi in (-0.4, -0.15, 0.0, 0.2, 0.4):
            upper = eta - sigma / xi if xi < 0.0 else np.inf
            values = (-0.2, eta + 0.75 * (upper - eta)) if xi < 0.0 else (-0.2, 0.9)
            for y in values:
                f = lambda value: float(observation.logpdf(y, value, sigma=sigma, xi=xi))
                grad = (f(eta + step) - f(eta - step)) / (2 * step)
                hess = (f(eta + step) - 2 * f(eta) + f(eta - step)) / step**2
                self.assertAlmostEqual(float(observation.grad_eta(y, eta, sigma, xi)), grad, places=6)
                self.assertAlmostEqual(float(observation.hess_eta(y, eta, sigma, xi)), hess, places=3)

    def test_local_level_kalman_likelihood_matches_joint_normal(self):
        y = np.asarray([0.2, -0.1, 0.4, 0.3])
        model = bx.Model(bx.Gaussian(), [bx.LocalLevel(initial_mean=0.0, initial_sd=1.2)])
        compiled = bx.compile_model(model, y)
        params = {"sd.level": 0.3, "sigma": 0.5}
        result = kalman_filter(y, compiled, params)
        n = y.size
        covariance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                covariance[i, j] = 1.2**2 + (min(i, j) + 1) * 0.3**2
        covariance += 0.5**2 * np.eye(n)
        exact = multivariate_normal.logpdf(y, mean=np.zeros(n), cov=covariance)
        self.assertAlmostEqual(result.log_likelihood, exact, places=9)

    def test_ffbs_draws_recover_smoother_moments(self):
        y = np.asarray([0.1, 0.4, 0.2, 0.7, 0.5])
        model = bx.Model(bx.Gaussian(), [bx.LocalLevel(initial_mean=0.0, initial_sd=1.0)])
        compiled = bx.compile_model(model, y)
        params = {"sd.level": 0.2, "sigma": 0.3}
        filt = kalman_filter(y, compiled, params)
        smooth = kalman_smoother(filt, compiled)
        rng = np.random.default_rng(20)
        draws = np.stack([ffbs(y, compiled, params, rng, filter_result=filt)[0] for _ in range(1500)])
        np.testing.assert_allclose(draws.mean(axis=0), smooth.mean, atol=0.035)

    def test_missing_observation_is_skipped(self):
        y = np.asarray([0.1, np.nan, 0.2])
        compiled = bx.compile_model(bx.Model(bx.Gaussian(), [bx.LocalLevel()]), y)
        result = kalman_filter(y, compiled, {"sd.level": 0.1, "sigma": 0.3})
        self.assertTrue(result.missing[1])
        np.testing.assert_allclose(result.filtered_mean[2], result.predicted_mean[2])


if __name__ == "__main__":
    unittest.main()
