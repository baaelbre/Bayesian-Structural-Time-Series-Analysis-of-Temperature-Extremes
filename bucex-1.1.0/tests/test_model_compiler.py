import unittest

import numpy as np

import bucex as bx


class TestModelGrammarAndCompiler(unittest.TestCase):
    def test_model_requires_exactly_one_trend_block(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            bx.Model(bx.Gaussian(), [bx.DummySeasonal(4)])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            bx.Model(bx.Gaussian(), [bx.LocalLevel(), bx.LocalLinearTrend()])

    def test_standard_compilation(self):
        model = bx.Model(
            bx.Gaussian(),
            [bx.LocalLinearTrend(), bx.DummySeasonal(4)],
        )
        compiled = bx.compile_model(model, np.arange(12.0))
        self.assertEqual(compiled.state_names, ("level", "slope", "seasonal[1]", "seasonal[2]", "seasonal[3]"))
        self.assertEqual(compiled.noise_names, ("level", "slope", "seasonal"))
        self.assertEqual(compiled.transition.shape, (5, 5))
        self.assertEqual(compiled.loading.shape, (5, 3))
        np.testing.assert_allclose(compiled.transition[:2, :2], [[1, 1], [0, 1]])

    def test_regression_design_uses_named_dataframe_columns(self):
        import pandas as pd

        model = bx.Model(
            bx.Gaussian(),
            [bx.LocalLevel(), bx.Regression(2, name="climate", feature_names=("nao", "enso"))],
        )
        frame = pd.DataFrame({"enso": [2.0, 3.0], "nao": [5.0, 7.0], "unused": [9.0, 9.0]})
        compiled = bx.compile_model(model, [0.0, 1.0], exog=frame)
        design = compiled.design()
        np.testing.assert_allclose(design[:, 1:], [[5.0, 2.0], [7.0, 3.0]])
        self.assertEqual(compiled.noise_names, ("level",))

    def test_dynamic_regression_adds_named_disturbances(self):
        model = bx.Model(
            bx.Gaussian(),
            [bx.LocalLevel(), bx.Regression(2, dynamic=True, name="x")],
        )
        compiled = bx.compile_model(model, np.ones(5), exog=np.ones((5, 2)))
        self.assertEqual(compiled.noise_names, ("level", "x[1]", "x[2]"))
        self.assertEqual(compiled.loading.shape, (3, 3))

    def test_noncentered_roundtrip_is_general(self):
        model = bx.Model(
            bx.Gaussian(),
            [
                bx.LocalLinearTrend(),
                bx.DummySeasonal(4),
                bx.Regression(1, dynamic=True, name="x"),
            ],
        )
        exog = np.linspace(-1, 1, 20)[:, None]
        params = {
            "sd.level": 0.1,
            "sd.slope": 0.01,
            "sd.seasonal": 0.05,
            "sd.x[1]": 0.03,
            "sigma": 0.2,
        }
        simulation = bx.simulate(
            model,
            20,
            params,
            exog=exog,
            initial_state=np.zeros(6),
            seed=7,
        )
        compiled = bx.compile_model(model, simulation.y, exog=exog)
        ncp = compiled.to_noncentered(simulation.states, params)
        reconstructed = compiled.from_noncentered(ncp, params)
        np.testing.assert_allclose(reconstructed, simulation.states, atol=1e-11)

    def test_prior_names_must_match_compiler(self):
        compiled = bx.compile_model(
            bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
            np.arange(8.0),
        )
        priors = bx.Priors(
            process={"wrong": bx.PCSD(1.0)},
            observation_sd=bx.HalfNormalSD(1.0),
        )
        from bucex.priors import resolve_priors

        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_priors(compiled, priors)

    def test_model_serialization_roundtrip(self):
        model = bx.Model(
            bx.GEV(xi_bounds=(-0.4, 0.3)),
            [bx.LocalLinearTrend(), bx.DummySeasonal(4)],
            name="tail",
        )
        restored = bx.Model.from_dict(model.to_dict())
        self.assertEqual(restored, model)


if __name__ == "__main__":
    unittest.main()

