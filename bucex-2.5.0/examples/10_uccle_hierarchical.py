"""Example 10: componentwise hierarchical SSVS for all six Uccle summaries.

Each channel retains its own likelihood, latent path, observation parameters,
and innovation scales. The population hierarchy pools probabilities for:

* level: fixed or dynamic;
* slope: zero, fixed, or dynamic;
* seasonality: fixed or dynamic.

Use Laplace for exploratory screening and PGAS for final mixed Gaussian--GEV
inference. A full-record Laplace archive can initialize the corresponding
full-record PGAS chains.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


def componentwise_prior(pool: str = "selection") -> bx.HierarchicalPrior:
    """The single componentwise hierarchy used by screening and exact jobs."""
    return bx.componentwise_hierarchical_prior(pool)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1980-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--pool", choices=("selection", "both"), default="selection")
    parser.add_argument("--engine", choices=("laplace", "pgas"), default="laplace")
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1_001)
    parser.add_argument("--save-fit", type=Path, default=None)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/10_uccle_hierarchical"),
    )
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    model = bx.make_uccle_hierarchical_model()
    data = bx.load_uccle_multiseries(start=args.start, end=args.end)
    prior = componentwise_prior(args.pool)

    fit_options = {}
    if args.engine == "pgas":
        fit_options["particles"] = bx.Particles(
            n=args.particles,
            proposal="guided",
        )

    fit = bx.fit_uccle_hierarchical(
        model=model,
        start=args.start,
        end=args.end,
        priors=prior,
        parameterization="fruehwirth_schnatter",
        engine=args.engine,
        asis=False,
        mcmc=bx.MCMC(
            draws=args.draws,
            warmup=args.warmup,
            chains=args.chains,
            seed=args.seed,
            progress=True,
        ),
        hierarchical_sampler=bx.HierarchicalSampler(
            initializer="laplace",
            channel_workers=args.workers,
        ),
        **fit_options,
    )

    scientific = [
        name
        for name in fit.parameter_draws
        if name.startswith(("sd.channel.", "sigma.", "xi.", "hierarchy."))
    ]
    diagnostics = fit.diagnostics()
    draw_metrics = fit.sampler_diagnostics.get("draw_metrics", {})
    sign_errors = np.asarray(draw_metrics.get("sign_invariance_error", np.nan))
    restored = int(fit.meta.get("restored_iterations", 0) or 0)

    print("\nDATA\n", data.describe().round(2))
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nSCIENTIFIC DIAGNOSTICS\n", diagnostics["parameters"].loc[scientific].round(4))
    print("\nENGINE DIAGNOSTICS\n", diagnostics["engine"])
    print("\nCHANNEL COMPONENT PROBABILITIES\n", fit.component_probabilities().round(3))
    print("\nCHANNEL JOINT STRUCTURES\n", fit.structural_model_probabilities().round(3))
    print("\nPOPULATION COMPONENT PROBABILITIES\n", fit.hierarchical_probabilities().round(3))
    print("\nPOOLED SLAB MULTIPLIERS\n", fit.hierarchical_slab_summary().round(3))
    print("\nALLOCATION SWITCHING\n", fit.component_transition_summary().round(3))
    print(
        "\nSAMPLER CONTRACT\n",
        {
            "exact_target": fit.plan.targets_exact_posterior,
            "model_space": fit.priors.hierarchy.model_space,
            "pool": fit.priors.hierarchy.pool,
            "sign_switching": fit.meta.get("sign_switching"),
            "sign_invariance_error": float(np.nanmax(sign_errors)),
            "restored_iterations": restored,
        },
    )
    if restored:
        warnings.warn(
            f"This fit restored {restored} iterations. Diagnose the failure "
            "types before treating it as a result.",
            RuntimeWarning,
        )

    print("\nANNUAL-MEAN COMPLETE-PREDICTOR RATES PER DECADE")
    for channel in model.channel_names:
        print(channel, fit.channel_rate_summary(channel))

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for channel in model.channel_names:
        fit.plot("channel", channel=channel, save=args.figure_dir / f"{channel}.png")
    fit.plot("component_probabilities", save=args.figure_dir / "allocations.png")
    fit.plot("hierarchy", save=args.figure_dir / "hierarchy.png")
    fit.plot("process_sd", save=args.figure_dir / "process_sds.png")
    fit.plot("traces", parameters=scientific, save=args.figure_dir / "traces.png")
    fit.plot("acf", parameters=scientific, max_lag=50, save=args.figure_dir / "acf.png")

    if args.save_fit is not None:
        args.save_fit.parent.mkdir(parents=True, exist_ok=True)
        fit.save(args.save_fit)
        print("\nSAVED FIT\n", args.save_fit)

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
