"""Example 15: run one independent componentwise-SSVS PGAS chain."""
from __future__ import annotations

import argparse
from pathlib import Path

import bucex as bx


def componentwise_prior(pool: str) -> bx.HierarchicalPrior:
    return bx.componentwise_hierarchical_prior(pool)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--screen", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hpc_chains"))
    parser.add_argument("--start", default="1892-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--pool", choices=("selection", "both"), default="selection")
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=24_100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    model = bx.make_uccle_hierarchical_model()
    prior = componentwise_prior(args.pool)

    initial = None
    if args.screen is not None:
        screen = bx.FitResult.load(args.screen)
        screen_chain = (args.chain - 1) % screen.n_chains
        initial = screen.warm_start(
            chain=screen_chain,
            draw=screen.draws_per_chain - 1,
        )

    fit = bx.fit_uccle_hierarchical(
        model=model,
        start=args.start,
        end=args.end,
        priors=prior,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        asis=False,
        init=initial,
        mcmc=bx.MCMC(
            draws=args.draws,
            warmup=args.warmup,
            chains=1,
            seed=args.base_seed + args.chain,
            progress=True,
        ),
        particles=bx.Particles(n=args.particles, proposal="guided"),
        hierarchical_sampler=bx.HierarchicalSampler(
            initializer="laplace",
            channel_workers=args.workers,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"chain_{args.chain:02d}.bucex"
    fit.save(target)
    print("\nSAVED\n", target)
    print("\nPLAN\n", fit.plan)
    print("\nENGINE DIAGNOSTICS\n", fit.diagnostics()["engine"])
    print("\nCOMPONENT PROBABILITIES\n", fit.component_probabilities().round(3))
    print("\nRESTORED ITERATIONS\n", fit.meta.get("restored_iterations"))


if __name__ == "__main__":
    main()
