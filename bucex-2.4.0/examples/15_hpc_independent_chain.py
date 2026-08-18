"""Example 15: run one publication PGAS chain in one HPC process.

Launch this script four times with ``--chain 1`` through ``--chain 4`` (or use
the accompanying Slurm array).  Each task gets an independent seed and writes
one checksummed ``.bucex`` archive.  Example 16 combines the four archives.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bucex as bx


BASE_SEED = 24_100
DRAWS = 2_000
WARMUP = 2_000
PARTICLES = 512
START = "1892-01-01"
END = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--screen", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hpc_chains"))
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    model = bx.make_uccle_hierarchical_model()
    prior = bx.HierarchicalPrior(pool="selection", model_space="joint_trend")
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
        start=START,
        end=END,
        priors=prior,
        engine="pgas",
        parameterization="fs",
        asis=False,
        init=initial,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=1,
            seed=BASE_SEED + args.chain,
            progress=True,
        ),
        particles=bx.Particles(n=PARTICLES, proposal="guided"),
        hierarchical_sampler=bx.HierarchicalSampler(
            initializer="laplace",
            channel_workers=args.workers,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"chain_{args.chain:02d}.bucex"
    fit.save(target)
    print(target)


if __name__ == "__main__":
    main()
