#!/usr/bin/env python3
"""Combine compatible, independently saved FS chain archives."""
from __future__ import annotations

import argparse
from pathlib import Path

import bucex as bx


parser = argparse.ArgumentParser()
parser.add_argument("fits", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--diagnostics", type=Path, default=None)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

if args.output.exists() and not args.overwrite:
    raise FileExistsError(
        f"{args.output} exists; choose another output or pass --overwrite."
    )

fits = [bx.PosteriorBundle.load(path) for path in args.fits]
combined = bx.combine_fs_fits(fits)
combined.save(args.output)

diagnostics = combined.diagnostics()
if args.diagnostics is not None:
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics["parameters"].to_csv(args.diagnostics)

print(
    f"Combined {combined.n_chains} chains and {combined.n_draws} draws "
    f"into {args.output}"
)
