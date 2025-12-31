# %% optimization/posterior_io.py
import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

@dataclass
class PosteriorBundle:
    """Container for saved posterior draws + metadata."""
    draws: Dict[str, np.ndarray]   # e.g., {'sigma': (N,), 'xi': (N,), 'mu': (N,T), 'Q': (N,dim), ...}
    meta: Dict[str, Any]           # metadata JSON (cfg, priors, indices, dims, etc.)
    npz_path: str                  # full path to posterior.npz
    meta_path: str                 # full path to posterior.meta.json

def _resolve_npz_and_meta(path: str) -> Tuple[str, str]:
    """
    Accept either a directory (containing 'posterior.npz') or a direct path to the npz file.
    Returns (npz_path, meta_path).
    """
    if os.path.isdir(path):
        npz_path = os.path.join(path, "posterior.npz")
        meta_path = os.path.join(path, "posterior.meta.json")
    else:
        if not path.endswith(".npz"):
            raise FileNotFoundError(f"Expected a .npz file or directory, got: {path}")
        npz_path = path
        meta_path = path.replace(".npz", ".meta.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find posterior npz at: {npz_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find metadata json at: {meta_path}")

    return npz_path, meta_path

def load_posterior(path: str) -> PosteriorBundle:
    """
    Load a saved posterior:
      - path: either a directory containing 'posterior.npz' (and '.meta.json'),
              or the direct path to 'posterior.npz'.
    Returns:
      PosteriorBundle(draws=dict_of_arrays, meta=dict, npz_path=..., meta_path=...)
    """
    npz_path, meta_path = _resolve_npz_and_meta(path)

    npz = np.load(npz_path)
    draws = {k: npz[k] for k in npz.files}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return PosteriorBundle(draws=draws, meta=meta, npz_path=npz_path, meta_path=meta_path)

# ---- Convenience helpers ----------------------------------------------

def list_runs(root: str = "results/simulations/DGEV") -> list:
    """
    Return full paths of *run folders that contain 'posterior.npz'*, sorted by modification time (newest last).

    Parameters
    ----------
    results_root : str
        Base folder to search. Defaults to 'results'.
        If 'uccle_series' is provided, this is ignored and the root becomes 'uccle/<uccle_series>'.
    mode_prefix : str | None
        If provided, only keep runs whose *mode folder* (the parent of the timestamp folder)
        starts with this prefix (e.g., 'deterministic', 'dynamic', 'none').
    """
    if not os.path.isdir(root):
        return []

    # Recursively find all posterior.npz files under the root
    npz_paths = glob.glob(os.path.join(root, "**", "posterior.npz"), recursive=True)

    run_dirs = []
    for npz_path in npz_paths:
        run_dir = os.path.dirname(npz_path)  # .../<mode>/<timestamp>
        run_dirs.append(run_dir)

    # Sort by npz file modification time
    run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(d, "posterior.npz")))
    return run_dirs

def find_latest_run(root: str = "results/simulations/DGEV") -> str or None:
    """
    Find the most recent run *directory that contains 'posterior.npz'*.

    - To search the old layout: pass results_root (default 'results').
    - To search the Uccle layout: set uccle_series='TX' or 'TN' to search under 'uccle/TX' or 'uccle/TN'.
    - Optionally filter by a mode folder prefix (e.g., 'deterministic', 'dynamic', 'none').
    """
    runs = list_runs(root=root)
    return runs[-1] if runs else None

# ---- Example usage -----------------------------------------------------
if __name__ == "__main__":
    # Example A: latest under the classic 'results/' tree
    latest = find_latest_run("results/simulations/DGEV")
    if latest:
        bundle = load_posterior(latest)
        print("[classic] Loaded:", bundle.npz_path)
        print("Keys:", list(bundle.draws.keys()))
        print("Meta keys:", list(bundle.meta.keys()))
        print("sigma shape:", bundle.draws["sigma"].shape)
        print("xi shape   :", bundle.draws["xi"].shape)

    # Example B: latest Uccle TX run under 'uccle/TX/**/posterior.npz'
    latest_tx = find_latest_run("uccle/TX")
    print(latest_tx)
    if latest_tx:
        bundle_tx = load_posterior(latest_tx)
        print("[uccle TX] Loaded:", bundle_tx.npz_path)

    # Example C: filter by mode prefix (e.g., only deterministic runs) under Uccle TN
    latest_tn = find_latest_run("uccle/TN")
    if latest_tn:
        bundle_tn = load_posterior(latest_tn)
        print("[uccle TN deterministic] Loaded:", bundle_tn.npz_path)

