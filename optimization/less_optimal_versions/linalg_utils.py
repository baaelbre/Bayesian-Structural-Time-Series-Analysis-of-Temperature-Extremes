# optimization/linalg_utils.py
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import cho_factor, cho_solve  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    cho_factor = cho_solve = None
    _HAVE_SCIPY = False


def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def spd_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-12, max_tries: int = 6) -> np.ndarray:
    """
    Solve A X = B for (approximately) SPD A, adding diagonal jitter if needed.
    Uses Cholesky (SciPy if available) and falls back to NumPy/pinv.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = A.shape[0]
    I = np.eye(n)

    # try cholesky with increasing jitter
    for k in range(max_tries):
        eps = jitter * (10.0**k)
        Aeps = A + eps * I
        try:
            if _HAVE_SCIPY:
                c, lower = cho_factor(Aeps, lower=True, check_finite=False)
                return cho_solve((c, lower), B, check_finite=False)
            else:
                L = np.linalg.cholesky(Aeps)
                Y = np.linalg.solve(L, B)
                return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue

    # last resort
    return np.linalg.pinv(A) @ B
