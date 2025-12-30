# optimization/utils.py
from __future__ import annotations

import math
import numpy as np

# -----------------------------------------------------------------------------
# Optional SciPy (linalg + inverse-Gaussian RNG)
# -----------------------------------------------------------------------------
try:
    from scipy.linalg import cho_factor, cho_solve  # type: ignore
    _HAVE_SCIPY_LINALG = True
except Exception:
    cho_factor = cho_solve = None
    _HAVE_SCIPY_LINALG = False

try:
    from scipy.stats import invgauss  # type: ignore
    _HAVE_SCIPY_STATS = True
except Exception:
    invgauss = None
    _HAVE_SCIPY_STATS = False


# =============================================================================
# Linear algebra helpers
# =============================================================================
def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
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

    for k in range(max_tries):
        eps = jitter * (10.0**k)
        Aeps = A + eps * I
        try:
            if _HAVE_SCIPY_LINALG:
                c, lower = cho_factor(Aeps, lower=True, check_finite=False)
                return cho_solve((c, lower), B, check_finite=False)
            L = np.linalg.cholesky(Aeps)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue

    return np.linalg.pinv(A) @ B


# =============================================================================
# RNG helpers
# =============================================================================
def _rand_invgauss_msh(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Michael–Schucany–Haas method for IG(mu, lam).
    """
    if mu <= 0.0 or lam <= 0.0:
        raise ValueError("Inverse-Gaussian requires mu>0 and lam>0")

    v = float(rng.normal())
    y = v * v
    mu2 = mu * mu
    term = mu2 * y
    x = mu + term / (2.0 * lam) - (mu / (2.0 * lam)) * math.sqrt(4.0 * mu * lam * y + term * y)
    u = float(rng.random())
    if u <= mu / (mu + x):
        return float(x)
    return float(mu2 / x)


def rand_invgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Draw X ~ IG(mu, lam) with density proportional to:
        sqrt(lam/(2π x^3)) exp(-lam (x-mu)^2 / (2 mu^2 x))

    If SciPy is available, use its invgauss sampler with scaling:
        if Y ~ IG(mu/lam, 1) then X = lam * Y ~ IG(mu, lam).
    Otherwise fall back to Michael–Schucany–Haas.
    """
    if mu <= 0.0 or lam <= 0.0:
        raise ValueError("Inverse-Gaussian requires mu>0 and lam>0")

    if _HAVE_SCIPY_STATS:
        y = invgauss.rvs(mu=mu / lam, random_state=rng)  # type: ignore
        return float(lam * y)

    return _rand_invgauss_msh(mu, lam, rng)


# =============================================================================
# GEV log-likelihood helpers (wrt location mu)
# =============================================================================
def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y | mu, sigma>0, xi) under GEV(mu, sigma, xi)."""
    if sigma <= 0.0 or not np.isfinite(mu) or not np.isfinite(y):
        return -np.inf

    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf

    if abs(xi) < 1e-8:  # Gumbel limit
        return -math.log(sigma) - z - math.exp(-z)

    return -math.log(sigma) - (1.0 + 1.0 / xi) * math.log(u) - u ** (-1.0 / xi)


def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi). Vectorised."""
    y = np.asarray(y, float)
    mu_vec = np.asarray(mu_vec, float)

    if sigma <= 0.0 or y.shape != mu_vec.shape or np.any(~np.isfinite(mu_vec)) or np.any(~np.isfinite(y)):
        return -np.inf

    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf

    if abs(xi) < 1e-8:
        return float(np.sum(-math.log(sigma) - z - np.exp(-z)))

    return float(np.sum(-math.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)))


def gev_score_hess_mu(y: float, mu: float, sigma: float, xi: float) -> tuple[float, float]:
    """
    First and second derivative of log f(y | mu, sigma, xi) w.r.t. mu.
    Returns (g, h) where g=dℓ/dmu, h=d²ℓ/dmu².

    Notes:
      - returns a weak negative curvature when outside support or numerically unstable,
        so Laplace weights remain positive.
    """
    if sigma <= 0.0 or not np.isfinite(mu) or not np.isfinite(y):
        return 0.0, -1e-8

    z = (y - mu) / sigma

    if abs(xi) < 1e-8:  # Gumbel
        e = math.exp(-z)
        g = (1.0 - e) / sigma
        h = -e / (sigma * sigma)
        if not np.isfinite(g) or not np.isfinite(h) or h >= 0.0:
            return 0.0, -1e-8
        return float(g), float(h)

    u = 1.0 + xi * z
    if u <= 0.0 or not np.isfinite(u):
        return 0.0, -1e-8

    # ℓ = -log σ - (1 + 1/ξ) log u - u^{-1/ξ}
    # g = (1/σ)[ (ξ + 1)/u - u^{-1/ξ - 1} ]
    g = ((xi + 1.0) / u - u ** (-1.0 / xi - 1.0)) / sigma

    # h = (1+ξ)/(σ²) [ ξ/u² - u^{-(1+2ξ)/ξ} ]
    h = (1.0 + xi) * (xi / (u * u) - u ** (-(1.0 + 2.0 * xi) / xi)) / (sigma * sigma)

    if not np.isfinite(g) or not np.isfinite(h) or h >= 0.0:
        return 0.0, -1e-8
    return float(g), float(h)
