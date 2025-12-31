# optimization/utils.py
from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

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
# RNG helpers (SAFE inverse-Gaussian)
# =============================================================================

# Hard safety bounds to prevent under/overflow / invalid SciPy params.
# These are *only* used when the chain goes numerically insane.
_IG_MU_MIN, _IG_MU_MAX = 1e-16, 1e16
_IG_LAM_MIN, _IG_LAM_MAX = 1e-16, 1e16
_IG_X_MIN, _IG_X_MAX = 1e-16, 1e16


def _clip_pos_finite(x: float, lo: float, hi: float, fallback: float) -> float:
    x = float(x)
    if (not np.isfinite(x)) or (x <= 0.0):
        return float(fallback)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _rand_invgauss_msh(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Michael–Schucany–Haas method for IG(mu, lam).
    (Assumes mu>0 and lam>0 and finite.)
    """
    if mu <= 0.0 or lam <= 0.0 or (not np.isfinite(mu)) or (not np.isfinite(lam)):
        raise ValueError("Inverse-Gaussian requires finite mu>0 and lam>0")

    v = float(rng.normal())
    y = v * v
    mu2 = mu * mu
    term = mu2 * y

    rad = 4.0 * mu * lam * y + term * y
    if (not np.isfinite(rad)) or (rad <= 0.0):
        raise FloatingPointError("IG MSH radicand invalid")

    x = mu + term / (2.0 * lam) - (mu / (2.0 * lam)) * math.sqrt(rad)

    if (not np.isfinite(x)) or (x <= 0.0):
        raise FloatingPointError("IG MSH produced nonpositive draw")

    u = float(rng.random())
    if u <= mu / (mu + x):
        return float(x)
    return float(mu2 / x)


def rand_invgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    SAFE draw X ~ IG(mu, lam).

    Same target as before, but **never raises**:
      - clamps invalid / extreme parameters into a safe positive range
      - catches SciPy/MSH numeric failures
      - guarantees a finite positive return
    """
    mu0 = float(mu)
    lam0 = float(lam)

    mu_safe = _clip_pos_finite(mu0, _IG_MU_MIN, _IG_MU_MAX, fallback=_IG_MU_MIN)
    lam_safe = _clip_pos_finite(lam0, _IG_LAM_MIN, _IG_LAM_MAX, fallback=_IG_LAM_MIN)

    fallback_x = max(mu_safe, _IG_X_MIN)

    try:
        if _HAVE_SCIPY_STATS:
            # SciPy: if Y ~ IG(mu/lam, 1), then X = lam * Y ~ IG(mu, lam).
            shape = mu_safe / lam_safe
            shape = _clip_pos_finite(shape, _IG_MU_MIN, _IG_MU_MAX, fallback=_IG_MU_MIN)
            y = invgauss.rvs(mu=shape, random_state=rng)  # type: ignore
            x = float(lam_safe * float(y))
        else:
            x = float(_rand_invgauss_msh(mu_safe, lam_safe, rng))
    except Exception:
        return float(_clip_pos_finite(fallback_x, _IG_X_MIN, _IG_X_MAX, fallback=_IG_X_MIN))

    return float(_clip_pos_finite(x, _IG_X_MIN, _IG_X_MAX, fallback=fallback_x))


# =============================================================================
# GEV log-likelihood helpers (overflow-safe)
# =============================================================================
_LOG_EXP_MAX = 700.0  # ~ log(max float)


def _safe_pow(u: float, p: float) -> float:
    """
    Return u**p using exp(p*log(u)) with overflow protection.
    If it would overflow -> +inf; if it would underflow -> 0.
    Requires u>0.
    """
    lu = math.log(u)
    t = p * lu
    if t > _LOG_EXP_MAX:
        return math.inf
    if t < -_LOG_EXP_MAX:
        return 0.0
    return math.exp(t)


def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y | mu, sigma>0, xi) under GEV(mu, sigma, xi)."""
    if sigma <= 0.0 or (not np.isfinite(mu)) or (not np.isfinite(y)) or (not np.isfinite(xi)):
        return -np.inf

    z = (y - mu) / sigma
    if abs(xi) < 1e-8:  # Gumbel
        # log f = -log σ - z - exp(-z)
        ez = math.exp(-z) if (-z) < _LOG_EXP_MAX else math.inf
        if not np.isfinite(ez):
            return -np.inf
        return -math.log(sigma) - z - ez

    u = 1.0 + xi * z
    if u <= 0.0 or (not np.isfinite(u)):
        return -np.inf

    # term = u^{-1/xi} = exp((-1/xi) log u), overflow -> -inf logpdf
    t = (-1.0 / xi) * math.log(u)
    if t > _LOG_EXP_MAX:
        return -np.inf
    term = math.exp(t)

    return -math.log(sigma) - (1.0 + 1.0 / xi) * math.log(u) - term


def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi). Vectorised + overflow-safe."""
    y = np.asarray(y, float)
    mu_vec = np.asarray(mu_vec, float)

    if sigma <= 0.0 or y.shape != mu_vec.shape or np.any(~np.isfinite(mu_vec)) or np.any(~np.isfinite(y)):
        return -np.inf
    if not np.isfinite(xi):
        return -np.inf

    z = (y - mu_vec) / sigma

    if abs(xi) < 1e-8:
        # log f = -log σ - z - exp(-z)
        ez = np.exp(np.clip(-z, -_LOG_EXP_MAX, _LOG_EXP_MAX))
        # if -z exceeded +LOG_EXP_MAX, ez is huge but finite; loglike is then very negative (ok)
        return float(np.sum(-math.log(sigma) - z - ez))

    u = 1.0 + xi * z
    if np.any(u <= 0.0) or np.any(~np.isfinite(u)):
        return -np.inf

    logu = np.log(u)
    t = (-1.0 / xi) * logu
    if np.any(t > _LOG_EXP_MAX):
        return -np.inf
    term = np.exp(np.clip(t, -_LOG_EXP_MAX, _LOG_EXP_MAX))

    return float(np.sum(-math.log(sigma) - (1.0 + 1.0 / xi) * logu - term))


def gev_score_hess_mu(y: float, mu: float, sigma: float, xi: float) -> tuple[float, float]:
    """
    First and second derivative of log f(y | mu, sigma, xi) w.r.t. mu.
    Returns (g, h) where g=dℓ/dmu, h=d²ℓ/dmu².

    Always returns a finite g and a strictly negative h (fallback) when unstable,
    so Laplace weights remain positive and finite.
    """
    if sigma <= 0.0 or (not np.isfinite(mu)) or (not np.isfinite(y)) or (not np.isfinite(xi)):
        return 0.0, -1e-8

    z = (y - mu) / sigma

    if abs(xi) < 1e-8:  # Gumbel
        # g = (1 - exp(-z))/σ ; h = -exp(-z)/σ²
        ez = math.exp(-z) if (-z) < _LOG_EXP_MAX else math.inf
        if not np.isfinite(ez):
            return 0.0, -1e-8
        g = (1.0 - ez) / sigma
        h = -ez / (sigma * sigma)
        if (not np.isfinite(g)) or (not np.isfinite(h)) or (h >= 0.0):
            return 0.0, -1e-8
        return float(g), float(h)

    u = 1.0 + xi * z
    if u <= 0.0 or (not np.isfinite(u)):
        return 0.0, -1e-8

    logu = math.log(u)

    # a = (-1/xi - 1) so u^a = exp(a logu)
    a = (-1.0 / xi) - 1.0
    t1 = a * logu
    if t1 > _LOG_EXP_MAX:
        # u^a overflow -> g,h blow -> fallback
        return 0.0, -1e-8
    u_a = math.exp(t1)

    g = ((xi + 1.0) / u - u_a) / sigma

    # h = (1+ξ)/(σ²) [ ξ/u² - u^{-(1+2ξ)/ξ} ]
    b = (-(1.0 + 2.0 * xi) / xi)
    t2 = b * logu
    if t2 > _LOG_EXP_MAX:
        return 0.0, -1e-8
    u_b = math.exp(t2)

    h = (1.0 + xi) * (xi / (u * u) - u_b) / (sigma * sigma)

    if (not np.isfinite(g)) or (not np.isfinite(h)) or (h >= 0.0):
        return 0.0, -1e-8
    return float(g), float(h)


# =============================================================================
# CLI parsing helpers
# =============================================================================
def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def parse_date(s: str | None) -> datetime:
    if not s:
        return datetime.today()
    parts = [int(p) for p in str(s).split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def parse_csv_floats(s: Optional[str], expected_len: Optional[int] = None) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    vals = [float(z) for z in ss.split(",")]
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(vals)}")
    return vals
