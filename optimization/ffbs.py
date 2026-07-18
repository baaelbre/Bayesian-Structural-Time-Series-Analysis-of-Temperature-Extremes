# optimization/ffbs.py
from __future__ import annotations

import numpy as np
try:
    from .utils_2 import spd_solve, symmetrize  # type: ignore
except ImportError:
    from utils_2 import spd_solve, symmetrize  # type: ignore


def ffbs_gaussian_1d(
    y: np.ndarray,              # (T,)
    G: np.ndarray,              # (d,d)
    Q: np.ndarray,              # (d,d)
    H: np.ndarray,              # (1,d)
    R: float,                   # scalar observation variance
    *,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Standard FFBS for:
        z_t = G z_{t-1} + w_t,     w_t ~ N(0, Q)
        y_t = H z_t + e_t,         e_t ~ N(0, R)
    with 1-dim observations. Returns z_{0:T} with shape (T+1, d).
    """
    y = np.asarray(y, float)
    T = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, float).reshape(1, d)

    if rng is None:
        rng = np.random.default_rng()
    if m0 is None:
        m0 = np.zeros(d, float)
    if C0 is None:
        C0 = 1e-6 * np.eye(d)

    m = np.zeros((T + 1, d), float)
    C = np.zeros((T + 1, d, d), float)
    a = np.zeros((T + 1, d), float)
    Rm = np.zeros((T + 1, d, d), float)

    m[0] = m0
    C[0] = symmetrize(C0) + jitter * np.eye(d)

    # ---- forward filter ----
    for t in range(1, T + 1):
        a[t] = G @ m[t - 1]
        Rm[t] = symmetrize(G @ C[t - 1] @ G.T + Q) + jitter * np.eye(d)

        F = float(H @ Rm[t] @ H.T + float(R))
        if not np.isfinite(F) or F <= 0.0:
            F = float(H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T + float(R))

        K = (Rm[t] @ H.T) / F  # (d,1)
        v = float(y[t - 1] - (H @ a[t]))

        m[t] = a[t] + K[:, 0] * v
        C[t] = symmetrize(Rm[t] - K @ (H @ Rm[t])) + jitter * np.eye(d)

    # ---- backward sample ----
    z = np.zeros((T + 1, d), float)
    z[T] = rng.multivariate_normal(m[T], C[T])

    I = np.eye(d)
    for t in range(T - 1, -1, -1):
        Rinv = spd_solve(Rm[t + 1], I, jitter=jitter)
        J = C[t] @ G.T @ Rinv

        mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
        cov = symmetrize(C[t] - J @ Rm[t + 1] @ J.T)

        eigmin = float(np.linalg.eigvalsh(cov).min())
        if not np.isfinite(eigmin) or eigmin < jitter:
            cov = cov + (jitter - (eigmin if np.isfinite(eigmin) else 0.0)) * np.eye(d)

        z[t] = rng.multivariate_normal(mean, cov)

    return z


def ffbs_gaussian_1d_tvR(
    y: np.ndarray,              # (T,)
    G: np.ndarray,              # (d,d)
    Q: np.ndarray,              # (d,d)
    H: np.ndarray,              # (1,d)
    R_t: np.ndarray,            # (T,) time-varying observation variance
    *,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
    R_floor: float = 1e-12,
) -> np.ndarray:
    """
    FFBS for:
        z_t = G z_{t-1} + w_t,     w_t ~ N(0, Q)
        y_t = H z_t + e_t,         e_t ~ N(0, R_t)
    1-dim observations, time-varying obs variance R_t.
    Returns z_{0:T} with shape (T+1, d).
    """
    y = np.asarray(y, float)
    R_t = np.asarray(R_t, float)
    T = int(y.size)
    if int(R_t.size) != T:
        raise ValueError("R_t must have length T")
    d = int(G.shape[0])
    H = np.asarray(H, float).reshape(1, d)

    if rng is None:
        rng = np.random.default_rng()
    if m0 is None:
        m0 = np.zeros(d, float)
    if C0 is None:
        C0 = 1e-6 * np.eye(d)

    m = np.zeros((T + 1, d), float)
    C = np.zeros((T + 1, d, d), float)
    a = np.zeros((T + 1, d), float)
    Rm = np.zeros((T + 1, d, d), float)

    m[0] = m0
    C[0] = symmetrize(C0) + jitter * np.eye(d)

    for t in range(1, T + 1):
        Robs = float(R_t[t - 1])
        if not np.isfinite(Robs) or Robs <= 0.0:
            Robs = R_floor
        Robs = max(Robs, R_floor)

        a[t] = G @ m[t - 1]
        Rm[t] = symmetrize(G @ C[t - 1] @ G.T + Q) + jitter * np.eye(d)

        F = float(H @ Rm[t] @ H.T + Robs)
        if not np.isfinite(F) or F <= 0.0:
            F = float(H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T + Robs)

        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - (H @ a[t]))

        m[t] = a[t] + K[:, 0] * v
        C[t] = symmetrize(Rm[t] - K @ (H @ Rm[t])) + jitter * np.eye(d)

    z = np.zeros((T + 1, d), float)
    z[T] = rng.multivariate_normal(m[T], C[T])

    I = np.eye(d)
    for t in range(T - 1, -1, -1):
        Rinv = spd_solve(Rm[t + 1], I, jitter=jitter)
        J = C[t] @ G.T @ Rinv

        mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
        cov = symmetrize(C[t] - J @ Rm[t + 1] @ J.T)

        eigmin = float(np.linalg.eigvalsh(cov).min())
        if not np.isfinite(eigmin) or eigmin < jitter:
            cov = cov + (jitter - (eigmin if np.isfinite(eigmin) else 0.0)) * np.eye(d)

        z[t] = rng.multivariate_normal(mean, cov)

    return z


def ffbs_dlm_ncp(
    *,
    y: np.ndarray,                          # (T,)
    G_tilde: np.ndarray,                    # (d,d)
    Q_tilde: np.ndarray,                    # (d,d)
    sigma2: float,                          # scalar
    alpha0: float,
    beta0: float,
    t1: np.ndarray,                         # (T,) with entries 1..T (float)
    season_design: np.ndarray | None,        # (T,K) or None
    gamma0: np.ndarray | None,              # (K,) or None
    s_alpha: float,
    s_beta: float,
    s_gamma: float,
    idx_tilde_alpha: int,
    idx_A: int,
    idx_tilde_g_start: int,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Convenience wrapper for the DLM NCP measurement:

      c_t = alpha0 + beta0*t + S[t]@gamma0
      y'_t = y_t - c_t
      y'_t = s_alpha*tilde_alpha_t + s_beta*A_t + s_gamma*tilde_g1_t + eps_t, eps_t~N(0,sigma2)

    Returns z_{0:T}.
    """
    y = np.asarray(y, float)
    t1 = np.asarray(t1, float)

    c = alpha0 + beta0 * t1
    if season_design is not None and gamma0 is not None:
        c = c + (np.asarray(season_design, float) @ np.asarray(gamma0, float))

    y_center = y - c

    d = int(G_tilde.shape[0])
    H = np.zeros((1, d), float)
    H[0, idx_tilde_alpha] = s_alpha
    H[0, idx_A] = s_beta
    H[0, idx_tilde_g_start] = s_gamma

    return ffbs_gaussian_1d(
        y=y_center,
        G=G_tilde,
        Q=Q_tilde,
        H=H,
        R=float(sigma2),
        m0=m0,
        C0=C0,
        rng=rng,
        jitter=jitter,
    )


def ffbs_dgev_ncp_laplace(
    *,
    z_star: np.ndarray,                     # (T,) Laplace pseudo-obs for (alpha_t+g1_t) after subtracting S@gamma0
    R_t: np.ndarray,                        # (T,) Laplace pseudo variances
    G_tilde: np.ndarray,                    # (d,d)
    Q_tilde: np.ndarray,                    # (d,d)
    alpha0: float,
    beta0: float,
    t1: np.ndarray,                         # (T,)
    s_alpha: float,
    s_beta: float,
    s_gamma: float,
    idx_tilde_alpha: int,
    idx_A: int,
    idx_tilde_g_start: int,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Laplace-approximated FFBS for DGEV in NCP:

      z_star_t ≈ alpha_t + g1_t + e_t,   e_t ~ N(0, R_t)

      alpha_t + g1_t
        = (alpha0 + beta0*t) + [s_alpha*tilde_alpha_t + s_beta*A_t + s_gamma*tilde_g1_t]

    So define:
      y_ncp_t = z_star_t - (alpha0 + beta0*t)
      y_ncp_t = H z_t + e_t
    with time-varying variance R_t.

    Returns z_{0:T}.
    """
    z_star = np.asarray(z_star, float)
    R_t = np.asarray(R_t, float)
    t1 = np.asarray(t1, float)

    y_ncp = z_star - (alpha0 + beta0 * t1)

    d = int(G_tilde.shape[0])
    H = np.zeros((1, d), float)
    H[0, idx_tilde_alpha] = s_alpha
    H[0, idx_A] = s_beta
    H[0, idx_tilde_g_start] = s_gamma

    return ffbs_gaussian_1d_tvR(
        y=y_ncp,
        G=G_tilde,
        Q=Q_tilde,
        H=H,
        R_t=R_t,
        m0=m0,
        C0=C0,
        rng=rng,
        jitter=jitter,
    )


def ffbs_dgev_ncp_laplace_ekf(
    *,
    y: np.ndarray,                          # (T,) raw data
    offset: np.ndarray,                     # (T,) offset_t = alpha0 + beta0*t + S@gamma0
    G_tilde: np.ndarray,                    # (d,d)
    Q_tilde: np.ndarray,                    # (d,d)
    sigma: float,
    xi: float,
    s_alpha: float,
    s_beta: float,
    s_gamma: float,
    idx_tilde_alpha: int,
    idx_A: int,
    idx_tilde_g_start: int,
    # priors / numerics
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
    R_floor: float = 1e-12,
    # Laplace curvature safety
    h_min: float = -1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-pass EKF-like Laplace FFBS for DGEV in NCP.

    Returns:
      z     : (T+1, d) sampled NCP path
      z_mu  : (T,)   Laplace pseudo obs for mu_t
      R_t   : (T,)   Laplace pseudo variances
    """
    from utils import gev_score_hess_mu  # type: ignore

    y = np.asarray(y, float)
    offset = np.asarray(offset, float)
    T = int(y.size)
    if int(offset.size) != T:
        raise ValueError("offset must have length T")

    d = int(G_tilde.shape[0])
    if rng is None:
        rng = np.random.default_rng()
    if m0 is None:
        m0 = np.zeros(d, float)
    if C0 is None:
        C0 = 1e-6 * np.eye(d)

    # measurement matrix H (1 x d)
    H = np.zeros((1, d), float)
    H[0, idx_tilde_alpha] = float(s_alpha)
    H[0, idx_A] = float(s_beta)
    H[0, idx_tilde_g_start] = float(s_gamma)

    # forward storage
    m = np.zeros((T + 1, d), float)
    C = np.zeros((T + 1, d, d), float)
    a = np.zeros((T + 1, d), float)
    Rm = np.zeros((T + 1, d, d), float)

    # Laplace pseudo-data
    z_mu = np.zeros(T, float)
    R_t = np.zeros(T, float)

    m[0] = np.asarray(m0, float)
    C[0] = symmetrize(np.asarray(C0, float)) + jitter * np.eye(d)

    # ---- forward EKF-like Laplace filter ----
    for t in range(1, T + 1):
        # prediction
        a[t] = G_tilde @ m[t - 1]
        Rm[t] = symmetrize(G_tilde @ C[t - 1] @ G_tilde.T + Q_tilde) + jitter * np.eye(d)

        # NOTE: (H @ a[t]) has shape (1,), so use float(...) not [0,0]
        Ha = float(H @ a[t])
        mu0 = float(offset[t - 1] + Ha)

        g, h = gev_score_hess_mu(float(y[t - 1]), mu0, float(sigma), float(xi))

        # safety: enforce concavity + finite numbers
        if (not np.isfinite(g)) or (not np.isfinite(h)) or (h >= h_min):
            g = 0.0
            h = h_min

        w = max(-float(h), 1e-12)          # w = -h > 0
        Robs = 1.0 / w
        Robs = float(np.clip(Robs, R_floor, 1e12))

        z_mu[t - 1] = mu0 - float(g) / float(h)   # Newton target
        R_t[t - 1] = Robs

        # linearised measurement: y_lin = z_mu - offset = H z + e
        y_lin = float(z_mu[t - 1] - offset[t - 1])

        # Kalman update
        F = float(H @ Rm[t] @ H.T + Robs)  # (1,1) -> float OK
        if not np.isfinite(F) or F <= 0.0:
            F = float(H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T + Robs)

        K = (Rm[t] @ H.T) / F                        # (d,1)
        v = float(y_lin - float(H @ a[t]))           # FIX: no [0,0]

        m[t] = a[t] + K[:, 0] * v
        C[t] = symmetrize(Rm[t] - K @ (H @ Rm[t])) + jitter * np.eye(d)

    # ---- backward simulation smoother ----
    z = np.zeros((T + 1, d), float)
    z[T] = rng.multivariate_normal(m[T], C[T])

    I = np.eye(d)
    for t in range(T - 1, -1, -1):
        Rinv = spd_solve(Rm[t + 1], I, jitter=jitter)
        J = C[t] @ G_tilde.T @ Rinv

        mean = m[t] + J @ (z[t + 1] - (G_tilde @ m[t]))
        cov = symmetrize(C[t] - J @ Rm[t + 1] @ J.T)

        eigmin = float(np.linalg.eigvalsh(cov).min())
        if not np.isfinite(eigmin) or eigmin < jitter:
            cov = cov + (jitter - (eigmin if np.isfinite(eigmin) else 0.0)) * np.eye(d)

        z[t] = rng.multivariate_normal(mean, cov)

    return z, z_mu, R_t

