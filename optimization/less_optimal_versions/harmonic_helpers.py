from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
# =============================================================================
# Dummies (length s)  ↔  Harmonics helpers (module-level, not in the sampler)
# =============================================================================

def center_and_report_dummies_full(dummies: Sequence[float], tol: float = 1e-12) -> np.ndarray:
    """
    Accept a full-length seasonal vector (length = s). Report its mean & sum,
    warn if the sum isn't ~0, and return a centered (mean-zero) vector.
    """
    x = np.asarray(dummies, float)
    s = x.size
    m = float(x.mean()); sm = float(x.sum())
    print(f"[season] Provided dummies: mean={m:.6g}, sum={sm:.6g}")
    if abs(sm) > tol:
        print(f"[season][warn] Dummies do not sum to ~0 (|sum|>{tol:g}). They will be centered.")
    return x - m

def dummies_full_to_harmonics_fft(dummies_full_centered: Sequence[float],
                                  K: int,
                                  use_nyquist: bool) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Map mean-zero full-length dummies to (cos[K], sin[K], nyq?) using rFFT.
    """
    xc = np.asarray(dummies_full_centered, float)
    s = int(xc.size)
    even = (s % 2) == 0

    F = np.fft.rfft(xc)
    # decide truncation bounds for bins 1..K and optional Nyquist
    kmax = (min(K, (s // 2) - 1) if even else min(K, (s - 1) // 2))

    keep = np.zeros_like(F, dtype=bool)
    keep[0] = True
    if kmax >= 1:
        keep[1:kmax+1] = True
    nyq_val = None
    if even and use_nyquist:
        keep[s//2] = True

    F_trunc = np.where(keep, F, 0.0)

    a = np.zeros(K, float)
    b = np.zeros(K, float)
    for k in range(1, K + 1):
        if k <= kmax:
            a[k-1] =  2.0 * F_trunc[k].real
            b[k-1] = -2.0 * F_trunc[k].imag
        else:
            a[k-1] = 0.0
            b[k-1] = 0.0

    if even and use_nyquist:
        nyq_val = 2.0 * F_trunc[s//2].real

    return a, b, nyq_val

def harmonics_to_dummies_full_fft(s: int,
                                  cos_coefs: Sequence[float],
                                  sin_coefs: Sequence[float],
                                  use_nyquist: bool,
                                  nyq_coef: Optional[float]) -> np.ndarray:
    """
    Rebuild a mean-zero seasonal profile of length s from (cos, sin, nyq) using irFFT mapping.
    """
    even = (s % 2) == 0
    F = np.zeros(s // 2 + 1, dtype=np.complex128)
    F[0] = 0.0  # DC=0 for seasonal component

    K = int(len(cos_coefs))
    kmax = (min(K, (s // 2) - 1) if even else min(K, (s - 1) // 2))
    for k in range(1, kmax + 1):
        F[k] = (float(cos_coefs[k-1]) / 2.0) - 1j * (float(sin_coefs[k-1]) / 2.0)

    if even and use_nyquist and (nyq_coef is not None):
        F[s//2] = float(nyq_coef) / 2.0

    out = np.fft.irfft(F, n=s)
    # enforce exact mean-zero (tiny roundoff)
    out -= out.mean()
    return out
