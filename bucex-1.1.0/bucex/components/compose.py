from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

from .base import Array, Component, ParamDict


@dataclass(frozen=True)
class ComposedMatrices:
    T: Array
    R: Array
    Q: Array
    c: Array
    Z: Array
    d: Array
    state_names: Tuple[str, ...]


def _blkdiag_rect(mats: List[Array]) -> Array:
    m_tot = sum(A.shape[0] for A in mats)
    r_tot = sum(A.shape[1] for A in mats)
    out = np.zeros((m_tot, r_tot), dtype=float)
    i = 0
    j = 0
    for A in mats:
        m_k, r_k = A.shape
        out[i : i + m_k, j : j + r_k] = A
        i += m_k
        j += r_k
    return out


def compose_initial_state(
    components: Sequence[Component],
    params: ParamDict,
) -> Tuple[Array, Array]:
    m0_blocks: List[Array] = []
    P0_blocks: List[Array] = []

    for comp in components:
        if comp.spec.state_dim == 0:
            continue
        m0_k, v0_k = comp.initial_mean_var(params)
        m0_k = np.asarray(m0_k, dtype=float).reshape(comp.spec.state_dim)
        v0_k = np.asarray(v0_k, dtype=float).reshape(comp.spec.state_dim)
        if np.any(v0_k < 0.0):
            raise ValueError(f"Negative initial variance in component '{comp.spec.name}'.")
        m0_blocks.append(m0_k)
        P0_blocks.append(np.diag(v0_k))

    if not m0_blocks:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

    return np.concatenate(m0_blocks), block_diag(*P0_blocks)




def compose_system_only(
    components: Sequence[Component],
    t: int,
    params: ParamDict,
) -> tuple[Array, Array, Array, Array, Tuple[str, ...]]:
    Tblk: List[Array] = []
    Rblk: List[Array] = []
    Qblk: List[Array] = []
    cblk: List[Array] = []
    names: List[str] = []

    for comp in components:
        if comp.spec.state_dim == 0:
            continue
        Tk, Rk, Qk, ck = comp.system_matrices(t=t, params=params)
        Tk = np.asarray(Tk, dtype=float)
        Rk = np.asarray(Rk, dtype=float)
        Qk = np.asarray(Qk, dtype=float)
        ck = np.asarray(ck, dtype=float).reshape(comp.spec.state_dim)
        Tblk.append(Tk)
        Rblk.append(Rk)
        Qblk.append(Qk)
        cblk.append(ck)
        names.extend(comp.spec.state_names)

    T = block_diag(*Tblk) if Tblk else np.zeros((0, 0), dtype=float)
    Q = block_diag(*Qblk) if Qblk else np.zeros((0, 0), dtype=float)
    R = _blkdiag_rect(Rblk) if Rblk else np.zeros((0, 0), dtype=float)
    c = np.concatenate(cblk) if cblk else np.zeros((0,), dtype=float)
    return T, R, Q, c, tuple(names)
def compose_components(
    components: Sequence[Component],
    t: int,
    params: ParamDict,
    exog_t: Optional[Array] = None,
) -> ComposedMatrices:
    Tblk: List[Array] = []
    Rblk: List[Array] = []
    Qblk: List[Array] = []
    cblk: List[Array] = []
    Zblk: List[Array] = []

    d = np.zeros((1,), dtype=float)
    names: List[str] = []

    for comp in components:
        Zk, dk = comp.design_matrices(t=t, params=params, exog_t=exog_t)
        Zk = np.asarray(Zk, dtype=float)
        dk = np.asarray(dk, dtype=float).reshape(1)
        d += dk

        if comp.spec.state_dim == 0:
            continue

        Tk, Rk, Qk, ck = comp.system_matrices(t=t, params=params)
        Tk = np.asarray(Tk, dtype=float)
        Rk = np.asarray(Rk, dtype=float)
        Qk = np.asarray(Qk, dtype=float)
        ck = np.asarray(ck, dtype=float).reshape(comp.spec.state_dim)

        Tblk.append(Tk)
        Rblk.append(Rk)
        Qblk.append(Qk)
        cblk.append(ck)
        Zblk.append(Zk)
        names.extend(comp.spec.state_names)

    T = block_diag(*Tblk) if Tblk else np.zeros((0, 0), dtype=float)
    Q = block_diag(*Qblk) if Qblk else np.zeros((0, 0), dtype=float)
    R = _blkdiag_rect(Rblk) if Rblk else np.zeros((0, 0), dtype=float)
    c = np.concatenate(cblk) if cblk else np.zeros((0,), dtype=float)
    Z = np.concatenate(Zblk, axis=1) if Zblk else np.zeros((1, 0), dtype=float)

    return ComposedMatrices(T=T, R=R, Q=Q, c=c, Z=Z, d=d, state_names=tuple(names))
