"""Crossing detection between two families of curves.

Generalizes the closest-approach numerics used by exp/04_export_waffle.py's
single-crossing detector (vectorized clamped-least-squares segment-segment
distance) to a full edge-pair distance matrix, so a curve pair can register
more than one crossing -- necessary since a rail crosses many rung curves.
"""
from dataclasses import dataclass, field

import numpy as np

from .curves import arclength
from .geometry import normalize


@dataclass
class Crossing:
    a_idx: int              # index of the family-A curve
    b_idx: int               # index of the family-B curve
    s_a: float               # arclength of the crossing on curve A (pre-redisc.)
    s_b: float                # arclength of the crossing on curve B (pre-redisc.)
    position: np.ndarray
    tangent_a: np.ndarray
    tangent_b: np.ndarray
    gap: float                # closest-approach distance (diagnostic)
    edge_a: int = field(default=-1)  # filled in once curves are re-discretized
    edge_b: int = field(default=-1)


def _segment_distance_matrix(P, Q):
    """Closest approach between every edge of P and every edge of Q.

    Returns (dist, s, t), each (nA-1, nB-1): dist is the closest-approach
    distance, s/t the clamped [0,1] parameters on edges of P/Q respectively.
    """
    p0, d1 = P[:-1], P[1:] - P[:-1]
    q0, d2 = Q[:-1], Q[1:] - Q[:-1]
    r = p0[:, None, :] - q0[None, :, :]
    a = np.einsum('ik,ik->i', d1, d1)[:, None]
    e = np.einsum('jk,jk->j', d2, d2)[None, :]
    f = np.einsum('ijk,jk->ij', r, d2)
    c = np.einsum('ijk,ik->ij', r, d1)
    b = np.einsum('ik,jk->ij', d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, (b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1.0)
    t = np.clip(t, 0.0, 1.0)
    s = (b * t - c) / np.where(a > 1e-12, a, 1.0)
    s = np.clip(s, 0.0, 1.0)
    cp = p0[:, None, :] + s[..., None] * d1[:, None, :]
    cq = q0[None, :, :] + t[..., None] * d2[None, :, :]
    dist = np.linalg.norm(cp - cq, axis=2)
    return dist, s, t


def _local_minima(dist, tol, min_edge_gap=2):
    """Greedily extract well-separated (i, j) pairs with dist <= tol, so one
    geometric crossing isn't double-counted across adjacent edge cells."""
    ni, nj = dist.shape
    claimed_i = np.zeros(ni, dtype=bool)
    claimed_j = np.zeros(nj, dtype=bool)
    out = []
    for idx in np.argsort(dist, axis=None):
        i, j = divmod(int(idx), nj)
        if dist[i, j] > tol:
            break
        if claimed_i[i] or claimed_j[j]:
            continue
        out.append((i, j))
        claimed_i[max(0, i - min_edge_gap):i + min_edge_gap + 1] = True
        claimed_j[max(0, j - min_edge_gap):j + min_edge_gap + 1] = True
    return out


def find_crossings(curves_a, curves_b, tol, min_edge_gap=2):
    """All well-separated crossings between two families of (points, normals)
    curves. `edge_a`/`edge_b` are left at -1 -- filled in once the curves are
    re-discretized (see rod_assembly_io.build_rod_assembly_io)."""
    out = []
    for a_idx, (PA, _NA) in enumerate(curves_a):
        sA = arclength(PA)
        for b_idx, (PB, _NB) in enumerate(curves_b):
            sB = arclength(PB)
            dist, s, t = _segment_distance_matrix(PA, PB)
            for i, j in _local_minima(dist, tol, min_edge_gap):
                si, tj = s[i, j], t[i, j]
                cpA = PA[i] + si * (PA[i + 1] - PA[i])
                cpB = PB[j] + tj * (PB[j + 1] - PB[j])
                out.append(Crossing(
                    a_idx=a_idx, b_idx=b_idx,
                    s_a=float(sA[i] + si * (sA[i + 1] - sA[i])),
                    s_b=float(sB[j] + tj * (sB[j + 1] - sB[j])),
                    position=0.5 * (cpA + cpB),
                    tangent_a=normalize(PA[i + 1] - PA[i]),
                    tangent_b=normalize(PB[j + 1] - PB[j]),
                    gap=float(dist[i, j]),
                ))
    return out
