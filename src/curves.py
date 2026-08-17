"""Post-processing of traced polylines.

Provides arclength resampling, barycentrically interpolated surface normals,
discrete Darboux-frame quantities (kappa_g, kappa_n, tau_g), a straightness
metric, and seed generation along a rail curve for net construction.

For an asymptotic curve the normal curvature vanishes, kappa_n = 0, and the
Beltrami--Enneper theorem gives tau_g^2 = -K; both identities are used as
quantitative validation criteria in exp/02_validate.py.
"""
import warnings

import numpy as np

from .geometry import barycentric, normalize, vertex_normals


def arclength(P):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interp_at(P, N, s, si):
    Pr = np.stack([np.interp(si, s, P[:, k]) for k in range(3)], axis=1)
    Nr = normalize(np.stack([np.interp(si, s, N[:, k]) for k in range(3)],
                            axis=1))
    return Pr, Nr


def sample_normals(P, faces, V, F, VN=None):
    """Interpolated vertex normals at polyline samples (PL normal field)."""
    if VN is None:
        VN = vertex_normals(V, F)
    N = np.zeros_like(P)
    for i in range(len(P)):
        f = faces[min(i, len(faces) - 1)]
        a, b, c = V[F[f]]
        w = barycentric(P[i], a, b, c)
        N[i] = w @ VN[F[f]]
    return normalize(N)


def resample(P, N, h):
    """Uniform arclength resampling of points and normals at spacing h."""
    s = arclength(P)
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    P, N, s = P[keep], N[keep], s[keep]
    if s[-1] < 2.0 * h or len(P) < 4:
        return P, N, s
    si = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / h)) + 1, 4))
    Pr, Nr = _interp_at(P, N, s, si)
    return Pr, Nr, si


def resample_with_joints(P, N, h, mandatory_s, min_gap=4):
    """Arclength resampling at spacing h, guaranteeing every arclength value
    in `mandatory_s` lands exactly at the midpoint of one edge of the result
    (used to place rod_assembly joints at a real vertex pair straddling each
    curve crossing, rather than merely near a vertex).

    With `mandatory_s` empty this is identical to resample(P, N, h).

    Interior spans (between two constrained edges) get at least `min_gap`
    free edges, matching rod_assembly's minimum joint-separation constraint.
    A mandatory arclength closer than 0.15*h to either endpoint is dropped
    (with a warning) rather than fully generalized.

    Returns (Pr, Nr, si, edge_idx): edge_idx[k] is the 0-based edge of the
    result whose midpoint is mandatory_s[k] (same order as the input, -1 for
    any dropped value).
    """
    s = arclength(P)
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    P, N, s = P[keep], N[keep], s[keep]
    L = s[-1]

    mandatory_s = np.asarray(mandatory_s, dtype=float)
    edge_idx = np.full(len(mandatory_s), -1, dtype=int)
    if len(mandatory_s) == 0:
        Pr, Nr, si = resample(P, N, h)
        return Pr, Nr, si, edge_idx

    order = np.argsort(mandatory_s)
    eps = 0.15 * h
    valid = (mandatory_s[order] > eps) & (mandatory_s[order] < L - eps)
    if not np.all(valid):
        warnings.warn(f"resample_with_joints: dropping {int((~valid).sum())} "
                      "crossing(s) too close to a curve endpoint")
    order = order[valid]
    s_sorted = mandatory_s[order]
    K = len(s_sorted)
    if K == 0:
        Pr, Nr, si = resample(P, N, h)
        return Pr, Nr, si, edge_idx

    delta = np.minimum(h, np.minimum(2.0 * s_sorted, 2.0 * (L - s_sorted)))
    lo, hi = s_sorted - delta / 2, s_sorted + delta / 2
    for k in range(K - 1):
        if hi[k] > lo[k + 1]:
            mid = 0.5 * (hi[k] + lo[k + 1])
            hi[k] = lo[k + 1] = mid

    grid = [0.0]
    edge_idx_sorted = np.empty(K, dtype=int)
    prev = 0.0
    for k in range(K):
        span = lo[k] - prev
        if span > 1e-12:
            n = max(1, round(span / h)) if k == 0 else max(min_gap, round(span / h))
            grid.extend(np.linspace(prev, lo[k], n + 1)[1:].tolist())
        elif abs(grid[-1] - lo[k]) > 1e-12:
            grid.append(lo[k])
        edge_idx_sorted[k] = len(grid) - 1
        grid.append(hi[k])
        prev = hi[k]
    span = L - prev
    if span > 1e-12:
        n = max(1, round(span / h))
        grid.extend(np.linspace(prev, L, n + 1)[1:].tolist())
    elif abs(grid[-1] - L) > 1e-12:
        grid.append(L)

    si = np.array(grid)
    Pr, Nr = _interp_at(P, N, s, si)
    edge_idx[order] = edge_idx_sorted
    return Pr, Nr, si, edge_idx


def darboux(P, N):
    """Discrete Darboux quantities by central differences.

    With unit tangent T, surface normal N and side vector U = N x T:
        dN/ds = -kappa_n T - tau_g U,     dT/ds . U = kappa_g.
    Endpoint values are copied from their interior neighbors.
    """
    s = arclength(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    T = normalize(T)
    U = normalize(np.cross(N, T))
    dN = np.zeros_like(P)
    dT = np.zeros_like(P)
    ds = (s[2:] - s[:-2])[:, None]
    dN[1:-1] = (N[2:] - N[:-2]) / ds
    dT[1:-1] = (T[2:] - T[:-2]) / ds
    kn = -np.einsum('ij,ij->i', dN, T)
    tg = -np.einsum('ij,ij->i', dN, U)
    kg = np.einsum('ij,ij->i', dT, U)
    for arr in (kn, tg, kg):
        arr[0], arr[-1] = arr[1], arr[-2]
    return dict(s=s, T=T, U=U, kappa_n=kn, tau_g=tg, kappa_g=kg)


def line_deviation(P):
    """Maximum distance of the polyline to its total-least-squares line."""
    Q = P - P.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    R = Q - np.outer(Q @ Vt[0], Vt[0])
    return float(np.linalg.norm(R, axis=1).max())


def seeds_along_curve(result, spacing, margin=None):
    """Seeds (face, point) sampled at uniform arclength along a rail curve."""
    P, Fc = result.points, result.faces
    s = arclength(P)
    if margin is None:
        margin = 0.5 * spacing
    targets = np.arange(margin, s[-1] - margin + 1e-12, spacing)
    seeds = []
    for st in targets:
        idx = int(np.clip(np.searchsorted(s, st) - 1, 0, len(P) - 2))
        ds = s[idx + 1] - s[idx]
        lam = 0.5 if ds < 1e-12 else float(np.clip((st - s[idx]) / ds,
                                                   0.05, 0.95))
        seeds.append((int(Fc[idx]), (1.0 - lam) * P[idx] + lam * P[idx + 1]))
    return seeds


def seeds_along_curve_n(result, n):
    """Like seeds_along_curve, but `n` (an integer count) is given directly
    instead of an arclength spacing: the rail is divided into n+1 equal
    segments and a seed is placed at each of the n interior break points
    (so this is exactly seeds_along_curve with spacing = margin = the
    resulting equal segment length, i.e. an evenly-spaced linspace)."""
    if n <= 0:
        return []
    total = arclength(result.points)[-1]
    spacing = total / (n + 1)
    return seeds_along_curve(result, spacing, margin=spacing)
