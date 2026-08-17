"""Construction of the two asymptotic line fields on a hyperbolic mesh region.

At a hyperbolic point with principal curvatures kappa_1 > 0 > kappa_2 and
principal directions e_1, e_2, the asymptotic directions are

    d_{+/-} = cos(phi) e_1 +/- sin(phi) e_2,   tan^2(phi) = -kappa_1 / kappa_2.

Since a connected region with K < 0 contains no umbilics, the two families are
globally separable; :func:`_label_families` propagates a consistent per-face
labeling (family 0 / family 1) by breadth-first search over face adjacency,
matching direction pairs across edges as unoriented lines.
"""
from collections import deque

import numpy as np

from .curvature import face_shape_operators_quadratic_heightfield
from .geometry import face_neighbors, normalize


class AsymptoticField:
    """Per-face asymptotic direction pairs (two labeled families).

    Attributes
    ----------
    dirs : (m, 2, 3) unit vectors; ``dirs[f, l]`` spans the line of family
        ``l`` in the plane of face f (sign is arbitrary). Zero on
        non-hyperbolic faces.
    K : (m,) Gaussian curvature (det of the estimated shape operator).
    hyperbolic : (m,) boolean mask, K < -eps_K.
    """

    def __init__(self, V, F, dirs, K, hyperbolic):
        self.V, self.F = V, F
        self.dirs = dirs
        self.K = K
        self.hyperbolic = hyperbolic

    def crossing_angles(self):
        """Angle 2*phi between the two families, in degrees (hyperbolic faces)."""
        c = np.abs(np.einsum('ij,ij->i',
                             self.dirs[:, 0], self.dirs[:, 1]))
        ang = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        return ang[self.hyperbolic]


def _label_families(dirs, hyperbolic, nbrs):
    """Propagate a globally consistent family labeling by BFS (in place)."""
    m = len(dirs)
    visited = np.zeros(m, dtype=bool)
    for seed in np.where(hyperbolic)[0]:
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for g in nbrs[f]:
                if visited[g] or not hyperbolic[g]:
                    continue
                keep = abs(dirs[f, 0] @ dirs[g, 0]) + abs(dirs[f, 1] @ dirs[g, 1])
                swap = abs(dirs[f, 0] @ dirs[g, 1]) + abs(dirs[f, 1] @ dirs[g, 0])
                if swap > keep:
                    dirs[g] = dirs[g, ::-1].copy()
                visited[g] = True
                queue.append(g)
    return dirs


def from_estimated_curvature(V, F, eps_K=6.4e-11):
    """Asymptotic field from the discrete per-face curvature tensor.

    Uses per-vertex osculating-jet fitting
    (face_shape_operators_quadratic_heightfield), which is robust on
    irregular meshes and near boundaries.

    K has units of 1/length^2; eps_K's default is calibrated for meshes on
    the order of this repo's own (mm-scale) test surfaces.
    """
    S, (t1, t2, n) = face_shape_operators_quadratic_heightfield(V, F)
    w, U = np.linalg.eigh(S)                    # ascending eigenvalues
    K = w[:, 0] * w[:, 1]
    hyperbolic = K < -eps_K
    kneg = np.where(hyperbolic, w[:, 0], -1.0)
    kpos = np.where(hyperbolic, w[:, 1], 1.0)
    phi = np.arctan(np.sqrt(np.maximum(-kpos / kneg, 0.0)))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    e1, e2 = U[:, :, 1], U[:, :, 0]             # 2D principal directions
    d0_2, d1_2 = c * e1 + s * e2, c * e1 - s * e2

    def lift(d2):
        return normalize(d2[:, 0:1] * t1 + d2[:, 1:2] * t2)

    dirs = np.stack([lift(d0_2), lift(d1_2)], axis=1)
    dirs[~hyperbolic] = 0.0
    _label_families(dirs, hyperbolic, face_neighbors(F))
    return AsymptoticField(V, F, dirs, K, hyperbolic)
