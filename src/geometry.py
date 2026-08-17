"""Triangle-mesh geometry utilities: normals, frames, adjacency, barycentrics."""
import numpy as np


def normalize(a, axis=-1, eps=1e-15):
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, eps)


def face_normals(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return normalize(np.cross(p1 - p0, p2 - p0))


def vertex_normals(V, F):
    """Angle-weighted vertex normals."""
    fn = face_normals(V, F)
    N = np.zeros_like(V)
    P = V[F]
    for k in range(3):
        a = normalize(P[:, (k + 1) % 3] - P[:, k])
        b = normalize(P[:, (k + 2) % 3] - P[:, k])
        ang = np.arccos(np.clip(np.einsum('ij,ij->i', a, b), -1.0, 1.0))
        np.add.at(N, F[:, k], fn * ang[:, None])
    return normalize(N)


def face_frames(V, F):
    """Per-face orthonormal frame (t1, t2, n), with t1 along the first edge."""
    n = face_normals(V, F)
    t1 = normalize(V[F[:, 1]] - V[F[:, 0]])
    t2 = np.cross(n, t1)
    return t1, t2, n


def build_edge_map(F):
    """Map from sorted vertex pair to the list of incident face indices."""
    emap = {}
    for f in range(len(F)):
        i, j, k = F[f]
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            emap.setdefault(key, []).append(f)
    return emap


def face_neighbors(F, emap=None):
    """List of edge-adjacent face indices per face."""
    if emap is None:
        emap = build_edge_map(F)
    nbrs = [[] for _ in range(len(F))]
    for faces in emap.values():
        if len(faces) == 2:
            f, g = faces
            nbrs[f].append(g)
            nbrs[g].append(f)
    return nbrs


def barycentric(p, a, b, c):
    """Barycentric coordinates of p with respect to triangle (a, b, c)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1.0 - v - w, v, w])


def rotate_between(d, n_from, n_to):
    """Rotate vector d by the minimal rotation taking n_from onto n_to (Rodrigues).

    Used as the hinge map transporting a direction across a mesh edge.
    """
    axis = np.cross(n_from, n_to)
    s = np.linalg.norm(axis)
    c = float(np.dot(n_from, n_to))
    if s < 1e-14:
        return d.copy()
    axis = axis / s
    return d * c + np.cross(axis, d) * s + axis * float(np.dot(axis, d)) * (1.0 - c)


def nearest_face_xy(V, F, xy):
    """Index of the face whose centroid is closest to xy in the parameter plane."""
    cen = V[F].mean(axis=1)
    d2 = (cen[:, 0] - xy[0]) ** 2 + (cen[:, 1] - xy[1]) ** 2
    return int(np.argmin(d2))
