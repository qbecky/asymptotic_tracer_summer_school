"""Second-fundamental-form estimation on triangle meshes.

:func:`face_shape_operators_quadratic_heightfield` (built on :func:`vertex_curvature_tensors`)
estimates a per-face curvature tensor by fitting an osculating jet
(quadratic height field) over each vertex's 2-ring in a frame aligned with
the angle-weighted normal (Cazals and Pouget style), and averaging the
resulting ambient tensors per face. This is the estimator used throughout
this tool.
"""
import numpy as np

from .geometry import face_frames, vertex_normals


def _vertex_rings(V, F, depth=2, min_size=6):
    """k-ring vertex neighborhoods (excluding the center vertex)."""
    n = len(V)
    adj = [set() for _ in range(n)]
    for tri in F:
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            adj[i].add(j)
            adj[j].add(i)
    rings = []
    for v in range(n):
        cur = {v} | adj[v]
        d = 1
        while d < depth or len(cur) - 1 < min_size:
            new = set()
            for u in cur:
                new |= adj[u]
            if not (new - cur):
                break
            cur |= new
            d += 1
        cur.discard(v)
        rings.append(np.fromiter(cur, int))
    return rings


def fit_quadratic_heightfield(uu, vv, hh):
    """Least-squares fit of a quadratic height field to local samples.

    Fits h(u, v) = a u + b v + (c u^2 + 2 e u v + f v^2)/2 to the sample
    points (uu, vv, hh) (the ring points' local coordinates and heights
    above the tangent plane).

    Parameters
    ----------
    uu, vv, hh : (k,) arrays
        Local u, v coordinates and height samples, k = ring size.

    Returns
    -------
    a, b, c, e, f : floats
        The fitted jet coefficients.
    """
    A = np.stack([uu, vv, 0.5 * uu * uu, uu * vv, 0.5 * vv * vv], axis=1)
    coef = np.linalg.lstsq(A, hh, rcond=None)[0]
    return coef[0], coef[1], coef[2], coef[3], coef[4]


def compute_first_second_fundamental_form_quadratic_heightfield(a, b, c, e, f):
    """First and second fundamental forms of the fitted quadratic graph.

    Parameters
    ----------
    a, b, c, e, f : floats
        Jet coefficients of h(u, v) = a u + b v + (c u^2 + 2 e u v + f v^2)/2.

    Returns
    -------
    I, II : (2, 2) arrays
        First and second fundamental form matrices.
    """
    w = np.sqrt(1.0 + a * a + b * b)
    Ifun = np.array([[1.0 + a * a, a * b], [a * b, 1.0 + b * b]])
    IIfun = np.array([[c, e], [e, f]]) / w
    return Ifun, IIfun


def vertex_curvature_tensors(V, F, depth=2):
    """Per-vertex ambient curvature tensors by osculating-jet fitting.

    For each vertex a quadratic height field
        h(u, v) = a u + b v + (c u^2 + 2 e u v + f v^2)/2
    is fitted over the k-ring in a frame aligned with the angle-weighted
    normal (Cazals & Pouget style). The Weingarten map of the fitted graph is
    diagonalized by the generalized symmetric eigenproblem II x = kappa I x,
    and returned as the ambient symmetric tensor
        T = kappa_1 e1 e1^T + kappa_2 e2 e2^T
    (3x3, with the fitted normal in its kernel). This estimator is
    second-order accurate on irregular meshes and remains consistent at
    boundaries, unlike normal-difference schemes with one-sided vertex
    normals.
    """
    VN0 = vertex_normals(V, F)
    rings = _vertex_rings(V, F, depth)
    n = len(V)
    T = np.zeros((n, 3, 3))
    N_fit = np.zeros((n, 3))
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    for v in range(n):
        n0 = VN0[v]
        u1 = np.cross(n0, ex if abs(n0[0]) < 0.9 else ey)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(n0, u1)
        d = V[rings[v]] - V[v]
        uu, vv, hh = d @ u1, d @ u2, d @ n0
        a, b, c, e, f = fit_quadratic_heightfield(uu, vv, hh)
        Ifun, IIfun = compute_first_second_fundamental_form_quadratic_heightfield(a, b, c, e, f)
        L = np.linalg.cholesky(Ifun)
        Li = np.linalg.inv(L)
        wk, Y = np.linalg.eigh(Li @ IIfun @ Li.T)
        X = Li.T @ Y
        phiu, phiv = u1 + a * n0, u2 + b * n0
        for i in (0, 1):
            di = X[0, i] * phiu + X[1, i] * phiv
            di /= np.linalg.norm(di)
            T[v] += wk[i] * np.outer(di, di)
        nf = np.cross(phiu, phiv)
        N_fit[v] = nf / np.linalg.norm(nf)
    return T, N_fit


def compute_shape_operator(t1, t2, Tf):
    """Restrict ambient curvature tensors to a per-face orthonormal frame.

    Parameters
    ----------
    t1, t2 : (m, 3) arrays
        Orthonormal in-plane frame vectors for each face.
    Tf : (m, 3, 3) array
        Ambient symmetric curvature tensor per face.

    Returns
    -------
    S : (m, 2, 2) array
        Symmetric shape operator in the (t1, t2) frame, with
        S[:, a, b] = t_a^T Tf t_b.
    """
    S = np.empty((Tf.shape[0], 2, 2))
    S[:, 0, 0] = np.einsum('ij,ijk,ik->i', t1, Tf, t1)
    S[:, 1, 1] = np.einsum('ij,ijk,ik->i', t2, Tf, t2)
    S[:, 0, 1] = S[:, 1, 0] = np.einsum('ij,ijk,ik->i', t1, Tf, t2)
    return S


def face_shape_operators_quadratic_heightfield(V, F, depth=2):
    """Per-face shape operator, averaged from its three corner vertices.

    Averages the per-vertex curvature tensor (see vertex_curvature_tensors)
    over each face's three corners, then restricts it to the face's tangent
    plane, giving a symmetric 2x2 tensor per face.
    """
    T, _ = vertex_curvature_tensors(V, F, depth)
    t1, t2, n = face_frames(V, F)
    Tf = T[F].mean(axis=1)
    S = compute_shape_operator(t1, t2, Tf)
    return S, (t1, t2, n)
