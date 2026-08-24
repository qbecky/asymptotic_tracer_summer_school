"""Face-walking tracer for asymptotic line fields on triangle meshes.

The tracer advances a state (face f, point p in f, direction d in the plane of
f). In each face both of the field's candidate directions (field.dirs[f, 0]
and field.dirs[f, 1]) are re-evaluated, and the incoming direction picks which
one to follow -- whichever is closer as an unoriented line -- and its sign.
That pick is overridden by the other candidate only when it clearly heads
back across the edge just crossed (i.e. into the previous face), by more than
a cosine margin (Tracer.backtrack_margin) so ordinary noise can't trip it;
this check itself is skipped when the incoming direction is nearly parallel
to that edge (a grazing crossing, Tracer.graze_eps), since the component it's
based on is floating-point noise rather than signal there. Neither candidate
is ever integrated, so no extrinsic drift is accumulated. This makes
the walk robust to any local error in field._label_families's precomputed,
globally-propagated family labeling: branch continuity is re-derived from the
actual incoming direction at every face, not trusted from that precomputed
label. The exit point is obtained by exact ray/edge intersection in the 2D
face frame, hence every polyline vertex lies on a mesh edge and the scheme
requires no step-size parameter. Crossing an edge, the direction is transported
by the hinge map (rotation by the dihedral angle about the edge) before the
next face's branch and sign are resolved.

Termination: mesh boundary, non-hyperbolic neighbor (K > -eps_K), length cap,
step cap, or a degenerate configuration.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import (barycentric, build_edge_map, face_frames,
                       rotate_between)


@dataclass
class TraceResult:
    points: np.ndarray        # (k, 3) polyline vertices (on mesh edges)
    faces: np.ndarray         # (k-1,) mesh face traversed by each segment
    family: int
    stop_backward: str
    stop_forward: str

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


class Tracer:
    def __init__(self, field, vertex_eps=1e-4, t_eps=1e-12, graze_eps=0.05,
                s_eps=1e-1, backtrack_margin=0.3):
        self.field = field
        V, F = field.V, field.F
        self.V, self.F = V, F
        self.t1, self.t2, self.n = face_frames(V, F)
        self.O = V[F[:, 0]]
        rel = V[F] - self.O[:, None, :]
        self.V2 = np.stack([np.einsum('ijk,ik->ij', rel, self.t1),
                            np.einsum('ijk,ik->ij', rel, self.t2)], axis=2)
        self.edge_map = build_edge_map(F)
        self.vertex_eps = vertex_eps   # barycentric clamp guarding vertex hits
        self.t_eps = t_eps
        # slack on the edge parameter s (accept s in [-s_eps, 1+s_eps]) so a
        # ray that grazes an edge nearly end-on -- e.g. running almost
        # parallel to the edge just entered, headed for its far vertex --
        # isn't rejected by floating-point overshoot past the vertex; the
        # accepted point is then pulled back to vertex_eps of the vertex below
        self.s_eps = s_eps
        # |d - (d.e_hat)e_hat| below this (d nearly parallel to the edge just
        # crossed, a near-tangential/grazing crossing) means that vector's
        # direction is floating-point noise, not signal -- see _trace_one
        self.graze_eps = graze_eps
        # cosine margin for the anti-backtrack check in _trace_one: the
        # natural pick must be more than acos(-backtrack_margin) from "into
        # the face" before it's overridden by the other candidate, so noise
        # in the edge-crossing reference can't force a wrong pick
        self.backtrack_margin = backtrack_margin

    # ------------------------------------------------------------------ #
    def sanitize_seed(self, f, p, min_bary=1e-3):
        """Project a seed to the face and pull it strictly into the interior."""
        a, b, c = self.V[self.F[f]]
        w = np.clip(barycentric(np.asarray(p, float), a, b, c), min_bary, None)
        w = w / w.sum()
        return w[0] * a + w[1] * b + w[2] * c

    # ------------------------------------------------------------------ #
    def _step(self, f, p, d3, entry_key):
        """One ray/edge intersection inside face f. Returns (q, key, nbr)."""
        t1, t2 = self.t1[f], self.t2[f]
        rel = p - self.O[f]
        p2 = np.array([rel @ t1, rel @ t2])
        d2 = np.array([d3 @ t1, d3 @ t2])
        nrm = np.linalg.norm(d2)
        if nrm < 1e-12:
            return None
        d2 /= nrm
        idx = self.F[f]
        best = None
        for k in range(3):
            i, j = int(idx[k]), int(idx[(k + 1) % 3])
            key = (i, j) if i < j else (j, i)
            if key == entry_key:
                continue
            a2 = self.V2[f, k]
            e2 = self.V2[f, (k + 1) % 3] - a2
            den = d2[0] * e2[1] - d2[1] * e2[0]
            if abs(den) < 1e-14:            # ray parallel to this edge
                continue
            w2 = a2 - p2
            t = (w2[0] * e2[1] - w2[1] * e2[0]) / den
            s = (w2[0] * d2[1] - w2[1] * d2[0]) / den
            if t <= self.t_eps or s < -self.s_eps or s > 1.0 + self.s_eps:
                continue
            if best is None or t < best[0]:
                best = (t, s, i, j, key)
        if best is None:
            return None
        _, s, i, j, key = best
        s = float(np.clip(s, self.vertex_eps, 1.0 - self.vertex_eps))
        q = (1.0 - s) * self.V[i] + s * self.V[j]
        faces = self.edge_map[key]
        nbr = None
        if len(faces) == 2:
            nbr = faces[1] if faces[0] == f else faces[0]
        return q, key, nbr

    def _trace_one(self, f0, p0, d0, max_steps, max_length):
        pts, fcs = [p0], []
        f, p, d, entry = f0, p0, d0, None
        length, reason = 0.0, 'max_steps'
        for _ in range(max_steps):
            cands = self.field.dirs[f]                # (2, 3): both candidate lines
            i_best = int(np.argmax(np.abs(cands @ d)))
            signed = np.where((cands @ d)[:, None] >= 0.0, cands, -cands)
            dfam = signed[i_best]                      # natural pick: closest to d
            if entry is not None:
                ei, ej = entry
                e_hat = self.V[ej] - self.V[ei]
                e_hat /= np.linalg.norm(e_hat)
                ref = d - (d @ e_hat) * e_hat      # keep only the component crossing the edge
                ref_norm = np.linalg.norm(ref)
                if ref_norm >= self.graze_eps:
                    ref_hat = ref / ref_norm
                    if dfam @ ref_hat < -self.backtrack_margin:  # clearly heads back
                        other = signed[1 - i_best]
                        if other @ ref_hat >= -self.backtrack_margin:
                            dfam = other
                # else: d is nearly parallel to the edge (a grazing crossing) --
                # ref's direction is noise here, not signal, so skip the check
            r = self._step(f, p, dfam, entry)
            if r is None:
                reason = 'degenerate'
                break
            q, key, nbr = r
            pts.append(q)
            fcs.append(f)
            length += float(np.linalg.norm(q - p))
            if length >= max_length:
                reason = 'max_length'
                break
            if nbr is None:
                reason = 'boundary'
                break
            if not self.field.hyperbolic[nbr]:
                reason = 'parabolic'
                break
            d = rotate_between(dfam, self.n[f], self.n[nbr])   # hinge map
            f, p, entry = nbr, q, key
        return pts, fcs, reason

    # ------------------------------------------------------------------ #
    def trace(self, f0, p0, family, max_steps=100000, max_length=100.0):
        """Bidirectional trace of the given family through seed (f0, p0)."""
        if not self.field.hyperbolic[f0]:
            raise ValueError("seed face is not hyperbolic")
        p0 = self.sanitize_seed(f0, p0)
        d0 = self.field.dirs[f0, family]
        fw_p, fw_f, fw_r = self._trace_one(f0, p0, d0,
                                           max_steps, 0.5 * max_length)
        bw_p, bw_f, bw_r = self._trace_one(f0, p0, -d0,
                                           max_steps, 0.5 * max_length)
        pts = np.array(bw_p[::-1][:-1] + fw_p)
        fcs = np.array(bw_f[::-1] + fw_f, dtype=int)
        return TraceResult(pts, fcs, family, bw_r, fw_r)
