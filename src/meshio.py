"""Minimal I/O: OBJ meshes, OBJ polylines ('l' records), JSON curve data."""
import json

import numpy as np


def save_obj(path, V, F):
    """Write vertices V and faces F (any polygon size, e.g. triangles or quads)."""
    with open(path, 'w') as fh:
        for v in V:
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for f in F:
            idx = " ".join(str(int(i) + 1) for i in f)
            fh.write(f"f {idx}\n")


def load_obj(path):
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                V.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                F.append([int(p.split('/')[0]) - 1 for p in parts[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def save_polylines_obj(path, curves):
    """Write a list of (k, 3) polylines as OBJ line elements."""
    with open(path, 'w') as fh:
        offset = 1
        for P in curves:
            for p in P:
                fh.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            idx = " ".join(str(offset + i) for i in range(len(P)))
            fh.write(f"l {idx}\n")
            offset += len(P)


def save_curves_json(path, payload):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, 'w') as fh:
        json.dump(payload, fh, default=default, indent=1)


def save_curve_network(path, curves):
    """Write TraceResult-like objects (attrs family, points, faces,
    stop_backward, stop_forward) to `path` as JSON at full fidelity: raw,
    unresampled points and their per-segment mesh face indices. This is
    what makes load_curve_network's round trip exact -- unlike
    save_curves_json's trial export (H-resampled, no face indices), a
    reloaded curve still has the .faces a rail needs to seed further nets,
    or to be re-exported."""
    payload = {"curves": [
        {"family": r.family, "points": r.points, "faces": r.faces,
         "stop_backward": r.stop_backward, "stop_forward": r.stop_forward}
        for r in curves
    ]}
    save_curves_json(path, payload)


def load_curve_network(path):
    """Inverse of save_curve_network: returns the raw list of curve dicts
    (plain JSON types). The caller reconstructs TraceResult objects --
    meshio doesn't import src.tracer, to keep this module dependency-free."""
    with open(path) as fh:
        return json.load(fh)["curves"]
