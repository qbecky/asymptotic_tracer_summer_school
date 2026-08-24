"""Interactive tracing UI (polyscope). Requires a display; not headless.

Usage:
    python exp/03_interactive_ui.py [data/hypar.obj]

Controls
--------
The UI is split into an "Import" section, a "Tracing" section, and an
"Export" section; "Export" itself has two indented subsections, "Geometry"
and "Fabrication".

Import:
* A dropdown lists every data/*.obj file; pick one and press "Load surface"
  to load it. The mesh is uniformly rescaled so its axis-aligned bounding
  box diagonal equals "target AABB diagonal (mm)" (default sqrt(3)*250mm,
  i.e. the diagonal of a 250mm cube -- this project's usual working scale;
  see src/surfaces.SCALE, a 250x250mm domain) -- edit that field before
  loading to work at a different physical size. Loading discards any
  currently traced curves (they belong to the previous mesh's faces) and
  rebuilds the asymptotic field and tracer from scratch. The mesh named on
  the command line (or data/EnneperSurfaceExample.obj by default) is loaded
  at startup with the default target diagonal.

Tracing:
* The surface is shown with the estimated Gaussian curvature K as a scalar
  field and the two asymptotic direction fields as face vector quantities.
* Ctrl-click (polyscope selection) a point on the mesh, then press
  "Trace family A" / "Trace family B" to trace the corresponding asymptotic
  curve through the selected point.
* "Trace net from last curve" distributes seeds at uniform arclength
  ("seed spacing", mm) along the most recent curve (the rail) and traces
  the opposite family through them (the rungs). "Trace net from selected
  curve" does the same but uses whichever traced curve is currently
  ctrl-click-selected as the rail instead -- handy for adding rungs off a
  curve that isn't the last one traced. "Seed regular net" traces a
  rail-and-rungs net from a default interior seed without requiring any
  selection.
* A second, identical trio of buttons below "number of curves" does the
  same three things, except the rail is cut into exactly that many evenly
  spaced rungs (src.curves.seeds_along_curve_n) instead of using "seed
  spacing" -- pick whichever is more convenient: a fixed spacing (rung
  count varies with rail length) or a fixed count (spacing varies with rail
  length).
* Sliders control the seed spacing, the per-curve length cap, and the
  curve radius used for display.
* "Undo last curve" removes only the most recently traced curve (like
  Ctrl-Z); "Redo last curve" brings back the most recently undone one (a
  no-op with a status message if there's nothing to redo -- e.g. nothing's
  been undone yet, or a new curve/delete happened since, which invalidates
  it). "Delete selected curve" removes one specific curve instead --
  ctrl-click *on a traced curve itself* (not the mesh) to select it first,
  then press the button (this does NOT feed the redo stack -- it's a
  deliberate, targeted delete, not an undo). "Clear curves" removes all of
  them.

Export (Geometry subsection):
* "Export curves (.obj)" writes the current set to output/ui_curves.obj (a
  flat dump, overwritten each time -- for quick inspection, not reloadable
  by any other script here).
* "Export trial (net.json)" writes a new, numbered output/<trial name>_<NNN>/
  geom/net.json (src.rundir.new_trial) in the schema src.meshio.save_curves_json
  writes: surface normals are sampled along each curve, then both are
  resampled to H spacing (src.curves.sample_normals/resample) -- raw traced
  points are far denser than that, which would otherwise confuse crossing
  detection on whatever consumes net.json later. "trial name" defaults to
  "default" (so repeated exports become output/default_000/,
  output/default_001/, ...). The written JSON's "surface" field is the stem
  of the mesh file this UI was launched with (e.g. "hypar" for the default
  data/hypar.obj).

Export (Fabrication subsection):
* "Export fabrication SVG" resamples the currently traced family-A/family-B
  curves to a consistent H spacing (raw traced curves are much denser than
  that, which would otherwise fool find_crossings's duplicate-suppression
  into reporting several near-duplicate "crossings" per real one), detects
  crossings between them (src.crossings.find_crossings), and drops any that
  are near-parallel-tangent degenerate or too close to a curve's own
  endpoint (no room for a notch there), so "joint" here means a clean,
  fabricatable half-lap intersection. For each rod with at least one
  surviving joint, unrolls it into a flat strip (per the cross-section
  width/height fields, mm) with a comb/finger-joint notch
  (src.fabrication.default_flap_slits) at every joint -- family-A rods
  notched from one edge, family-B from the other, so crossing strips
  interlock flush -- packed left-aligned and stacked into one SVG at a new,
  numbered output/<trial name>_<NNN>/geom/fabrication.svg (src.rundir.new_trial,
  same "trial name" field as "Export trial (net.json)"). Alongside it, writes
  geom/fabrication_guide.svg (src.fabrication.pack_svg_guide): the same
  outlines, but with the identity of the OTHER family's beam engraved inside
  each slit (e.g. beam A0's guide shows "B3" sitting in the slit where rod B3
  crosses it) -- a reference for assembly, not meant to be cut from (its
  guide text uses a third color, distinct from the cut and label colors).
  See src/fabrication.py.

Curves are tracked in `state["curves"]`, a dict keyed by a stable
ever-incrementing id (not list position), so deleting one from the middle
never renames/renumbers the others' polyscope structures (`curve_<id>`).
`state["redo_stack"]` holds (id, TraceResult) pairs removed by "Undo last
curve", most-recent last, so "Redo last curve" can pop and re-register the
exact same id (`_register_curve`, shared with `_add_curve`) rather than
appending a fresh one.

The picking helper (`_current_selection`) supports both the tuple-based
selection API of older polyscope releases and the PickResult object of
polyscope >= 2.x. If your polyscope version exposes yet another interface,
adapt it there -- `_picked_vertex` (mesh) and `_picked_curve` (traced
curves) both build on it.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polyscope as ps            # noqa: E402
import polyscope.imgui as psim    # noqa: E402

from src import curves as cv      # noqa: E402
from src import field as fld_mod  # noqa: E402
from src import meshio, rundir    # noqa: E402
from src.crossings import find_crossings  # noqa: E402
from src.fabrication import (beam_outline, default_flap_slits,  # noqa: E402
                             joint_label_position, label_position,
                             layout_from_joints, pack_svg, pack_svg_guide)
from src.geometry import nearest_face_xy, vertex_normals  # noqa: E402
from src.surfaces import SCALE    # noqa: E402
from src.tracer import Tracer     # noqa: E402

MESH_NAME = "surface"
FAMILY_COLORS = ((0.12, 0.47, 0.71), (0.84, 0.15, 0.16))
MIN_CURVE_POINTS = 5             # minimum sample count for a curve to be exportable
H = 0.04 * SCALE                     # re-discretization spacing (mm), used before export/crossing detection
# Crossing-detection tolerance (mm): a closest-approach distance below this
# counts as a joint. A genuine crossing's gap is bounded by resampling
# discretization error (a couple hundredths of a mm, worst case, across this
# project's sample meshes); a spurious near-miss (two curves merely passing
# close without truly meeting) sits at least an order of magnitude above
# that. The old 0.09*SCALE (11.25mm) was loose enough to accept such
# near-misses as extra "crossings" alongside a real one on the same curve
# pair -- visible as one beam's partner label engraved twice in the assembly
# guide for a single true joint. Tying TOL to H, well under it, fixes that
# without capping how many times a pair is allowed to genuinely cross.
TOL = 0.1 * H                          # crossing-detection tolerance (mm)
CURVE_RADIUS = 0.00005 * SCALE  # display-only tube radius (mm)
DEFAULT_TARGET_DIAGONAL = 300.0  # mm

DATA_DIR = ROOT / "data"

state = dict(curves={}, next_id=0, redo_stack=[], last=None, spacing=0.18 * SCALE,
             n_curves=5, max_length=8.0 * SCALE, radius=CURVE_RADIUS, width=0.5, height=10.0,
             export_name="default", mesh_name=None, V=None, F=None, VN=None, fld=None,
             tr=None, data_files=[], data_idx=0, target_diagonal=DEFAULT_TARGET_DIAGONAL,
             msg="ctrl-click the surface, then trace")


def _current_selection():
    """Return (structure_name, local_index) from the current polyscope
    selection, or (None, None) if nothing is selected -- handles both the
    tuple-based selection API of older polyscope releases and the
    PickResult object of polyscope >= 2.x. If your polyscope version
    exposes yet another interface, adapt this."""
    try:
        sel = ps.get_selection()
    except Exception:
        return None, None
    if sel is None:
        return None, None
    if isinstance(sel, tuple):                       # legacy API
        return sel
    if not getattr(sel, "is_hit", False):             # PickResult (>= 2.x)
        return None, None
    return getattr(sel, "structure_name", None), getattr(sel, "local_index", None)


def _picked_vertex(n_vertices, n_faces, F):
    """Return a seed face index from the current mesh selection, or None."""
    name, idx = _current_selection()
    if name != MESH_NAME or idx is None:
        return None
    idx = int(idx)
    if idx < n_vertices:                             # vertex pick
        inc = np.where((F == idx).any(axis=1))[0]
        return int(inc[0]) if len(inc) else None
    if idx < n_vertices + n_faces:                   # face pick (offset)
        return idx - n_vertices
    return None


def _picked_curve():
    """Return the id of the currently selected curve network (ctrl-click a
    traced curve, not the mesh), or None if nothing/something else is
    selected, or the selected curve was already removed."""
    name, _idx = _current_selection()
    if name is None or not name.startswith("curve_"):
        return None
    try:
        cid = int(name[len("curve_"):])
    except ValueError:
        return None
    return cid if cid in state["curves"] else None


def _register_curve(cid, res):
    """Register (or re-register, for redo) a TraceResult as polyscope
    structure `curve_<cid>` and record it in state["curves"] -- shared by
    _add_curve (fresh id) and _redo_last (an id popped off the redo stack)."""
    P = res.points
    E = np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))])
    net = ps.register_curve_network(f"curve_{cid}", P, E,
                                    radius=state["radius"])
    net.set_color(FAMILY_COLORS[res.family])
    state["curves"][cid] = res
    state["last"] = res


def _add_curve(res):
    cid = state["next_id"]
    state["next_id"] += 1
    _register_curve(cid, res)
    state["redo_stack"].clear()  # a genuinely new curve invalidates any pending redo


def _remove_curve(cid):
    ps.remove_curve_network(f"curve_{cid}", error_if_absent=False)
    del state["curves"][cid]
    state["last"] = next(reversed(state["curves"].values()), None)
    state["redo_stack"].clear()


def _undo_last():
    """Remove the most recently added curve (like Ctrl-Z), pushing it onto
    the redo stack so "Redo last curve" can bring it straight back."""
    if not state["curves"]:
        state["msg"] = "no curves to undo"
        return
    cid = next(reversed(state["curves"]))
    res = state["curves"][cid]
    _remove_curve(cid)
    state["redo_stack"].append((cid, res))
    state["msg"] = f"undid curve {cid} ({len(state['curves'])} left)"


def _redo_last():
    """Restore the most recently undone curve, if there is one."""
    if not state["redo_stack"]:
        state["msg"] = "nothing to redo"
        return
    cid, res = state["redo_stack"].pop()
    _register_curve(cid, res)
    state["msg"] = f"redid curve {cid}"


def _trace_from(tr, fld, f0, family):
    p0 = fld.V[fld.F[f0]].mean(axis=0)
    _add_curve(tr.trace(f0, p0, family, max_length=state["max_length"]))


def _trace_net_with_seeds(tr, fld, rail, seeds):
    """Trace the opposite family through every hyperbolic seed -- the step
    shared by both seeding strategies (arclength spacing vs. curve count)
    and both rail choices (last curve vs. ctrl-click-selected curve)."""
    for fs, pseed in seeds:
        if fld.hyperbolic[fs]:
            _add_curve(tr.trace(fs, pseed, 1 - rail.family,
                                max_length=state["max_length"]))


def _rail_last():
    rail = state["last"]
    if rail is None:
        state["msg"] = "no rail curve yet"
    return rail


def _rail_selected():
    """The ctrl-click-selected traced curve, or None (with a status
    message) if nothing/something else is selected."""
    cid = _picked_curve()
    if cid is None:
        state["msg"] = "ctrl-click a traced curve first"
        return None
    return state["curves"][cid]


def _trace_net_from_last(tr, fld):
    rail = _rail_last()
    if rail is not None:
        _trace_net_with_seeds(tr, fld, rail, cv.seeds_along_curve(rail, state["spacing"]))


def _trace_net_from_last_n(tr, fld):
    rail = _rail_last()
    if rail is not None:
        _trace_net_with_seeds(tr, fld, rail, cv.seeds_along_curve_n(rail, state["n_curves"]))


def _trace_net_from_selected(tr, fld):
    rail = _rail_selected()
    if rail is not None:
        _trace_net_with_seeds(tr, fld, rail, cv.seeds_along_curve(rail, state["spacing"]))


def _trace_net_from_selected_n(tr, fld):
    rail = _rail_selected()
    if rail is not None:
        _trace_net_with_seeds(tr, fld, rail, cv.seeds_along_curve_n(rail, state["n_curves"]))


def _export_fabrication():
    fam = {0: [], 1: []}
    for r in state["curves"].values():
        if len(r.points) < MIN_CURVE_POINTS:
            continue
        fam[r.family].append(r.points)

    # Resample to a consistent arclength spacing before crossing detection.
    # Raw traced curves are far denser than H, and find_crossings's
    # duplicate-suppression (min_edge_gap, counted in EDGES, not mm) only
    # rejects near-duplicate detections correctly at that consistent
    # spacing -- skipping this found 36 "crossings" on a real test net
    # where only 12 were real. Normals aren't needed for crossing detection
    # (find_crossings never reads them), so pass zeros.
    fam0 = [cv.resample(P, np.zeros_like(P), H)[0] for P in fam[0]]
    fam1 = [cv.resample(P, np.zeros_like(P), H)[0] for P in fam[1]]

    crossings = find_crossings([(P, None) for P in fam0], [(P, None) for P in fam1], TOL)

    # Drop crossings between near-parallel tangents (not a clean transversal
    # intersection -- no well-defined joint rotation axis) and crossings too
    # close to a curve's own endpoint (no room for the notch/overhang
    # there), so "joint" here means a clean, fabricatable half-lap
    # intersection, matching the margins default_flap_slits assumes below.
    eps = 0.15 * H
    len0 = [cv.arclength(P)[-1] for P in fam0]
    len1 = [cv.arclength(P)[-1] for P in fam1]

    def _valid(c):
        if np.linalg.norm(np.cross(c.tangent_a, c.tangent_b)) < 1e-8:
            return False
        if not (eps < c.s_a < len0[c.a_idx] - eps):
            return False
        return eps < c.s_b < len1[c.b_idx] - eps

    crossings = [c for c in crossings if _valid(c)]

    # Each entry also carries the OTHER family's beam label crossing there
    # (e.g. "B3"), purely for the assembly guide -- layout_from_joints below
    # only reads the (s, side) part of each tuple.
    per_rod = {}
    for c in crossings:
        per_rod.setdefault(('A', c.a_idx), []).append((c.s_a, 0, f"B{c.b_idx}"))
        per_rod.setdefault(('B', c.b_idx), []).append((c.s_b, 1, f"A{c.a_idx}"))

    overhang = state["height"] / 2.0
    slit0, slit1, notch_len, depth = default_flap_slits(state["height"], state["width"])
    beams = []
    guide_beams = []
    for fam_label, curves in (('A', fam0), ('B', fam1)):
        for idx in range(len(curves)):
            key = (fam_label, idx)
            if key not in per_rod:
                continue
            joints = [(s, side) for s, side, _partner in per_rod[key]]
            strip_length, local = layout_from_joints(joints, overhang)
            shapes = beam_outline(local, strip_length, state["height"], notch_len,
                                  depth=depth, slit=slit0, slit1=slit1)
            pos = label_position(local, state["height"])
            label = f"{fam_label}{idx}"
            beams.append((shapes, label, pos))
            joint_labels = [
                (*joint_label_position(x, side, state["height"], depth), partner)
                for (x, side), (_s, _side, partner) in zip(local, per_rod[key])
            ]
            guide_beams.append((shapes, label, pos, joint_labels))

    if not beams:
        state["msg"] = "no crossings found -- trace a net first"
        return

    run = rundir.new_trial(state["export_name"])
    out = run / "geom" / "fabrication.svg"
    out.write_text(pack_svg(beams))
    guide_out = run / "geom" / "fabrication_guide.svg"
    guide_out.write_text(pack_svg_guide(guide_beams))
    state["msg"] = f"fabrication: {len(beams)} beams -> {out.relative_to(ROOT)} (+ guide)"


def _export_trial():
    """Write the current curves as a new output/<export_name>_<XXX>/geom/
    net.json trial (src.rundir.new_trial), in the simple JSON schema
    src.meshio.save_curves_json writes.

    Curves are resampled to H spacing before export -- raw traced points
    are far denser than that, and a crossing-detection pass over whatever's
    in net.json can't tell a real crossing from several near-duplicate
    detections of it at that density (duplicate suppression is counted in
    edges, not mm) -- not overlapping curves, just points too dense for
    crossing detection to resolve correctly.

    Also writes geom/surface.obj: state["V"]/state["F"] AS RESCALED by
    _load_surface's target-AABB-diagonal step, not the raw data/<surface>.obj
    on disk (those only coincide when target_diagonal happens to match the
    file's own raw diagonal) -- so any downstream consumer of this trial
    sees the surface at the same scale as the curves actually exported."""
    if not state["curves"]:
        state["msg"] = "no curves to export"
        return
    run = rundir.new_trial(state["export_name"])
    payload = {"surface": state["mesh_name"], "curves": []}
    for r in state["curves"].values():
        N = cv.sample_normals(r.points, r.faces, state["V"], state["F"], state["VN"])
        P2, N2, _s = cv.resample(r.points, N, H)
        payload["curves"].append({
            "family": int(r.family),
            "points": P2,
            "normals": N2,
        })
    meshio.save_curves_json(run / "geom" / "net.json", payload)
    meshio.save_obj(run / "geom" / "surface.obj", state["V"], state["F"])
    state["msg"] = f"exported {len(state['curves'])} curves -> {run.relative_to(ROOT)}"


def _load_surface(name):
    """(Re)load data/<name>, uniformly rescaled so its axis-aligned bounding
    box diagonal equals state["target_diagonal"] (mm), rebuild the
    asymptotic field and tracer from scratch, and re-register the polyscope
    surface -- discarding any curves already traced, since they belong to
    the previous mesh's faces/geometry and wouldn't make sense on this one.
    Used both at startup and by the "Load surface" button."""
    path = DATA_DIR / name
    V, F = meshio.load_obj(path)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    V = V * (state["target_diagonal"] / diag)
    fld = fld_mod.from_estimated_curvature(V, F)
    tr = Tracer(fld)

    for cid in list(state["curves"]):
        ps.remove_curve_network(f"curve_{cid}", error_if_absent=False)
    state["curves"].clear()
    state["redo_stack"].clear()
    state["last"] = None
    state.update(V=V, F=F, VN=vertex_normals(V, F), mesh_name=Path(path).stem,
                fld=fld, tr=tr)

    ps.remove_surface_mesh(MESH_NAME, error_if_absent=False)
    mesh = ps.register_surface_mesh(MESH_NAME, V, F, smooth_shade=True)
    mesh.add_scalar_quantity("K (estimated)", fld.K, defined_on="faces",
                             enabled=False, cmap="viridis")
    mesh.add_vector_quantity("asymptotic dir A", fld.dirs[:, 0],
                             defined_on="faces", color=FAMILY_COLORS[0])
    mesh.add_vector_quantity("asymptotic dir B", fld.dirs[:, 1],
                             defined_on="faces", color=FAMILY_COLORS[1])
    state["msg"] = f"loaded {name} (AABB diagonal -> {state['target_diagonal']:.1f}mm)"


def main():
    state["data_files"] = sorted(p.name for p in DATA_DIR.glob("*.obj"))
    default_name = "EnneperSurfaceExample.obj"
    if len(sys.argv) > 1:
        default_name = Path(sys.argv[1]).name
    state["data_idx"] = (state["data_files"].index(default_name)
                         if default_name in state["data_files"] else 0)

    ps.init()
    ps.set_up_dir("z_up")
    _load_surface(state["data_files"][state["data_idx"]])

    def callback():
        psim.TextUnformatted("Import")
        psim.Separator()
        _, state["data_idx"] = psim.Combo(
            "surface (data/*.obj)", state["data_idx"], state["data_files"])
        _, state["target_diagonal"] = psim.InputFloat(
            "target AABB diagonal (mm)", state["target_diagonal"])
        if psim.Button("Load surface"):
            _load_surface(state["data_files"][state["data_idx"]])

        psim.Spacing()
        psim.Separator()
        psim.TextUnformatted("Tracing")
        psim.Separator()
        _, state["max_length"] = psim.SliderFloat(
            "max curve length", state["max_length"], 1.0 * SCALE, 20.0 * SCALE)
        _, state["radius"] = psim.SliderFloat(
            "curve radius", state["radius"], 0.001 * SCALE, 0.02 * SCALE)
        V, F, fld, tr = state["V"], state["F"], state["fld"], state["tr"]
        for fam, label in ((0, "Trace family A"), (1, "Trace family B")):
            if psim.Button(label):
                f0 = _picked_vertex(len(V), len(F), F)
                if f0 is not None and fld.hyperbolic[f0]:
                    _trace_from(tr, fld, f0, fam)
                    state["msg"] = f"traced family {'AB'[fam]} curve"
                else:
                    state["msg"] = "no valid selection (ctrl-click the mesh)"
            psim.SameLine()
        psim.NewLine()
        psim.Spacing()
        _, state["spacing"] = psim.SliderFloat(
            "seed spacing", state["spacing"], 0.05 * SCALE, 0.5 * SCALE)
        if psim.Button("Trace net from last curve"):
            _trace_net_from_last(tr, fld)
        psim.SameLine()
        if psim.Button("Trace net from selected curve"):
            _trace_net_from_selected(tr, fld)
        if psim.Button("Seed regular net"):
            f0 = nearest_face_xy(V, F, (0.1 * SCALE, 0.13 * SCALE))
            _trace_from(tr, fld, f0, 0)
            _trace_net_from_last(tr, fld)
            _trace_from(tr, fld, f0, 1)

        psim.Spacing()
        _, state["n_curves"] = psim.InputInt("number of curves", state["n_curves"])
        if psim.Button("Trace net from last curve##n"):
            _trace_net_from_last_n(tr, fld)
        psim.SameLine()
        if psim.Button("Trace net from selected curve##n"):
            _trace_net_from_selected_n(tr, fld)
        if psim.Button("Seed regular net##n"):
            f0 = nearest_face_xy(V, F, (0.1 * SCALE, 0.13 * SCALE))
            _trace_from(tr, fld, f0, 0)
            _trace_net_from_last_n(tr, fld)
            _trace_from(tr, fld, f0, 1)

        psim.Spacing()
        if psim.Button("Undo last curve"):
            _undo_last()
        psim.SameLine()
        if psim.Button("Redo last curve"):
            _redo_last()

        if psim.Button("Delete selected curve"):
            cid = _picked_curve()
            if cid is not None:
                _remove_curve(cid)
                state["msg"] = f"deleted curve {cid}"
            else:
                state["msg"] = "ctrl-click a traced curve first"
        psim.SameLine()
        if psim.Button("Clear curves"):
            for cid in list(state["curves"]):
                ps.remove_curve_network(f"curve_{cid}", error_if_absent=False)
            state["curves"].clear()
            state["redo_stack"].clear()
            state["last"] = None

        psim.Spacing()
        psim.Separator()
        psim.TextUnformatted("Export")
        psim.Separator()
        psim.TextUnformatted("Trial name (for output/<export_name>_<XXX>/geom/*)")
        _, state["export_name"] = psim.InputText("", state["export_name"])

        psim.Indent()

        psim.TextUnformatted("Geometry")
        psim.Separator()
        if psim.Button("Export curves (.obj)"):
            out = ROOT / "output"
            out.mkdir(exist_ok=True)
            meshio.save_polylines_obj(out / "ui_curves.obj",
                                      [r.points for r in state["curves"].values()])
            state["msg"] = f"wrote {len(state['curves'])} curves"
        if psim.Button("Export trial (net.json)"):
            _export_trial()

        psim.Spacing()
        psim.TextUnformatted("Fabrication")
        psim.Separator()
        _, state["width"] = psim.InputFloat("cross-section width (mm)", state["width"])
        _, state["height"] = psim.InputFloat("cross-section height (mm)", state["height"])
        if psim.Button("Export fabrication SVG"):
            _export_fabrication()

        psim.Unindent()

        psim.Spacing()
        psim.Separator()
        psim.TextUnformatted(state["msg"])

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
