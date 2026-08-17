# Asymptotic Curve Tracer

A small, dependency-light Python tool for tracing discrete **asymptotic curve
networks** on negatively curved (K < 0) triangle meshes, targeted at the
design of asymptotic gridshells. Given a mesh, it estimates the curvature
tensor, extracts the two asymptotic line fields, separates them into globally
consistent families, and traces curves and rail-and-rungs nets by exact
face-walking. Traced curves are exported with surface normals and discrete
Darboux quantities (geodesic curvature, normal curvature, geodesic torsion).

## Method

**Curvature estimation.** The estimator fits an osculating jet (quadratic
height field) over the 2-ring of every vertex in a frame aligned with the
angle-weighted normal, in the spirit of Cazals and Pouget. The Weingarten map
of the fitted graph is diagonalized through the generalized symmetric
eigenproblem II x = kappa I x, and stored as an ambient curvature tensor
kappa_1 e1 e1^T + kappa_2 e2 e2^T. Corner tensors are averaged per face and
restricted to the face plane, yielding a symmetric 2x2 tensor per face.

**Asymptotic fields.** At a hyperbolic face with principal curvatures
kappa_1 > 0 > kappa_2 and principal directions e1, e2, the asymptotic
directions are d = cos(phi) e1 +/- sin(phi) e2 with
tan^2(phi) = -kappa_1 / kappa_2. Since a connected region with K < 0 contains
no umbilics, the two families are globally separable; a breadth-first search
over face adjacency propagates a consistent family labeling by matching
direction pairs across edges as unoriented lines.

**Tracing.** The tracer walks the piecewise-constant field face by face. In
each face the family direction is re-evaluated; the incoming direction only
resolves the sign of the line field and is never integrated, so no extrinsic
drift accumulates. Exit points are computed by exact ray/edge intersection in
the 2D face frame (every polyline vertex lies on a mesh edge; no step-size
parameter), and directions are transported across edges by the hinge map
(rotation by the dihedral angle). Traces terminate at the boundary, at
non-hyperbolic faces (K > -eps_K), or at length/step caps, and run
bidirectionally from the seed.

**Curve quantities.** Along each curve, surface normals are interpolated
barycentrically from vertex normals and the Darboux quantities kappa_g,
kappa_n, tau_g are evaluated by central differences after uniform arclength
resampling. For asymptotic curves, kappa_n = 0 and (Beltrami--Enneper)
tau_g^2 = -K; both identities serve as validation criteria.

## Repository structure

```
data/     input meshes (.obj) -- analytic test surfaces and generated examples
src/      library code
  geometry.py     normals, frames, adjacency, barycentrics, hinge map
  surfaces.py     analytic test surfaces z = f(x, y)
  curvature.py    osculating-jet shape-operator estimator
  field.py        asymptotic fields and global family labeling
  tracer.py       face-walking tracer
  curves.py       resampling, Darboux quantities, rail seeding
  crossings.py    curve/curve crossing detection
  fabrication.py  flat-pattern (laser-cut) SVG export
  meshio.py       OBJ / JSON I/O
  rundir.py       per-trial output/ layout
exp/
  03_interactive_ui.py   polyscope interactive UI (requires a display)
output/   created on first export; gitignored
```

## Setup

Requires a display for the interactive UI (not runnable headless).

```
conda env create -f environment.yml
conda activate asymptotic_tracer_dtu
```

This installs Python 3.10, numpy, and polyscope -- the only third-party
dependencies (a minimal `requirements.txt` is also provided for plain
pip/venv setups).

## Usage

```
python exp/03_interactive_ui.py [data/<mesh>.obj]
```

Opens a polyscope viewer showing the estimated Gaussian curvature K and both
asymptotic direction fields on the chosen mesh (default:
`data/EnneperSurfaceExample.obj`). Ctrl-click a point on the mesh, then press
"Trace family A" / "Trace family B" to trace a single asymptotic curve, or
use the "Trace net" buttons to seed a rail-and-rungs net. "Export" writes the
current curves to `output/` as a flat OBJ dump, a JSON trial
(`output/<name>_<NNN>/geom/net.json`), or a laser-cut-ready fabrication SVG.
See the module docstring at the top of `exp/03_interactive_ui.py` for the
full control reference.

## References

* F. Cazals, M. Pouget. Estimating differential quantities using polynomial
  fitting of osculating jets. CAGD 2005.
