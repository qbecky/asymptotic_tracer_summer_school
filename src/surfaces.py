"""Analytic graph surfaces z = f(x, y) used for mesh generation and validation.

All lengths in this repo are in mm (consistent mm-g-ms-MPa unit system: force
comes out in N, pressure in MPa, energy in N*mm). Both test surfaces are
negatively curved on the whole domain and are scaled to fit a 250x250mm box
(domain [-SCALE, SCALE]^2):

* ``HYPAR``:       z = x*y. Doubly ruled; the asymptotic curves are exactly the
                   straight rulings x = const and y = const, which gives a
                   strong ground truth for the tracer (straightness test).
* ``TRANS_SADDLE``: z = log cosh x - log cosh y. A translational surface with
                   K = -sech^2(x) sech^2(y) / (1 + tanh^2 x + tanh^2 y)^2 < 0
                   everywhere and genuinely *curved* asymptotic lines, whose
                   parameter-space ODE dy/dx = +/- cosh(y)/cosh(x) can be
                   integrated to high accuracy for an independent cross-check.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

SCALE = 125.0  # mm; domain is [-SCALE, SCALE]^2, a 250x250mm box


@dataclass
class GraphSurface:
    name: str
    f: Callable
    grad: Callable   # returns (f_x, f_y)
    hess: Callable   # returns (f_xx, f_xy, f_yy)

    def gaussian_curvature(self, x, y):
        fx, fy = self.grad(x, y)
        fxx, fxy, fyy = self.hess(x, y)
        return (fxx * fyy - fxy ** 2) / (1.0 + fx ** 2 + fy ** 2) ** 2


def _scale(surface, s):
    """Uniformly scale a unit-domain GraphSurface by s (preserves shape and
    curvature ratios exactly; K scales as 1/s^2)."""
    f, grad, hess = surface.f, surface.grad, surface.hess
    return GraphSurface(
        name=surface.name,
        f=lambda x, y: s * f(x / s, y / s),
        grad=lambda x, y: grad(x / s, y / s),
        hess=lambda x, y: tuple(v / s for v in hess(x / s, y / s)),
    )


_HYPAR_UNIT = GraphSurface(
    name="hypar",
    f=lambda x, y: x * y,
    grad=lambda x, y: (y, x),
    hess=lambda x, y: (np.zeros_like(x), np.ones_like(x), np.zeros_like(x)),
)

_TRANS_SADDLE_UNIT = GraphSurface(
    name="trans_saddle",
    f=lambda x, y: np.log(np.cosh(x)) - np.log(np.cosh(y)),
    grad=lambda x, y: (np.tanh(x), -np.tanh(y)),
    hess=lambda x, y: (1.0 / np.cosh(x) ** 2,
                       np.zeros_like(x),
                       -1.0 / np.cosh(y) ** 2),
)

HYPAR = _scale(_HYPAR_UNIT, SCALE)
TRANS_SADDLE = _scale(_TRANS_SADDLE_UNIT, SCALE)

SURFACES = {s.name: s for s in (HYPAR, TRANS_SADDLE)}
