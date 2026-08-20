"""Flat-pattern (laser-cut) fabrication export.

Each beam is unrolled into a straight strip: length = arc length between its
outermost joints, plus a fixed overhang at each end, width = the beam's tall
(along-normal) cross-section dimension laid flat in the cutting plane -- the
thin (in-plane) dimension is the physical sheet material's own thickness,
never drawn. A notch is cut half-way into the strip's width at every joint
along it, so that two crossing strips interlock flush (each contributing
half its height) when assembled -- a half-lap joint. Which edge a joint's
notch comes from is a caller-supplied 0/1 "side": callers set side = a
joint's own family (family 0 -> side 0 -> notched from the y=0 edge; family
1 -> side 1 -> notched from the y=width edge), the same side for every
joint on a given beam, since a beam's family never changes along its length
-- so family 0 and family 1 are already fully distinguished geometrically
(opposite edges), before any color or label is involved.

A beam is represented as a LIST of separate, independent polylines, not one
combined outline: the plain outer rectangle, plus every polyline making up
each joint's slit. A slit's polylines typically overlap the outer rectangle
along the edge segment they're cut from (by design; a laser re-cutting that
shared segment is harmless/redundant, and it's what lets each one stay
independent of the others and of beam assembly order, rather than being a
detour merged into one combined path). Polylines are drawn exactly as
given, each as its own SVG <polyline> -- not <polygon>, which auto-closes
and so can't represent multiple independent, possibly-open shapes as one
element anyway; a polyline that needs to look closed (like the outer
rectangle, or rectangular_slit's notch) repeats its own first point as its
last, but this isn't required -- some slit designs are intentionally open
(e.g. flaps_slit_fam0/_fam1's individual flaps).

Slit shapes are decoupled from beam assembly: a "slit shape" is a small
function of (notch_len, depth) returning a LIST of one or more independent
polylines for ONE joint (rectangular_slit returns a single one;
flaps_slit_fam0/_fam1 return several separate, unconnected ones -- one per
flap -- since those are NOT meant to be merged into a single path), in a
canonical frame (as if cut into the y=0 edge, x centered on the joint, y=0
on the edge increasing into the material). `beam_outline` places that same
list of polylines at every joint along a beam, mirroring/reversing each one
automatically for edges cut from the far side of the strip -- so a new slit
design (trapezoidal, keyed, rounded, ... e.g. rectangular_slit,
flaps_slit_fam0) only requires writing one new function with that
signature and passing it as `slit=`; nothing about beam assembly changes.

Some shapes aren't mirror-images of themselves and must be authored
separately per edge instead (e.g. flaps_slit_fam1, the side-1 counterpart to
flaps_slit_fam0 -- it computes its own points directly against the y=width
edge rather than being derived by flipping a side-0 shape). For those, pass
the side-1 shape as `beam_outline`'s `slit1=`, which places it verbatim (no
mirroring, no reversal) instead of deriving it from `slit`.

Two-step API: `layout_from_joints` turns a beam's raw joint arclength
positions into a strip length + local notch positions; `beam_outline` turns
that into the beam's list of closed shapes. `label_position` picks a
slit-free spot for a beam's identifying label. `pack_svg` lays a list of
(shapes, label) pairs out left-aligned, stacked top to bottom, into one
laser-ready SVG document (units = mm), each labeled in a different color so
the label reads as "engrave, don't cut" rather than a third cut path.

`joint_label_position` + `pack_svg_guide` build an assembly-guide variant of
that same packed sheet: alongside each beam's own identifying label,
`pack_svg_guide` also engraves, right inside each slit, the identity of the
OTHER family's beam that notches through it there (e.g. beam A0's sheet
shows "B3" sitting in the slit where rod B3 crosses it) -- so the guide
answers "which beam goes in this slit?" by eye during assembly, without
consulting the joint list separately. It is not meant to be cut from (hence
a third, distinct color from CUT_COLOR/LABEL_COLOR) -- print or view it
alongside the plain `pack_svg` output, which stays the actual cut file.

Note: labels are emitted as raw SVG <text>, not vector outlines -- some
laser-cutting software requires text converted to paths before it will
engrave it; that conversion is out of scope here.
"""
import functools

CUT_COLOR = "#000000"
LABEL_COLOR = "#0000ff"
GUIDE_COLOR = "#ff0000"


def layout_from_joints(joints, overhang):
    """joints: [(s, side), ...] -- s = arclength position of a joint along
    the beam's own curve (mm), side = 0/1 selecting which edge that joint's
    notch is cut from. `side` should be the SAME value for every joint here
    -- it's meant to be the beam's own family (0 or 1), not a per-joint
    choice, so a beam's notches are always cut from one edge only, never a
    mix of both. Returns (strip_length, local_joints), with x already
    shifted so the first (smallest-s) joint's overhang starts at local x=0."""
    if not joints:
        raise ValueError("a beam needs at least one joint to fabricate")
    s_min = min(s for s, _side in joints)
    s_max = max(s for s, _side in joints)
    strip_length = 2.0 * overhang + (s_max - s_min)
    local = [(overhang + (s - s_min), side) for s, side in joints]
    return strip_length, local


def rectangular_slit(notch_len, depth):
    """The default slit shape: a simple rectangular notch, `notch_len` wide
    and `depth` deep. Canonical frame: centered on dx=0, dy=0 on the edge,
    increasing into the material -- see the module docstring."""
    h = notch_len / 2.0
    return [[(-h, 0.0), (-h, depth), (h, depth), (h, 0.0), (-h, 0.0)]]

def flaps_slit_fam0(notch_len, depth, flap_len, n_flaps, ribbon_width):
    """A comb-shaped slit for family-0 (side-0, y=0-edge) beams: `n_flaps`
    alternating teeth cut into the material, each tooth stepping in by
    `flap_len` from the notch's outer edges (h = notch_len/2) before
    returning to the full notch_len width -- a friction/glue-surface
    variant of rectangular_slit, not a plain rectangle. `depth` (the total
    reach into the material, as in rectangular_slit) is split evenly across
    `2*n_flaps + 1` bands of height `flap_depth = depth / (2*n_flaps + 1)`.
    Canonical frame: centered on dx=0, dy=0 on the edge, increasing into the
    material -- see the module docstring. `ribbon_width` is only used to
    validate `depth <= ribbon_width`, not in the returned points -- this is
    still a side-0-only shape (its points are all relative to the y=0 edge
    it's cut from) -- pass it as `beam_outline`'s `slit=`.

    Returns `n_flaps + 1` SEPARATE, unconnected polylines (one per tooth,
    plus a closing one) -- not merged into a single path, since the teeth
    aren't meant to be connected to each other."""
    if n_flaps < 1:
        raise ValueError("n_flaps must be >= 1")
    if depth > ribbon_width:
        raise ValueError("depth must be <= ribbon_width")

    flap_depth = depth / (2.0 * n_flaps + 1.0)
    h = notch_len / 2.0
    pts = []
    for id_flap in range(n_flaps):
        y0 = id_flap * 2.0 * flap_depth
        pts.append([
            (-h, y0 + flap_depth),
            (-h, y0),
            (h, y0),
            (h, y0 + flap_depth),
            (h - flap_len, y0 + flap_depth),
            (h - flap_len, y0 + 2.0 * flap_depth),
            (-h, y0 + 2.0 * flap_depth),
        ])

    y0 = n_flaps * 2.0 * flap_depth
    pts.append([
        (-h, y0 + flap_depth),
        (-h, y0),
        (h, y0),
        (h, y0 + flap_depth),
        (-h, y0 + flap_depth),
    ])
    return pts


def flaps_slit_fam1(notch_len, depth, n_flaps, ribbon_width):
    """The family-1 (side-1, y=width-edge) counterpart to flaps_slit_fam0 --
    same comb pattern, but NOT a mirror image computed from it: this shape
    computes its own points directly against `ribbon_width` (pass the
    beam's `width`), reaching exactly up to the y=width edge on its own, the
    same way rectangular_slit's side-0 points reach exactly down to y=0.
    Because of that, it must be placed on the beam VERBATIM, not mirrored/
    reversed the way a side-0-only shape (like rectangular_slit or
    flaps_slit_fam0) would be when reused for side 1 -- pass it as
    `beam_outline`'s `slit1=`, which places it exactly as returned.

    Returns `n_flaps + 1` SEPARATE, unconnected polylines, same as
    flaps_slit_fam0 -- not merged into a single path."""
    if n_flaps < 1:
        raise ValueError("n_flaps must be >= 1")
    if depth > ribbon_width:
        raise ValueError("depth must be <= ribbon_width")

    flap_depth = depth / (2.0 * n_flaps + 1.0)
    h = notch_len / 2.0
    pts = []
    for id_flap in range(n_flaps):
        y0 = id_flap * 2.0 * flap_depth
        pts.append([
            (-h, y0 + 2.0 * flap_depth),
            (-h, y0 + flap_depth),
            (h, y0 + flap_depth),
            (h, y0 + 2.0 * flap_depth),
            (-h, y0 + 2.0 * flap_depth),
        ])

    y0 = (n_flaps * 2.0 + 1.0) * flap_depth
    pts.append([
        (-h, ribbon_width),
        (-h, y0),
        (h, y0),
        (h, ribbon_width),
        (-h, ribbon_width),
    ])
    return pts


def default_flap_slits(ribbon_width, ribbon_thickness, n_flaps=1):
    """This project's default slit design: flaps_slit_fam0/_fam1 (a comb/
    finger-joint notch, with more glue/friction surface than a plain
    rectangular half-lap), sized off the beam's own cross-section --
    notch_len = 1.5 * ribbon_thickness, flap_len = 4 * ribbon_thickness,
    depth = 2/3 * ribbon_width, n_flaps = 2 by default (e.g. ribbon_width =
    5mm, ribbon_thickness = 0.5mm -> notch_len = 0.75mm, flap_len = 2mm,
    depth = 3.33mm).

    ribbon_width: the beam's tall/along-normal cross-section dimension (its
    width as drawn in the SVG -- beam_outline's own `width` argument).
    ribbon_thickness: the beam's thin/in-plane cross-section dimension (the
    physical sheet material's thickness -- what the crossing beam's own
    material needs to slide through, hence notch_len is derived from it,
    not from ribbon_width).

    Returns (slit0, slit1, notch_len, depth) ready to pass straight through
    as beam_outline's slit=, slit1=, notch_len=, depth= -- flaps_slit_fam0/
    _fam1's own extra parameters (flap_len, n_flaps, ribbon_width) are
    pre-bound here via functools.partial, so beam_outline's call sites
    (`slit(notch_len, depth)`) don't need to know about them."""
    notch_len = 1.5 * ribbon_thickness
    flap_len = 4.0 * ribbon_thickness
    depth = (2.0 / 3.0) * ribbon_width
    slit0 = functools.partial(flaps_slit_fam0, flap_len=flap_len,
                              n_flaps=n_flaps, ribbon_width=ribbon_width)
    slit1 = functools.partial(flaps_slit_fam1, n_flaps=n_flaps,
                              ribbon_width=ribbon_width)
    return slit0, slit1, notch_len, depth


def _place_slit(shapes, x, side, width, mirror=True):
    """Absolute (x, y) polylines for one slit applied at local beam-x `x` on
    the given side -- `shapes` is a LIST of one or more independent
    polylines (as returned by a slit function, e.g. rectangular_slit
    returns one, flaps_slit_fam0/_fam1 return several separate, unconnected
    ones), each placed/transformed independently and kept separate in the
    returned list (never merged into each other). Side 0: each polyline is
    used verbatim (translated by x only) -- side-0 shapes are already in
    the canonical y=0-edge frame. Side 1: if `mirror` (default), each
    polyline is treated as a side-0-style shape being reused for the far
    edge, so it's mirrored (y -> width - y) and reversed (point order
    flipped, to match a side-0 shape's own orientation); if not, used
    verbatim, exactly as for side 0 -- for shapes (like flaps_slit_fam1)
    that already compute their own points directly against `width` and
    must not be re-flipped."""
    def _place_one(shape):
        if side == 0 or not mirror:
            return [(x + dx, y) for dx, y in shape]
        return [(x + dx, width - y) for dx, y in reversed(shape)]
    return [_place_one(shape) for shape in shapes]


def beam_outline(local_joints, strip_length, width, notch_len, depth=None,
                 slit=rectangular_slit, slit1=None):
    """A list of separate, independent polylines (each a list of (x, y) mm
    tuples -- not necessarily self-closing; open polylines, like
    flaps_slit_fam0/_fam1's individual flaps, are fine) making up one beam:
    the plain strip_length x width outer rectangle (self-closing), plus
    every polyline from each joint's slit shape in local_joints -- side 0
    from the y=0 edge, side 1 from the y=width edge. A slit shape's
    polylines typically overlap the outer rectangle along the edge segment
    they're cut from (by design -- a laser re-cutting that shared segment
    is harmless/redundant, and it's what lets each one stay an independent
    shape rather than a detour merged into one combined path). `depth`
    defaults to width/2 (half way through the ribbon).

    `slit(notch_len, depth)` supplies the side-0 shape(s) -- a LIST of one
    or more independent polylines (e.g. rectangular_slit returns one;
    flaps_slit_fam0 returns several separate, unconnected ones, one per
    flap) -- and, by default, is reused (mirrored/reversed) for side 1 too;
    this is exactly the original single-design behavior, appropriate
    whenever one shape design, cut from either edge, is a valid mirror
    image of itself.

    `slit1(notch_len, depth)`, if given, supplies a SEPARATE side-1 shape
    list used verbatim (no mirroring, no reversal) instead -- for a shape
    that was authored specifically for the y=width edge and already
    computes its own points there (e.g. flaps_slit_fam1, which needs
    `width` itself and so must be bound via functools.partial first, e.g.
    functools.partial(flaps_slit_fam1, ribbon_width=width, n_flaps=3)).

    Either `slit`/`slit1` needing more than (notch_len, depth) -- like
    flaps_slit_fam0/_fam1's `flap_len`/`ribbon_width`/`n_flaps` -- should be
    bound the same way via functools.partial before being passed in, so
    beam_outline's own call sites (`slit(notch_len, depth)`) never change."""
    depth = width / 2.0 if depth is None else depth
    # side 0 -> family 0 -> the bottom (y=0) edge; side 1 -> family 1 -> the
    # top (y=width) edge -- every joint on a real beam shares one side value
    # (its family), so exactly one of these two lists is ever non-empty in
    # practice, but both are handled for generality.
    bottom = sorted(x for x, side in local_joints if side == 0)
    top = sorted(x for x, side in local_joints if side == 1)

    side1_slit = slit if slit1 is None else slit1
    mirror_side1 = slit1 is None

    outer = [(0.0, 0.0), (strip_length, 0.0), (strip_length, width),
            (0.0, width), (0.0, 0.0)]
    shapes = [outer]
    # Every polyline (outer rectangle, every slit's every polyline) is kept
    # independent -- none are merged into a combined path -- so unlike the
    # old single-path design, the order joints are visited in doesn't
    # matter for validity.
    for x in bottom:
        shapes.extend(_place_slit(slit(notch_len, depth), x, side=0, width=width))
    for x in top:
        shapes.extend(_place_slit(side1_slit(notch_len, depth), x, side=1,
                                  width=width, mirror=mirror_side1))
    return shapes


def label_position(local_joints, width):
    """Local (x, y) anchor for a beam's label: horizontally centered between
    its first and second joint (by x), guaranteed clear of every slit, since
    slits only exist at joint x-positions. If a beam has only one joint,
    centered between the beam's start (x=0, the overhang -- always
    slit-free) and that joint instead."""
    xs = sorted(x for x, _side in local_joints)
    x = 0.5 * (xs[0] + xs[1]) if len(xs) >= 2 else xs[0] / 2.0
    return x, width / 2.0


def joint_label_position(x, side, width, depth):
    """Local (x, y) anchor for one joint's crossing-beam label (for
    pack_svg_guide), centered on the slit at local beam-x `x` and sunk
    half-way into the notch's own depth from whichever edge it's cut from
    -- depth/2 from y=0 for side 0, depth/2 up from y=width for side 1 --
    so the text sits on the crossing beam's actual footprint rather than out
    on the strip's plain body."""
    y = depth / 2.0 if side == 0 else width - depth / 2.0
    return x, y


def pack_svg(beams, gap=1.0, margin=5.0):
    """beams: [(shapes, label, label_xy), ...] -- shapes: a list of
    independent (x, y) mm polylines, as returned by beam_outline() (local
    beam frame) -- the outer rectangle plus every polyline from every
    joint's slit. Each is drawn as its own <polyline> exactly as given, so
    it's each polyline's own job (not pack_svg's) to close itself (repeat
    its first point as its last) if it needs to look closed -- some don't
    (e.g. flaps_slit_fam0/_fam1's individual flaps are open by design).
    label: str or None. label_xy: the local (x, y) to center it at (e.g.
    from label_position()). Stacks beams left-aligned (shared x=0) top to
    bottom with `gap` mm between. Returns a complete SVG document string
    (units = mm, 1 svg-unit = 1mm).

    This function is family-agnostic: it never looks at which family a beam
    belongs to and draws every shape in the same CUT_COLOR. Family 0 vs.
    family 1 is already fully encoded in each slit's geometry by this point
    (which edge it indents from -- see beam_outline), so there is nothing
    left for pack_svg to distinguish."""
    if not beams:
        raise ValueError("no beams to pack")
    max_len = max(max(x for shape in shapes for x, _y in shape)
                  for shapes, _label, _pos in beams)

    y_cursor = margin
    body = []
    for shapes, label, (lx, ly) in beams:
        beam_w = max(y for shape in shapes for _x, y in shape)
        for shape in shapes:
            pts = " ".join(f"{x + margin:.3f},{y + y_cursor:.3f}" for x, y in shape)
            body.append(f'<polyline points="{pts}" fill="none" '
                        f'stroke="{CUT_COLOR}" stroke-width="0.1"/>')
        if label:
            body.append(
                f'<text x="{margin + lx:.3f}" y="{y_cursor + ly:.3f}" '
                f'font-size="{0.6 * beam_w:.3f}" fill="{LABEL_COLOR}" '
                f'text-anchor="middle" dominant-baseline="middle">{label}</text>')
        y_cursor += beam_w + gap

    canvas_w = max_len + 2.0 * margin
    canvas_h = y_cursor - gap + margin
    header = (f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{canvas_w:.3f}mm" height="{canvas_h:.3f}mm" '
             f'viewBox="0 0 {canvas_w:.3f} {canvas_h:.3f}">')
    return header + "".join(body) + "</svg>"


def pack_svg_guide(beams, gap=1.0, margin=5.0):
    """Assembly-guide variant of pack_svg: same beam outlines, packed the
    same left-aligned, top-to-bottom way, but each beam's tuple carries a
    fourth element, `joint_labels` -- [(x, y, text), ...] in the beam's own
    local frame (e.g. from joint_label_position), one entry per slit --
    engraved as small GUIDE_COLOR text centered on that slit, naming the
    OTHER family's beam that crosses through it there. So a beam's sheet
    doubles as an assembly aid: e.g. beam A0's guide shows "B3" sitting
    inside the slit where rod B3 notches in, alongside A0's own label --
    without needing to cross-reference the joint list separately.

    beams: [(shapes, label, label_xy, joint_labels), ...]. Not a cut file --
    GUIDE_COLOR is a third color distinct from CUT_COLOR/LABEL_COLOR so a
    laser workflow that only reacts to CUT_COLOR ignores these engravings
    same as it already ignores LABEL_COLOR ones. Returns a complete SVG
    document string, same conventions as pack_svg."""
    if not beams:
        raise ValueError("no beams to pack")
    max_len = max(max(x for shape in shapes for x, _y in shape)
                  for shapes, _label, _pos, _joints in beams)

    y_cursor = margin
    body = []
    for shapes, label, (lx, ly), joint_labels in beams:
        beam_w = max(y for shape in shapes for _x, y in shape)
        for shape in shapes:
            pts = " ".join(f"{x + margin:.3f},{y + y_cursor:.3f}" for x, y in shape)
            body.append(f'<polyline points="{pts}" fill="none" '
                        f'stroke="{CUT_COLOR}" stroke-width="0.1"/>')
        if label:
            body.append(
                f'<text x="{margin + lx:.3f}" y="{y_cursor + ly:.3f}" '
                f'font-size="{0.6 * beam_w:.3f}" fill="{LABEL_COLOR}" '
                f'text-anchor="middle" dominant-baseline="middle">{label}</text>')
        for jx, jy, text in joint_labels:
            body.append(
                f'<text x="{margin + jx:.3f}" y="{y_cursor + jy:.3f}" '
                f'font-size="{0.35 * beam_w:.3f}" fill="{GUIDE_COLOR}" '
                f'text-anchor="middle" dominant-baseline="middle">{text}</text>')
        y_cursor += beam_w + gap

    canvas_w = max_len + 2.0 * margin
    canvas_h = y_cursor - gap + margin
    header = (f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{canvas_w:.3f}mm" height="{canvas_h:.3f}mm" '
             f'viewBox="0 0 {canvas_w:.3f} {canvas_h:.3f}">')
    return header + "".join(body) + "</svg>"
