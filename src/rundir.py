"""Per-trial output layout.

Each trace of a surface gets its own numbered trial directory under output/,
so repeated runs don't clobber each other and the different render types stay
organized:

    output/<surface>_<NNN>/
        geom/                  net.obj, net.json, waffle.json
        render/
            positions/           reusable camera .position files
            geometry/            plain (uncolored) gridshell renders
            energy/              renders colored by a metric (SqrtBend, ...)
            equilibrium/         equilibrium-solve frame sequences (no target surface)
            equilibrium_surface/ equilibrium-solve frame sequences with the target surface shown
            preview/             quick matplotlib debug plots
"""
import re
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / 'output'
RENDER_KINDS = ('geometry', 'energy', 'equilibrium', 'equilibrium_surface', 'preview')


def _trials(surface):
    pat = re.compile(rf'^{re.escape(surface)}_(\d+)$')
    found = []
    if OUTPUT.exists():
        for p in OUTPUT.iterdir():
            m = pat.match(p.name)
            if p.is_dir() and m:
                found.append((int(m.group(1)), p))
    return sorted(found)


def new_trial(surface):
    """Create and return a fresh output/<surface>_<NNN>/ trial directory."""
    existing = _trials(surface)
    next_id = (existing[-1][0] + 1) if existing else 0
    run = OUTPUT / f'{surface}_{next_id:03d}'
    (run / 'geom').mkdir(parents=True)
    (run / 'render' / 'positions').mkdir(parents=True)
    for kind in RENDER_KINDS:
        (run / 'render' / kind).mkdir(parents=True)
    return run


def trial_dir(name, id):
    """Return (output/<name>_<id>/, created) for an EXACT (name, id) pair
    chosen by the caller -- id: an int or the zero-padded string form.
    Unlike new_trial, which always picks the next free number, this reuses
    the same directory (and its existing contents) on repeated calls with
    the same pair rather than incrementing past it; its geom/ + render/*
    skeleton is created (or completed, if only partially present) as
    needed. `created` is True the first time this exact pair is used,
    False if the directory already existed."""
    if isinstance(id, int):
        id = f'{id:03d}'
    run = OUTPUT / f'{name}_{id}'
    created = not run.is_dir()
    (run / 'geom').mkdir(parents=True, exist_ok=True)
    (run / 'render' / 'positions').mkdir(parents=True, exist_ok=True)
    for kind in RENDER_KINDS:
        (run / 'render' / kind).mkdir(parents=True, exist_ok=True)
    return run, created


def trial_names():
    """Every distinct trial base name (the '<name>' in output/<name>_<NNN>/)
    with at least one existing trial, sorted alphabetically."""
    pat = re.compile(r'^(.+)_(\d+)$')
    names = set()
    if OUTPUT.exists():
        for p in OUTPUT.iterdir():
            m = pat.match(p.name)
            if p.is_dir() and m:
                names.add(m.group(1))
    return sorted(names)


def trial_ids(name):
    """Every existing trial id for base name `name`, as zero-padded 3-digit
    strings (e.g. '000'), ascending -- the '<NNN>' in output/<name>_<NNN>/."""
    return [f'{i:03d}' for i, _p in _trials(name)]


def resolve_trial(name):
    """Resolve `name` to a trial directory: an explicit '<surface>_<NNN>' name
    resolves directly, a bare surface name resolves to its latest trial."""
    direct = OUTPUT / name
    if (direct / 'geom').is_dir():
        return direct
    existing = _trials(name)
    if not existing:
        raise FileNotFoundError(f"no trial found for '{name}' under {OUTPUT}")
    return existing[-1][1]
