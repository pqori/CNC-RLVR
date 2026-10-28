"""World model: tools, fixed cutting policy, features, task generation.

Coordinates: part frame, mm. Stock is [0,60]x[0,60]x[0,20], z up.
Top face = Z+ (tool approaches from +z), bottom = Z-, sides = X+/X-/Y+/Y-.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

STOCK = (60.0, 60.0, 20.0)
RES = 0.2          # training-grid voxel size, mm
AUDIT_RES = 0.1    # routine audit grid (every eval)
SPOT_RES = 0.05    # spot-audit grid (escalations + random sample)
TOL_MM = 0.1       # declared part tolerance, mm -- PHYSICAL and
                   # grid-independent; every grid approximates the
                   # +-TOL_MM band as well as its pitch allows (sim.py)


@dataclass(frozen=True)
class Tool:
    tool_id: str
    kind: str          # "mill" | "drill"
    diameter: float    # mm
    flutes: int
    flute_len: float   # cutting length, mm
    stickout: float    # tip to holder face, mm
    holder_d: float    # holder diameter, mm
    center_cutting: bool


# The crib's endmills are ALL non-center-cutting -- a deliberate
# problem-space decision: closed-cavity entries must be AUTHORED
# (helix where the cavity has room, angle-limited ramp where it does
# not); no mill may feed straight down into material. Drills are the
# only axial-plunging tools.
TOOLS = {t.tool_id: t for t in [
    Tool("EM3",  "mill",  3.0, 2,  8.0, 20.0, 16.0, False),
    Tool("EM3L", "mill",  3.0, 2, 14.0, 28.0, 16.0, False),
    Tool("EM6",  "mill",  6.0, 3, 13.0, 25.0, 20.0, False),
    Tool("EM10", "mill", 10.0, 3, 22.0, 35.0, 25.0, False),
    Tool("EM12", "mill", 12.0, 4, 26.0, 42.0, 25.0, False),
    Tool("D3",   "drill", 3.0, 2, 20.0, 35.0, 16.0, True),
    Tool("D5",   "drill", 5.0, 2, 28.0, 45.0, 20.0, True),
    Tool("D8",   "drill", 8.0, 2, 40.0, 60.0, 25.0, True),
]}


@dataclass(frozen=True)
class Material:
    name: str
    vc_mill: tuple     # (lo, hi) surface speed band, m/min
    vc_drill: tuple
    fz_per_d: tuple    # chipload band as fraction of diameter, mm/tooth/mm
    taylor_n: float
    taylor_c: float
    kc: float          # specific cutting force, N/mm^2 (load model)


MATERIALS = {
    "AL6061": Material("AL6061", (150.0, 400.0), (60.0, 120.0),
                       (0.004, 0.012), 0.40, 900.0, kc=800.0),
}


@dataclass(frozen=True)
class CutParams:
    """Canonical AL6061 spindle speed and cutting feed for one tool.

    Cutting parameters are controlled experimental conditions, not policy
    actions. Values are the rounded midpoint of the published AL6061 bands:
    mills use Vc=275 m/min, drills Vc=90 m/min, and both use
    fz=0.008*D mm/tooth. Every accepted program must use these exact values;
    descending mill entries use ENTRY_FEED_FRAC in expand.py.
    """
    rpm: int
    feed: float


CUT_PARAMS = {
    "EM3":  CutParams(29178, 1400.5),
    "EM3L": CutParams(29178, 1400.5),
    "EM6":  CutParams(14589, 2100.8),
    "EM10": CutParams(8754, 2101.0),
    "EM12": CutParams(7295, 2801.3),
    "D3":   CutParams(9549, 458.4),
    "D5":   CutParams(5730, 458.4),
    "D8":   CutParams(3581, 458.4),
}
assert CUT_PARAMS.keys() == TOOLS.keys()

# Entry contract (paired with the all-non-center-cutting crib above):
# a CUTTING mill move may descend no steeper than MAX_RAMP_DEG. A
# pure-vertical mill feed into material is plunge_non_center; without
# the angle cap a "ramp" at 89.9 deg with a micron of XY drift would
# be a plunge with a fake moustache, and the sweep's slant staircase
# (over-removal <= chord * tan(angle)) also degrades with steepness.
# The teacher authors entries at 3 deg (expand.RAMP_ANGLE_DEG); the
# cap leaves headroom for legitimate student-authored pitches. Drills
# are exempt (axial by definition; lateral cutting is drill_lateral).
MAX_RAMP_DEG = 5.0

# Helix feasibility is a GENERATOR INVARIANT, not a per-plan hope: a
# helix bore needs min cavity dimension >= HELIX_ROOM_FACTOR * D
# (enforced at expansion), and make_task only emits milled features
# with room AND reach for at least the smallest endmill family
# (D = 3 mm: EM3 to 8 mm deep, EM3L to 14 mm) -- so a helix entry
# always exists for some tool. Concretely this floors slot widths at
# 6 mm (a 4 mm slot admits no 3 mm helix: 4 < 4.8) and caps milled
# depths at the long 3 mm flute. Asserted per generated task.
HELIX_ROOM_FACTOR = 1.6
_MIN_MILL_D = min(t.diameter for t in TOOLS.values() if t.kind == "mill")
HELIX_MIN_ROOM_MM = HELIX_ROOM_FACTOR * _MIN_MILL_D           # 4.8
SMALL_MILL_REACH_MM = max(t.flute_len for t in TOOLS.values()
                          if t.kind == "mill"
                          and t.diameter == _MIN_MILL_D)      # 14 (EM3L)

# ---- load model (engagement grading) ----
# Mean tangential cutting force per move: F = P_cut / v_c, where
# P_cut = kc * MRR and MRR is MEASURED from the voxel sweep (removed
# volume / move time), so engagement is graded from material state,
# not from nominal parameters. Caps:
SPINDLE_POWER_W = 5000.0    # spindle power ceiling, all tools
CARBIDE_E_MPA = 600000.0    # Young's modulus, carbide
DEFLECT_FRAC = 0.01         # allowed tip deflection as fraction of D


def tool_force_limit_n(tool) -> float:
    """Max allowed mean tangential force on an endmill: cantilever tip
    deflection delta = F L^3 / (3 E I) capped at 1% of diameter -- the
    chatter/tolerance limit that actually governs roughing, and far
    tighter than static breakage. F_lim = 3 E (pi D^4/64)(0.01 D)/L^3.
    Drills fail by axial thrust/torque, not bending; under the fixed
    drilling schedule they are policed by the spindle power cap."""
    import math
    if tool.kind == "drill":
        return math.inf
    I = math.pi * tool.diameter ** 4 / 64.0
    return (3.0 * CARBIDE_E_MPA * I * DEFLECT_FRAC * tool.diameter
            / tool.stickout ** 3)


@dataclass
class Feature:
    """One machinable feature. Kinds:
    - pocket_rect: cx, cy, w, h, depth, corner_r
    - pocket_circ: cx, cy, d, depth
    - slot:        x1, y1, x2, y2, width, depth
    - hole:        u, v, d, depth
    Any kind may live on any of the six faces. Face is the orientation
    the tool must approach from. All coordinates are FACE-LOCAL (u, v),
    matching sim._face_view: Z faces (u,v)=(x,y); X faces (u,v)=(y,z);
    Y faces (u,v)=(x,z). For Z faces this coincides with part-frame x,y.
    """
    name: str
    kind: str
    face: str
    params: dict

    def to_prompt_line(self) -> str:
        p = self.params
        if self.kind == "pocket_rect":
            return (f"{self.name}: rect pocket on {self.face}, center "
                    f"({p['cx']:.1f},{p['cy']:.1f}), {p['w']:.1f}x{p['h']:.1f} mm, "
                    f"depth {p['depth']:.1f} mm, corner radius {p['corner_r']:.1f} mm")
        if self.kind == "pocket_circ":
            return (f"{self.name}: circular pocket on {self.face}, center "
                    f"({p['cx']:.1f},{p['cy']:.1f}), diameter {p['d']:.1f} mm, "
                    f"depth {p['depth']:.1f} mm")
        if self.kind == "slot":
            return (f"{self.name}: slot on {self.face}, from "
                    f"({p['x1']:.1f},{p['y1']:.1f}) to ({p['x2']:.1f},{p['y2']:.1f}), "
                    f"width {p['width']:.1f} mm, depth {p['depth']:.1f} mm")
        if self.kind == "hole":
            return (f"{self.name}: hole on {self.face}, at "
                    f"({p['u']:.1f},{p['v']:.1f}), diameter {p['d']:.1f} mm, "
                    f"depth {p['depth']:.1f} mm")
        raise ValueError(self.kind)


@dataclass
class Task:
    seed: int
    split: str
    material: str
    features: list

    def feature(self, name: str):
        for f in self.features:
            if f.name == name:
                return f
        return None

    def prompt(self) -> str:
        lines = [f"Stock: {STOCK[0]:.0f}x{STOCK[1]:.0f}x{STOCK[2]:.0f} mm, "
                 f"material {self.material}.",
                 "Features to machine:"]
        lines += ["  " + f.to_prompt_line() for f in self.features]
        return "\n".join(lines)

    def orientations_required(self) -> list:
        seen = []
        for f in self.features:
            if f.face not in seen:
                seen.append(f.face)
        return seen


# ---------------------------------------------------------------- geometry

def _grid_axes(res):
    nx, ny, nz = (int(round(s / res)) for s in STOCK)
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    zs = (np.arange(nz) + 0.5) * res
    return xs, ys, zs


def _rounded_rect_mask(xs, ys, cx, cy, w, h, r):
    """2D mask of a rounded rectangle footprint."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx = np.abs(X - cx) - (w / 2 - r)
    dy = np.abs(Y - cy) - (h / 2 - r)
    dx = np.maximum(dx, 0.0)
    dy = np.maximum(dy, 0.0)
    return dx * dx + dy * dy <= r * r


def _capsule_mask(xs, ys, x1, y1, x2, y2, r):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    vx, vy = x2 - x1, y2 - y1
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        t = np.zeros_like(X)
    else:
        t = np.clip(((X - x1) * vx + (Y - y1) * vy) / L2, 0.0, 1.0)
    px, py = x1 + t * vx, y1 + t * vy
    return (X - px) ** 2 + (Y - py) ** 2 <= r * r


def _inflated(f: Feature, t: float) -> Feature:
    """The feature grown by t mm in every surface-normal direction:
    lateral dims +2t (radius-like dims +t), depth +t. The offset of a
    rounded rectangle is a rounded rectangle with corner_r+t; of a
    capsule, a capsule with width+2t -- both exact."""
    p = dict(f.params)
    if f.kind == "pocket_rect":
        p["w"] += 2 * t; p["h"] += 2 * t
        p["corner_r"] += t; p["depth"] += t
    elif f.kind == "pocket_circ":
        p["d"] += 2 * t; p["depth"] += t
    elif f.kind == "slot":
        p["width"] += 2 * t; p["depth"] += t
    elif f.kind == "hole":
        p["d"] += 2 * t; p["depth"] += t
    else:
        raise ValueError(f.kind)
    return Feature(f.name, f.kind, f.face, p)


def target_solid(task: Task, res: float = RES,
                 inflate_mm: float = 0.0) -> np.ndarray:
    """Boolean voxel array of the FINISHED part (True = material).

    inflate_mm > 0 returns the part with every feature grown by that
    margin: the MUST-KEEP envelope. Material outside a feature's
    +inflate_mm offset may never be cut -- the verifier's gouge region.
    Analytic (from the same feature geometry as the nominal target),
    so it has none of the concave-corner traps of voxel erosion."""
    xs, ys, zs = _grid_axes(res)
    solid = np.ones((len(xs), len(ys), len(zs)), dtype=bool)
    for f in task.features:
        if inflate_mm > 0.0:
            f = _inflated(f, inflate_mm)
        if f.kind in ("pocket_rect", "pocket_circ", "slot"):
            solid[_milled_mask(xs, ys, zs, f)] = False
        elif f.kind == "hole":
            solid[_hole_mask(xs, ys, zs, f)] = False
        else:
            raise ValueError(f.kind)
    return solid


def _milled_mask(xs, ys, zs, f: Feature) -> np.ndarray:
    """Milled feature (pocket/slot) on ANY face: 2D footprint built in
    the face's local (u, v) plane, extruded `depth` into the stock from
    that face. The (u, v) frame matches _hole_mask and sim._face_view."""
    p = f.params
    ax = f.face[0]
    a1, a2 = {"Z": (xs, ys), "X": (ys, zs), "Y": (xs, zs)}[ax]
    if f.kind == "pocket_rect":
        m2 = _rounded_rect_mask(a1, a2, p["cx"], p["cy"],
                                p["w"], p["h"], p["corner_r"])
    elif f.kind == "pocket_circ":
        m2 = _capsule_mask(a1, a2, p["cx"], p["cy"], p["cx"], p["cy"],
                           p["d"] / 2)
    else:
        m2 = _capsule_mask(a1, a2, p["x1"], p["y1"], p["x2"], p["y2"],
                           p["width"] / 2)
    d = p["depth"]
    if ax == "Z":
        span = zs > STOCK[2] - d if f.face == "Z+" else zs < d
        return m2[:, :, None] & span[None, None, :]
    if ax == "X":
        span = xs > STOCK[0] - d if f.face == "X+" else xs < d
        return span[:, None, None] & m2[None, :, :]
    span = ys > STOCK[1] - d if f.face == "Y+" else ys < d
    return m2[:, None, :] & span[None, :, None]


def _hole_mask(xs, ys, zs, f: Feature) -> np.ndarray:
    """Cylinder along the face's axis, entered from that face."""
    p = f.params
    r, depth = p["d"] / 2, p["depth"]
    ax = f.face[0]
    if ax == "Z":
        m2 = _capsule_mask(xs, ys, p["u"], p["v"], p["u"], p["v"], r)
        span = zs > STOCK[2] - depth if f.face == "Z+" else zs < depth
        return m2[:, :, None] & span[None, None, :]
    if ax == "X":
        m2 = _capsule_mask(ys, zs, p["u"], p["v"], p["u"], p["v"], r)  # (y,z)
        span = xs > STOCK[0] - depth if f.face == "X+" else xs < depth
        return span[:, None, None] & m2[None, :, :]
    # Y faces: footprint in (x,z)
    m2 = _capsule_mask(xs, zs, p["u"], p["v"], p["u"], p["v"], r)
    span = ys > STOCK[1] - depth if f.face == "Y+" else ys < depth
    return m2[:, None, :] & span[None, :, None]


# ---------------------------------------------------------------- task gen

def _rng_feature_positions(rng, n, wmax=26.0, margin=6.0):
    """Rejection-sample n non-overlapping footprint centers/sizes (top view)."""
    placed = []
    for _ in range(400):
        if len(placed) == n:
            break
        w = rng.uniform(10.0, wmax)
        h = rng.uniform(10.0, wmax)
        cx = rng.uniform(margin + w / 2, STOCK[0] - margin - w / 2)
        cy = rng.uniform(margin + h / 2, STOCK[1] - margin - h / 2)
        ok = all(abs(cx - q[0]) > (w + q[2]) / 2 + 4.0
                 or abs(cy - q[1]) > (h + q[3]) / 2 + 4.0 for q in placed)
        if ok:
            placed.append((cx, cy, w, h))
    return placed


def make_task(seed: int, split: str = "train") -> Task:
    """Milled features (rectangular AND circular pockets, slots at any
    angle) and drilled holes on any of the six faces; side faces use
    side-appropriate dimensions (20 mm tall).

    Splits:
    - train:        1-3 orientations, corner_r=2 pockets capped depth<=7.5
    - eval_iid:     same distribution as train, fresh seeds
    - oot_multitool: one deep corner_r=2 pocket (depth 9-13) + fillers
    - oot_setups:   3 orientations, feature order adversarially interleaved
    """
    rng = np.random.default_rng(seed)
    # Material is fixed so RL optimizes process structure and toolpaths, not
    # feeds-and-speeds. Consume the historical binary material draw to keep
    # each seed's geometry stable across this benchmark revision.
    rng.choice(2)
    material = "AL6061"
    feats = []

    def add_pocket(i, face, deep_tight=False, allow_tight=True):
        cx, cy, w, h = _rng_feature_positions(rng, 1)[0]
        if deep_tight:
            corner_r, depth = 2.0, float(rng.uniform(9.0, 13.0))
            w, h = float(rng.uniform(18, 30)), float(rng.uniform(18, 30))
        else:
            corner_r = float(rng.choice([2.0, 3.0, 5.0] if allow_tight
                                        else [3.0, 5.0]))
            depth = float(rng.uniform(3.0, 7.5 if corner_r == 2.0 else 12.0))
        feats.append(Feature(f"F{i}", "pocket_rect", face, dict(
            cx=cx, cy=cy, w=w, h=h, depth=depth, corner_r=corner_r)))

    def add_hole(i, face):
        if face[0] == "Z":
            u = float(rng.uniform(8, 52)); v = float(rng.uniform(8, 52))
            depth = float(rng.uniform(4, 15))
        else:
            u = float(rng.uniform(10, 50))       # along the face
            v = float(rng.uniform(5, 15))        # height on the side face
            depth = float(rng.uniform(6, 20))
        d = float(rng.choice([3.0, 5.0, 8.0]))
        feats.append(Feature(f"F{i}", "hole", face, dict(
            u=u, v=v, d=d, depth=depth)))

    def add_side_pocket(i, face):
        """Milled pocket on a side face: 60 mm wide, 20 mm tall, so the
        v-extent is tight. Same corner_r/depth blind-spot cap as Z faces.
        Height floors at 10 mm: that clears EM6's helix bore (9.6 mm),
        so corner_r=3 side pockets never push the teacher onto the
        slender 3 mm family at depths where the load model snaps it."""
        corner_r = float(rng.choice([2.0, 3.0]))
        depth = float(rng.uniform(3.0, 7.5 if corner_r == 2.0 else 10.0))
        w = float(rng.uniform(10.0, 24.0))
        h = float(rng.uniform(10.0, 13.0))
        cu = float(rng.uniform(6.0 + w / 2, STOCK[0] - 6.0 - w / 2))
        cv = float(rng.uniform(3.5 + h / 2, STOCK[2] - 3.5 - h / 2))
        feats.append(Feature(f"F{i}", "pocket_rect", face, dict(
            cx=cu, cy=cv, w=w, h=h, depth=depth, corner_r=corner_r)))

    def add_side_slot(i, face):
        v = float(rng.uniform(5.0, 15.0))
        u1, u2 = sorted(rng.uniform(8, 52, size=2))
        if u2 - u1 < 12:
            u2 = min(52.0, u1 + 12)
        feats.append(Feature(f"F{i}", "slot", face, dict(
            x1=float(u1), y1=v, x2=float(u2), y2=v,
            width=6.0,      # helix-room floor; the 20 mm side face
                            # keeps side slots at the narrow end
            depth=float(rng.uniform(2.0, 6.0))))
        )

    def add_circ_pocket(i, face):
        """Circular pocket. On Z faces the diameter rides the shared
        footprint sampler; on side faces the 20 mm face height caps it."""
        if face[0] == "Z":
            cx, cy, w, h = _rng_feature_positions(rng, 1)[0]
            d = float(min(w, h))                      # 10-26 mm
            depth = float(rng.uniform(3.0, 12.0))
        else:
            # floor 10 mm = EM6 helix room, same rationale as side pockets
            d = float(rng.uniform(10.0, 12.0))
            cx = float(rng.uniform(6.0 + d / 2, STOCK[0] - 6.0 - d / 2))
            cy = float(rng.uniform(3.5 + d / 2, STOCK[2] - 3.5 - d / 2))
            depth = float(rng.uniform(3.0, 8.0))
        feats.append(Feature(f"F{i}", "pocket_circ", face, dict(
            cx=cx, cy=cy, d=d, depth=depth)))

    def add_diag_slot(i, face):
        """Slot at an arbitrary angle (Z faces): center + angle +
        half-length capped so both endpoints stay in the safe margin."""
        cx = float(rng.uniform(18.0, 42.0))
        cy = float(rng.uniform(18.0, 42.0))
        ang = float(rng.uniform(0.0, np.pi))
        c, s = float(np.cos(ang)), float(np.sin(ang))
        lim = min((cx - 8.0) / max(abs(c), 1e-9),
                  (52.0 - cx) / max(abs(c), 1e-9),
                  (cy - 8.0) / max(abs(s), 1e-9),
                  (52.0 - cy) / max(abs(s), 1e-9))
        hl = float(min(rng.uniform(7.0, 15.0), lim))
        feats.append(Feature(f"F{i}", "slot", face, dict(
            x1=cx - hl * c, y1=cy - hl * s,
            x2=cx + hl * c, y2=cy + hl * s,
            width=float(rng.choice([6.0, 8.0])),
            depth=float(rng.uniform(2.0, 6.0)))))

    def add_slot(i, face):
        y = float(rng.uniform(10, 50))
        x1, x2 = sorted(rng.uniform(8, 52, size=2))
        if x2 - x1 < 12:
            x2 = min(52.0, x1 + 12)
        feats.append(Feature(f"F{i}", "slot", face, dict(
            x1=float(x1), y1=y, x2=float(x2), y2=y,
            width=float(rng.choice([6.0, 8.0])),
            depth=float(rng.uniform(2.0, 6.0))))
        )

    i = 1
    if split in ("train", "eval_iid"):
        n_orient = int(rng.choice([1, 1, 2, 2, 3]))
        orients = ["Z+"] + list(rng.choice(["Z-", "X+", "X-", "Y+", "Y-"],
                                           size=max(0, n_orient - 1),
                                           replace=False))
        for face in orients:
            if face[0] == "Z":
                for _ in range(int(rng.integers(1, 3))):
                    kind = rng.choice(
                        ["pocket", "circ", "hole", "slot", "diag_slot"],
                        p=[0.35, 0.15, 0.25, 0.13, 0.12])
                    if kind == "pocket":
                        add_pocket(i, face)
                    elif kind == "circ":
                        add_circ_pocket(i, face)
                    elif kind == "hole":
                        add_hole(i, face)
                    elif kind == "slot":
                        add_slot(i, face)
                    else:
                        add_diag_slot(i, face)
                    i += 1
            else:
                for _ in range(int(rng.integers(1, 3))):
                    kind = rng.choice(["pocket", "circ", "hole", "slot"],
                                      p=[0.3, 0.1, 0.4, 0.2])
                    if kind == "pocket":
                        add_side_pocket(i, face)
                    elif kind == "circ":
                        add_circ_pocket(i, face)
                    elif kind == "hole":
                        add_hole(i, face)
                    else:
                        add_side_slot(i, face)
                    i += 1
        rng.shuffle(feats)
        feats[:] = [Feature(f"F{k+1}", f.kind, f.face, f.params)
                    for k, f in enumerate(feats)]
    elif split == "oot_multitool":
        add_pocket(i, "Z+", deep_tight=True); i += 1
        add_hole(i, "Z+"); i += 1
    elif split == "oot_setups":
        # three orientations, ordering interleaved to punish greedy planners
        sides = list(rng.choice(["X+", "X-", "Y+", "Y-"], size=2,
                                replace=False))
        seq = ["Z+", sides[0], "Z+", sides[1], sides[0]]
        for face in seq:
            if face[0] == "Z":
                add_pocket(i, face, allow_tight=False)
            elif rng.random() < 0.5:
                add_side_pocket(i, face)
            else:
                add_hole(i, face)
            i += 1
    else:
        raise ValueError(split)

    # problem-space invariant: every milled feature admits a helix
    # entry by the smallest endmill family -- room for its bore AND
    # flute reach to full depth. The teacher may pick bigger tools;
    # this guarantees a helix-capable tool always EXISTS.
    for f in feats:
        if f.kind == "hole":
            continue
        p = f.params
        room = (min(p["w"], p["h"]) if f.kind == "pocket_rect"
                else p["d"] if f.kind == "pocket_circ"
                else p["width"])
        assert room >= HELIX_MIN_ROOM_MM - 1e-9, \
            f"seed {seed} {f.name}: no helix room ({room:.2f} mm)"
        assert p["depth"] <= SMALL_MILL_REACH_MM + 1e-9, \
            f"seed {seed} {f.name}: beyond small-mill reach"

    return Task(seed=seed, split=split, material=material, features=feats)
