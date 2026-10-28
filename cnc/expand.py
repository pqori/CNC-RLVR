"""Deterministic path expander: typed ops -> motion primitives -> moves.

The TEACHER-side motion generator (the student writes G-code directly;
this module produces demonstrations and scripted baselines). Every op
expands, in setup-local coordinates (u, v, w) with w the depth below
the entered face, into motion Prims (rapids, lines, arcs/helixes,
drill strokes) with REAL entry moves -- helix, ramp and plunge are
geometry that gets swept and timed, not a coefficient.

One canonical motion program per plan (plan_program): ops are stitched
with explicit clearance-height hops so the stream is CONTINUOUS (each
prim starts where the previous ended). emit_gcode serializes it one
line per prim; cnc/gcode.py parses the text back into the same
structure. Round-trip exactness (audited): parse(emit(plan)) samples
into bit-identical chord Moves, hence the identical voxel sweep. The
twin grades ONE motion language for both arms; the expander is just
one author of it.

Exactness rules that make the round trip hold:
- every Prim field is quantized at construction (QUANT = 0.001 mm,
  the G-code text precision; feeds to 0.1 mm/min),
- arc geometry (radius, angles, sweep) is DERIVED from the quantized
  endpoints/center in sample_prims -- never from stored analytic
  angles -- so both sides sample the same chords bit-for-bit,
- degenerate prims are normalized identically on both sides
  (normalize_prims).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, replace

from .ir import Op, Plan
from .world import TOOLS, CUT_PARAMS, Feature, Task, HELIX_ROOM_FACTOR

SAGITTA_MAX = 0.008    # mm max chord error on sampled arcs
QUANT = 0.001          # mm coordinate quantum (= G-code text precision)
TOOLCHANGE_S = 8.0
SETUP_CHANGE_S = 180.0
RAPID_FEED = 5000.0    # mm/min rapid traverse rate (time model)
CLEAR_W = -2.0         # clearance plane, 2 mm ABOVE the face (w < 0)
APPROACH_W_GAP = 0.5   # approach/retract plane above the resume floor
MAX_FEED = 10000.0     # machine feed ceiling (contract bound)
MIN_ARC_R = 0.005      # arcs below this radius normalize to lines

# Entry geometry (teacher policy; the student writes its own entries).
# HELIX_ROOM_FACTOR lives in world.py: it is a problem-space contract
# (the generator guarantees every milled feature clears it for the
# smallest endmill family), not just an expander parameter.
HELIX_RADIUS_FRAC = 0.45     # helix radius = 0.45 * D (bore dia 0.9*D)
RAMP_ANGLE_DEG = 3.0         # ramp/helix descent angle
ENTRY_FEED_FRAC = 0.5        # entry moves run at half the op feed
PLUNGE_FEED_FRAC = 0.3       # straight mill plunges at 0.3x op feed
# Geometric sweep chord bound. Load timing is integrated from newly hit
# voxel-contact intervals in Verifier, so this cannot dilute engagement.
SWEEP_CHORD_MM = 2.0


def q(x: float) -> float:
    """Quantize to machine resolution (also the G-code text quantum)."""
    return round(round(x / QUANT) * QUANT, 3)


def fq(f: float) -> float:
    """Quantize a canonical feed to 0.1 mm/min."""
    return round(f, 1)


# ---------------------------------------------------------------- prims

@dataclass
class Prim:
    """One motion primitive in setup-local coords, fields quantized.
    Kinds:
    - "rapid": straight 3D traverse, no cutting expected
    - "line":  straight feed move (slants allowed: ramps)
    - "arc":   XY arc around center (cu,cv) from (u1,v1) to (u2,v2),
               direction ccw; equal endpoints = full circle; w linear
               in swept angle (helix if w1 != w2)
    - "drill": vertical feed stroke at (u1,v1) from w1 down to w2
    """
    kind: str
    u1: float = 0.0; v1: float = 0.0; w1: float = 0.0
    u2: float = 0.0; v2: float = 0.0; w2: float = 0.0
    cu: float = 0.0; cv: float = 0.0
    ccw: bool = True
    feed: float = 0.0          # mm/min (rapids: RAPID_FEED)

    @property
    def r(self) -> float:
        return math.hypot(self.u1 - self.cu, self.v1 - self.cv)


def mk(kind, u1, v1, w1, u2, v2, w2, feed, cu=0.0, cv=0.0,
       ccw=True) -> Prim:
    # rapids carry feed 0 canonically (the text has no F on a G0, and
    # sample_prims times them at RAPID_FEED regardless)
    return Prim(kind, q(u1), q(v1), q(w1), q(u2), q(v2), q(w2),
                q(cu), q(cv), ccw, 0.0 if kind == "rapid" else fq(feed))


def normalize_prims(prims: list) -> list:
    """Canonical cleanup applied identically to expander output and
    parsed G-code (round-trip exactness depends on it): tiny arcs
    become lines, vertical lines become drill strokes (a G1 Z is not
    distinguishable from a 'plunge' in text), zero-motion prims drop,
    and contiguous collinear feeds merge so equivalent G1 segmentation
    cannot move load-window boundaries."""
    out = []
    for p in prims:
        if p.kind == "arc" and p.r < MIN_ARC_R:
            p = Prim("line", p.u1, p.v1, p.w1, p.u2, p.v2, p.w2,
                     feed=p.feed)
        if p.kind == "line" and abs(p.u2 - p.u1) < 1e-9 \
                and abs(p.v2 - p.v1) < 1e-9:
            p = Prim("drill", p.u1, p.v1, p.w1, p.u2, p.v2, p.w2,
                     feed=p.feed)
        if p.kind in ("line", "rapid") \
                and abs(p.u2 - p.u1) < 1e-9 and abs(p.v2 - p.v1) < 1e-9 \
                and abs(p.w2 - p.w1) < 1e-9:
            continue
        if p.kind == "drill" and abs(p.w2 - p.w1) < 1e-9:
            continue
        if out and p.kind in ("line", "drill"):
            prev = out[-1]
            a = np.array((prev.u2 - prev.u1, prev.v2 - prev.v1,
                          prev.w2 - prev.w1))
            b = np.array((p.u2 - p.u1, p.v2 - p.v1, p.w2 - p.w1))
            connected = (
                abs(prev.u2 - p.u1) < 1e-9
                and abs(prev.v2 - p.v1) < 1e-9
                and abs(prev.w2 - p.w1) < 1e-9
            )
            collinear = np.linalg.norm(np.cross(a, b)) \
                <= 1e-9 * max(np.linalg.norm(a) * np.linalg.norm(b), 1.0)
            if (prev.kind == p.kind and prev.feed == p.feed
                    and connected and collinear and float(a @ b) > 0.0):
                out[-1] = Prim(
                    prev.kind, prev.u1, prev.v1, prev.w1,
                    p.u2, p.v2, p.w2, feed=prev.feed,
                )
                continue
        out.append(p)
    return out


@dataclass
class Move:
    """One chord for the voxel sweep, setup-local. (u1,v1)->(u2,v2) at
    tip depth w (deeper end of the chord; w_from = shallower end).
    kind: "cut" | "drill" | "rapid". episode=True marks the first
    cutting chord of a new engagement episode (after a rapid or a feed
    change -- both visible in raw G-code text)."""
    u1: float; v1: float; u2: float; v2: float
    w: float
    feed: float
    kind: str = "cut"
    w_from: float = 0.0
    episode: bool = False
    planned_w: float = 0.0
    duration_s: float = 0.0
    ramp_angle_deg: float = 0.0


def _arc_geometry(p: Prim):
    """Derive (a0, signed sweep) from the QUANTIZED fields -- the only
    source the expander and the interpreter share bit-for-bit. Equal
    endpoints mean a full circle (the G-code convention)."""
    a0 = math.atan2(p.v1 - p.cv, p.u1 - p.cu)
    if abs(p.u2 - p.u1) < 1e-9 and abs(p.v2 - p.v1) < 1e-9:
        return a0, (2 * math.pi if p.ccw else -2 * math.pi)
    a1 = math.atan2(p.v2 - p.cv, p.u2 - p.cu)
    ccw_sweep = (a1 - a0) % (2 * math.pi)
    if ccw_sweep < 1e-12:
        ccw_sweep = 2 * math.pi
    return a0, (ccw_sweep if p.ccw else ccw_sweep - 2 * math.pi)


def _anchored_steps(length: float, max_step: float) -> list:
    """Distances after the start, with full steps anchored at the endpoint.

    Any short remainder is placed at the beginning. Extending a move backward
    through already-cleared space therefore cannot shift the sampling windows
    where its suffix engages material -- the anti-padding invariant used by
    the load model.
    """
    if length <= 1e-12:
        return []
    n_full = int(math.floor(length / max_step + 1e-12))
    rem = length - n_full * max_step
    out = [rem] if rem > 1e-9 else []
    out.extend(rem + max_step * k for k in range(1, n_full + 1))
    if not out or abs(out[-1] - length) > 1e-9:
        out.append(length)
    else:
        out[-1] = length
    return out


def _arc_points(cu, cv, r, a0, sweep, dw=0.0):
    """Sample an arc/helix by sagitta and voxel-scale 3-D chord bounds."""
    if r <= 1e-12:
        return [(cu + r * math.cos(a0 + sweep),
                 cv + r * math.sin(a0 + sweep), 1.0)]
    if r <= SAGITTA_MAX:
        dth_sag = abs(sweep)
    else:
        dth_sag = 2.0 * math.acos(
            max(-1.0, min(1.0, 1.0 - SAGITTA_MAX / r)))
    helix_mm_per_rad = math.hypot(r, dw / max(abs(sweep), 1e-12))
    dth_len = SWEEP_CHORD_MM / helix_mm_per_rad
    max_dth = max(1e-12, min(dth_sag, dth_len))
    sign = 1.0 if sweep >= 0.0 else -1.0
    return [(cu + r * math.cos(a0 + sign * d),
             cv + r * math.sin(a0 + sign * d),
             d / abs(sweep))
            for d in _anchored_steps(abs(sweep), max_dth)]


def _canonical_prim_length(p: Prim) -> float:
    """Historical motion length, independent of load-sweep subdivision.

    Frozen cycle times used sagitta-only arc chords. Keep that metrology
    unchanged while allowing a finer voxel-scale sweep for load safety.
    """
    if p.kind != "arc":
        return math.hypot(
            math.hypot(p.u2 - p.u1, p.v2 - p.v1), p.w2 - p.w1)
    a0, sweep = _arc_geometry(p)
    if p.r <= SAGITTA_MAX:
        n = 1
    else:
        dth = 2.0 * math.acos(
            max(-1.0, min(1.0, 1.0 - SAGITTA_MAX / p.r)))
        n = max(1, int(math.ceil(abs(sweep) / dth)))
    pts = [(p.u1, p.v1, p.w1)]
    for k in range(1, n + 1):
        frac = k / n
        pts.append((
            q(p.cu + p.r * math.cos(a0 + sweep * frac)),
            q(p.cv + p.r * math.sin(a0 + sweep * frac)),
            q(p.w1 + (p.w2 - p.w1) * frac),
        ))
    pts[-1] = (p.u2, p.v2, p.w2)
    return sum(
        math.hypot(math.hypot(b[0] - a[0], b[1] - a[1]), b[2] - a[2])
        for a, b in zip(pts, pts[1:])
    )


def sample_prims(prims: list) -> list:
    """Prims -> chord Moves for the sweep. Deterministic and shared by
    both arms. Feed moves use endpoint-anchored, geometrically bounded 3-D
    chords. Load timing is independently integrated from newly removed voxel
    contact intervals, so chord length cannot dilute or re-phase engagement.
    Slanted chords sweep at their DEEPER end: conservative extra removal is
    bounded by one chord, never below the primitive's target depth. Episodes
    break on rapids AND on feed changes."""
    moves = []
    new_episode = True
    last_feed = None
    for p in prims:
        canonical_duration = (
            60.0 * _canonical_prim_length(p)
            / (RAPID_FEED if p.kind == "rapid" else max(p.feed, 1e-6))
        )
        if p.kind == "rapid":
            moves.append(Move(p.u1, p.v1, p.u2, p.v2,
                              w=max(p.w1, p.w2), feed=RAPID_FEED,
                              kind="rapid", w_from=min(p.w1, p.w2),
                              planned_w=max(p.w1, p.w2),
                              duration_s=canonical_duration))
            new_episode = True
            continue
        if last_feed is not None and abs(p.feed - last_feed) > 1e-9:
            new_episode = True
        last_feed = p.feed
        is_drill = p.kind == "drill" or (
            p.kind == "line"
            and math.hypot(p.u2 - p.u1, p.v2 - p.v1) < 1e-9
        )
        if p.kind in ("line", "drill"):
            du, dv, dw = p.u2 - p.u1, p.v2 - p.v1, p.w2 - p.w1
            planned_xy = math.hypot(du, dv)
            planned_ramp_angle = math.degrees(
                math.atan2(max(0.0, dw), planned_xy))
            length = math.hypot(math.hypot(du, dv), dw)
            ds = _anchored_steps(length, SWEEP_CHORD_MM)
            pts = [(p.u1, p.v1, p.w1, 0.0)]
            pts.extend((p.u1 + du * d / length,
                        p.v1 + dv * d / length,
                        p.w1 + dw * d / length, d) for d in ds)
        else:                                 # arc / helix
            a0, sweep = _arc_geometry(p)
            planned_xy = p.r * abs(sweep)
            planned_ramp_angle = math.degrees(math.atan2(
                max(0.0, p.w2 - p.w1), planned_xy))
            arc_pts = _arc_points(
                p.cu, p.cv, p.r, a0, sweep, p.w2 - p.w1)
            length = math.hypot(p.r * abs(sweep), p.w2 - p.w1)
            pts = [(p.u1, p.v1, p.w1, 0.0)]
            for x, y, frac in arc_pts:
                pts.append((x, y, p.w1 + (p.w2 - p.w1) * frac,
                            length * frac))
            # land exactly on the quantized endpoint (continuity)
            pts[-1] = (p.u2, p.v2, p.w2, length)
        prev = pts[0]
        for cur in pts[1:]:
            if math.hypot(cur[0] - prev[0], cur[1] - prev[1]) < 1e-6 \
                    and abs(cur[2] - prev[2]) < 1e-6:
                continue
            moves.append(Move(q(prev[0]), q(prev[1]), q(cur[0]), q(cur[1]),
                              w=q(max(prev[2], cur[2])),
                              w_from=q(min(prev[2], cur[2])),
                              feed=p.feed, kind="drill" if is_drill else "cut",
                              episode=new_episode,
                              planned_w=max(p.w1, p.w2),
                              duration_s=canonical_duration
                              * (cur[3] - prev[3]) / length,
                              ramp_angle_deg=planned_ramp_angle))
            new_episode = False
            prev = cur
        if is_drill:
            new_episode = True
    return moves


def moves_time_s(moves: list) -> float:
    """Cycle time of a move list: 3D length / feed. No coefficients --
    entries, links, repositions and retracts are real geometry."""
    t = 0.0
    for m in moves:
        if m.duration_s > 0.0:
            t += m.duration_s
        else:
            length = math.hypot(math.hypot(m.u2 - m.u1, m.v2 - m.v1),
                                m.w - m.w_from)
            t += 60.0 * length / max(m.feed, 1e-6)
    return t


# ---------------------------------------------------------------- loops
# Loop builders return (prims, start_point). All milling loops are
# closed and climb-oriented: pocket walls CCW under M3 (material
# radially outward of the path), the standard convention.

def _rounded_rect_prims(cx, cy, w, h, r, wlev, feed):
    r = max(r, 1e-4)
    hw, hh = w / 2 - r, h / 2 - r
    cs = [(cx + hw, cy - hh), (cx + hw, cy + hh),
          (cx - hw, cy + hh), (cx - hw, cy - hh)]
    a = [-math.pi / 2, 0.0, math.pi / 2, math.pi]
    prims = []
    start = (cs[0][0] + r * math.cos(a[0]), cs[0][1] + r * math.sin(a[0]))
    cur = start
    for i in range(4):
        c = cs[i]
        end_ang = a[i] + math.pi / 2
        arc_end = (c[0] + r * math.cos(end_ang),
                   c[1] + r * math.sin(end_ang))
        prims.append(mk("arc", cur[0], cur[1], wlev,
                        arc_end[0], arc_end[1], wlev, feed,
                        cu=c[0], cv=c[1], ccw=True))
        nxt = cs[(i + 1) % 4]
        nxt_pt = (nxt[0] + r * math.cos(end_ang),
                  nxt[1] + r * math.sin(end_ang))
        if math.hypot(nxt_pt[0] - arc_end[0], nxt_pt[1] - arc_end[1]) \
                > 1e-9:
            prims.append(mk("line", arc_end[0], arc_end[1], wlev,
                            nxt_pt[0], nxt_pt[1], wlev, feed))
        cur = nxt_pt
    return prims, start


def _circle_prims(cx, cy, r, wlev, feed):
    """Full CCW circle as two half arcs (unambiguous I/J G-code)."""
    p0 = (cx + r, cy)
    p1 = (cx - r, cy)
    return [mk("arc", p0[0], p0[1], wlev, p1[0], p1[1], wlev, feed,
               cu=cx, cv=cy, ccw=True),
            mk("arc", p1[0], p1[1], wlev, p0[0], p0[1], wlev, feed,
               cu=cx, cv=cy, ccw=True)], p0


def _capsule_prims(u1, v1, u2, v2, r, wlev, feed):
    """Closed capsule boundary (slot) with end-cap radius r, CCW."""
    th = math.atan2(v2 - v1, u2 - u1)
    n = (-math.sin(th), math.cos(th))
    pa = (u1 - n[0] * r, v1 - n[1] * r)     # right of axis at start
    pb = (u2 - n[0] * r, v2 - n[1] * r)
    pc = (u2 + n[0] * r, v2 + n[1] * r)
    pd = (u1 + n[0] * r, v1 + n[1] * r)
    prims = [mk("line", pa[0], pa[1], wlev, pb[0], pb[1], wlev, feed),
             mk("arc", pb[0], pb[1], wlev, pc[0], pc[1], wlev, feed,
                cu=u2, cv=v2, ccw=True),
             mk("line", pc[0], pc[1], wlev, pd[0], pd[1], wlev, feed),
             mk("arc", pd[0], pd[1], wlev, pa[0], pa[1], wlev, feed,
                cu=u1, cv=v1, ccw=True)]
    return prims, pa


def _line_pass_prims(u1, v1, u2, v2, wlev, feed):
    return [mk("line", u1, v1, wlev, u2, v2, wlev, feed)], (u1, v1)


# ---------------------------------------------------------------- entries

def _helix_entry(cu, cv, r_h, w_from, w_to, feed, cleanup_feed):
    """CCW descending helix about (cu,cv): full revolutions with pitch
    set by RAMP_ANGLE on the circumference, then one flat clean-up
    circle at the target level."""
    circ = 2 * math.pi * r_h
    dz_per_rev = circ * math.tan(math.radians(RAMP_ANGLE_DEG))
    n_rev = max(1, int(math.ceil((w_to - w_from) / max(dz_per_rev, 1e-6))))
    start = (cu + r_h, cv)
    prims = []
    w = w_from
    for k in range(n_rev):
        w_next = w_from + (w_to - w_from) * (k + 1) / n_rev
        prims.append(mk("arc", start[0], start[1], w,
                        start[0], start[1], w_next, feed,
                        cu=cu, cv=cv, ccw=True))
        w = w_next
    prims.append(mk("arc", start[0], start[1], w_to,
                    start[0], start[1], w_to, cleanup_feed,
                    cu=cu, cv=cv, ccw=True))
    return prims, start


def _ramp_entry(p1, p2, w_from, w_to, feed):
    """Back-and-forth zigzag ramp along p1->p2 descending to w_to; an
    even leg count returns the tool to p1."""
    L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    dz_per_leg = L * math.tan(math.radians(RAMP_ANGLE_DEG))
    n_leg = max(1, int(math.ceil((w_to - w_from) / max(dz_per_leg, 1e-6))))
    if n_leg % 2 == 1:
        n_leg += 1
    prims = []
    a, b = p1, p2
    w = w_from
    for k in range(n_leg):
        w_next = w_from + (w_to - w_from) * (k + 1) / n_leg
        prims.append(mk("line", a[0], a[1], w, b[0], b[1], w_next, feed))
        a, b = b, a
        w = w_next
    return prims, p1


# ---------------------------------------------------------------- features

def _feature_local(feature: Feature):
    p = feature.params
    if feature.kind == "pocket_rect":
        return dict(kind="rect", cu=p["cx"], cv=p["cy"], w=p["w"],
                    h=p["h"], r=p["corner_r"], depth=p["depth"])
    if feature.kind == "pocket_circ":
        return dict(kind="circ", cu=p["cx"], cv=p["cy"], d=p["d"],
                    depth=p["depth"])
    if feature.kind == "slot":
        return dict(kind="slot", u1=p["x1"], v1=p["y1"], u2=p["x2"],
                    v2=p["y2"], width=p["width"], depth=p["depth"])
    if feature.kind == "hole":
        return dict(kind="hole", cu=p["u"], cv=p["v"], d=p["d"],
                    depth=p["depth"])
    raise ValueError(feature.kind)


def _level_loops(op: Op, g: dict, rt: float, step: float, wlev: float):
    """Loop family for one depth level, outside-in (wall ring first --
    the teacher's honest naive offset policy; a smarter author orders
    center-out to shave peak engagement)."""
    loops = []
    if g["kind"] == "rect":
        inset0 = rt + op.wall_stock
        max_inset = min(g["w"], g["h"]) / 2.0
        if inset0 >= max_inset:
            raise ValueError("tool too big for rect pocket")
        if op.op_type == "pocket_finish":
            # wall pass last; one overlap pass first, but only when the
            # wall ring alone cannot reach the enclosed center (a big
            # tool in a small pocket covers everything from the wall
            # path), and never degenerate (>= 0.5 mm of loop extent)
            insets = [inset0]
            if inset0 + rt < max_inset - 1e-6:
                insets.insert(0, min(inset0 + step, max_inset - 0.5))
        else:
            insets = list(np.arange(inset0, max_inset, step))
            if max_inset - insets[-1] > rt:
                insets.append(max_inset - 1e-3)
        for e in insets:
            rr = max(g["r"] - e, 1e-4)
            loops.append(_rounded_rect_prims(g["cu"], g["cv"],
                                             g["w"] - 2 * e,
                                             g["h"] - 2 * e,
                                             rr, wlev, op.feed))
    elif g["kind"] == "circ":
        r_wall = g["d"] / 2.0 - rt - op.wall_stock
        if r_wall <= 0:
            raise ValueError("tool too big for circular pocket")
        if op.op_type == "pocket_finish":
            # same rule as rect: overlap pass only if the wall circle
            # cannot cover the center, inner-then-wall order
            radii = [r_wall]
            if r_wall > rt + 1e-6:
                radii.insert(0, max(r_wall - step, 0.5))
        else:
            radii = list(np.arange(r_wall, 0.0, -step)) or [r_wall]
            if min(radii) > 0.8 * rt:
                radii.append(0.0)
        for r in radii:
            if r < 1e-6:
                loops.append(_line_pass_prims(g["cu"], g["cv"],
                                              g["cu"] + 1.0, g["cv"],
                                              wlev, op.feed))
            else:
                loops.append(_circle_prims(g["cu"], g["cv"], r, wlev,
                                           op.feed))
    elif g["kind"] == "slot":
        if 2 * rt > g["width"] + 1e-9:
            raise ValueError("tool wider than slot")
        cap0 = g["width"] / 2.0 - rt - op.wall_stock
        if cap0 <= 1e-6:
            loops.append(_line_pass_prims(g["u1"], g["v1"], g["u2"],
                                          g["v2"], wlev, op.feed))
        else:
            caps = list(np.arange(cap0, 0.0, -step))
            if not caps or min(caps) > 0.8 * rt:
                caps.append(0.0)
            for cr in caps:
                if cr <= 1e-6:
                    loops.append(_line_pass_prims(g["u1"], g["v1"],
                                                  g["u2"], g["v2"],
                                                  wlev, op.feed))
                else:
                    loops.append(_capsule_prims(g["u1"], g["v1"],
                                                g["u2"], g["v2"],
                                                cr, wlev, op.feed))
    else:
        raise ValueError(g["kind"])
    return loops


def expand_op(op: Op, feature: Feature) -> list:
    """Expand one op into a CONTINUOUS prim stream that starts and ends
    at the clearance plane (w = CLEAR_W); the plan assembler stitches
    ops with XY hops at clearance. Raises ValueError on geometric
    nonsense; the verifier turns that into an infeasible plan."""
    tool = TOOLS.get(op.tool_id)
    if tool is None:
        raise ValueError(f"unknown tool {op.tool_id}")
    # Spindle speed and feed are controlled machine conditions, not policy
    # actions. Legacy typed plans may still carry either field, but expansion
    # always replaces both from the selected tool's AL6061 schedule.
    fixed = CUT_PARAMS[op.tool_id]
    op = replace(op, rpm=fixed.rpm, feed=fixed.feed)
    g = _feature_local(feature)
    rt = tool.diameter / 2.0

    if op.op_type == "drill":
        if tool.kind != "drill":
            raise ValueError("drill op requires a drill")
        if g["kind"] != "hole":
            raise ValueError("drill op on non-hole feature")
        u, v = g["cu"], g["cv"]
        return [mk("rapid", u, v, CLEAR_W, u, v, -APPROACH_W_GAP,
                   RAPID_FEED),
                mk("drill", u, v, -APPROACH_W_GAP, u, v, g["depth"],
                   op.feed),
                mk("rapid", u, v, g["depth"], u, v, CLEAR_W, RAPID_FEED)]

    if tool.kind != "mill":
        raise ValueError("milling op requires an endmill")
    if g["kind"] == "hole":
        raise ValueError("milling op on a hole feature")

    if op.entry == "helix":
        room = (min(g["w"], g["h"]) if g["kind"] == "rect"
                else g["d"] if g["kind"] == "circ" else g["width"])
        if room < HELIX_ROOM_FACTOR * tool.diameter:
            raise ValueError("no room to helix")
    if op.entry == "plunge" and not tool.center_cutting:
        raise ValueError("plunge with non-center-cutting tool")

    depth = g["depth"]
    n_levels = max(1, int(math.ceil(depth / op.stepdown)))
    levels = [min(depth, (k + 1) * op.stepdown) for k in range(n_levels)]
    step = max(0.2, op.stepover * tool.diameter)
    entry_feed = fq(max(1.0, op.feed * ENTRY_FEED_FRAC))
    plunge_feed = fq(max(1.0, op.feed * PLUNGE_FEED_FRAC))

    prims = []
    pos = None
    w_prev = 0.0
    for w in levels:
        loops = _level_loops(op, g, rt, step, w)
        first_start = loops[0][1]

        if op.entry == "helix":
            if g["kind"] in ("rect", "circ"):
                hc, room = (g["cu"], g["cv"]), \
                    (min(g["w"], g["h"]) if g["kind"] == "rect"
                     else g["d"])
            else:
                hc = ((g["u1"] + g["u2"]) / 2, (g["v1"] + g["v2"]) / 2)
                room = g["width"]
            # helix bore must stay inside the cavity walls: radius
            # capped so bore edge clears wall stock with margin
            r_h = min(HELIX_RADIUS_FRAC * tool.diameter,
                      room / 2.0 - rt - op.wall_stock - 0.05)
            if r_h < 0.05 * tool.diameter:
                raise ValueError("no room to helix")
            ent, ent_pos = _helix_entry(hc[0], hc[1], r_h,
                                        w_prev, w, entry_feed, op.feed)
        elif op.entry == "ramp":
            # ramp along the first loop's longest straight prim; if the
            # loop is all arcs (circles, round-cornered squares), ramp
            # along a chord between two points ON the loop path -- the
            # path already carries the rt + wall_stock margin and every
            # loop is convex, so the chord cannot touch a wall
            seg = None
            for p in loops[0][0]:
                if p.kind == "line":
                    L = math.hypot(p.u2 - p.u1, p.v2 - p.v1)
                    if seg is None or L > seg[0]:
                        seg = (L, (p.u1, p.v1), (p.u2, p.v2))
            if seg is None or seg[0] < 1.0:
                pts = [(p.u1, p.v1) for p in loops[0][0]]
                far = max(pts, key=lambda pt: math.hypot(
                    pt[0] - first_start[0], pt[1] - first_start[1]))
                L = math.hypot(far[0] - first_start[0],
                               far[1] - first_start[1])
                if L < 0.5:
                    raise ValueError("no room to ramp")
                seg = (L, first_start, far)
            ent, ent_pos = _ramp_entry(seg[1], seg[2], w_prev, w,
                                       entry_feed)
        else:   # plunge (center-cutting checked above; with the all-
                # non-center-cutting crib this branch is mill-unreachable)
            ent = [mk("drill", first_start[0], first_start[1], w_prev,
                      first_start[0], first_start[1], w, plunge_feed)]
            ent_pos = first_start

        e0 = ent[0]
        # hop at clearance to the entry XY, then rapid down to just
        # above the previous floor (air: earlier levels already cut)
        if pos is not None and math.hypot(pos[0] - e0.u1,
                                          pos[1] - e0.v1) > 1e-9:
            prims.append(mk("rapid", pos[0], pos[1], CLEAR_W,
                            e0.u1, e0.v1, CLEAR_W, RAPID_FEED))
        approach_w = max(CLEAR_W, w_prev - APPROACH_W_GAP)
        prims.append(mk("rapid", e0.u1, e0.v1, CLEAR_W,
                        e0.u1, e0.v1, approach_w, RAPID_FEED))
        if approach_w < e0.w1 - 1e-9:
            prims.append(mk("line", e0.u1, e0.v1, approach_w,
                            e0.u1, e0.v1, e0.w1, entry_feed))
        prims += ent

        if math.hypot(ent_pos[0] - first_start[0],
                      ent_pos[1] - first_start[1]) > 1e-6:
            prims.append(mk("line", ent_pos[0], ent_pos[1], w,
                            first_start[0], first_start[1], w,
                            op.feed))

        # loop family; G1 links between loops (a rapid hop could clip
        # the unswept sliver ahead of the next ring)
        cur = first_start
        for li, (lp, lstart) in enumerate(loops):
            if li > 0 and math.hypot(cur[0] - lstart[0],
                                     cur[1] - lstart[1]) > 1e-6:
                prims.append(mk("line", cur[0], cur[1], w,
                                lstart[0], lstart[1], w, op.feed))
            prims += lp
            last = lp[-1]
            cur = (last.u2, last.v2)
        prims.append(mk("rapid", cur[0], cur[1], w,
                        cur[0], cur[1], CLEAR_W, RAPID_FEED))
        pos = cur
        w_prev = w

    if not prims:
        raise ValueError("op expanded to no moves")
    return prims


def expand_op_moves(op: Op, feature: Feature) -> list:
    """Sweep-ready chord Moves for one op (entries and rapids included)."""
    return sample_prims(normalize_prims(expand_op(op, feature)))


# ------------------------------------------------------------- program

TOOL_NUMBERS = {tid: i + 1 for i, tid in enumerate(sorted(TOOLS))}
TOOL_BY_NUMBER = {n: tid for tid, n in TOOL_NUMBERS.items()}


@dataclass
class Segment:
    """A run of motion under one (tool, rpm) context."""
    tool_id: str
    rpm: float
    prims: list


@dataclass
class SetupProgram:
    orientation: str
    segments: list          # list[Segment]


def plan_program(plan: Plan, task: Task) -> list:
    """Assemble the canonical, continuous motion program for a typed
    plan: per setup, per op -> one Segment whose stream starts with an
    XY hop at clearance from the previous op's end. Setups begin at
    the local origin at clearance height (the interpreter's convention
    too). This is the single structure the verifier grades, the
    emitter serializes, and the interpreter reconstructs from text."""
    program = []
    for setup in plan.setups:
        segs = []
        pos = (0.0, 0.0)
        for op in setup.ops:
            feat = task.feature(op.feature_ref)
            if feat is None:
                raise ValueError(f"unknown feature {op.feature_ref}")
            prims = expand_op(op, feat)
            e0 = prims[0]
            stream = []
            if math.hypot(pos[0] - e0.u1, pos[1] - e0.v1) > 1e-9:
                stream.append(mk("rapid", pos[0], pos[1], CLEAR_W,
                                 e0.u1, e0.v1, CLEAR_W, RAPID_FEED))
            stream += prims
            stream = normalize_prims(stream)
            last = stream[-1]
            pos = (last.u2, last.v2)
            segs.append(Segment(
                op.tool_id, float(CUT_PARAMS[op.tool_id].rpm), stream))
        program.append(SetupProgram(setup.orientation, segs))
    return program


def program_time_s(program: list) -> float:
    """Wall-clock estimate for a whole program: motion time plus tool
    change and setup change overheads (identical for both arms; the
    verifier recomputes the same quantity during the sweep)."""
    t = 0.0
    last_tool = None
    for si, sp in enumerate(program):
        if si > 0:
            t += SETUP_CHANGE_S
        for seg in sp.segments:
            if seg.tool_id != last_tool:
                t += TOOLCHANGE_S
                last_tool = seg.tool_id
            t += moves_time_s(sample_prims(seg.prims))
    return t


# ------------------------------------------------------------------ G-code

def _f3(x: float) -> str:
    return f"{q(x):.3f}"


def emit_gcode(plan: Plan, task: Task, program: list = None) -> str:
    """Serialize the canonical program as ONE strict-dialect document:
    (SETUP <face>) headers, T<n> M6, M3, then exactly one line
    per motion prim: G0 rapids, G1 lines (Z word on slants and drill
    strokes), G2/G3 arcs (I/J center offsets, Z word on helixes; equal
    or omitted XY = full circle). Z is NEGATIVE below the entered
    face. Spindle speed and feed are omitted because the interpreter injects
    the selected tool's canonical schedule; XYZ is modal within a setup and
    T across the program. cnc/gcode.py parses this text back to the identical
    program (audited exactly)."""
    if program is None:
        program = plan_program(plan, task)
    out = ["(CNC-RLVR PROGRAM)", f"(MATERIAL {task.material})",
           "G21 G90"]
    last_tool = None
    for sp in program:
        out.append(f"(SETUP {sp.orientation})")
        pos = (0.0, 0.0, CLEAR_W)
        for seg in sp.segments:
            if seg.tool_id != last_tool:
                out.append(f"T{TOOL_NUMBERS[seg.tool_id]} M6")
                last_tool = seg.tool_id
            # M3 also delimits normalized segments when consecutive
            # operations reuse one tool.
            out.append("M3")
            for p in seg.prims:
                words = []
                if abs(p.u2 - pos[0]) > 1e-9 or abs(p.v2 - pos[1]) > 1e-9:
                    words.append(f"X{_f3(p.u2)} Y{_f3(p.v2)}")
                if abs(p.w2 - pos[2]) > 1e-9:
                    words.append(f"Z{_f3(-p.w2)}")
                if p.kind == "rapid":
                    if words:
                        out.append("G0 " + " ".join(words))
                elif p.kind in ("line", "drill"):
                    if words:
                        out.append("G1 " + " ".join(words))
                else:       # arc (equal endpoints emit no XY: full circle)
                    gc = "G3" if p.ccw else "G2"
                    ij = f"I{_f3(p.cu - p.u1)} J{_f3(p.cv - p.v1)}"
                    xy = ""
                    if abs(p.u2 - p.u1) > 1e-9 or abs(p.v2 - p.v1) > 1e-9:
                        xy = f"X{_f3(p.u2)} Y{_f3(p.v2)} "
                    zw = f"Z{_f3(-p.w2)} " if abs(p.w2 - p.w1) > 1e-9 \
                        else ""
                    out.append(f"{gc} {xy}{zw}{ij}")
                pos = (p.u2, p.v2, p.w2)
        out.append("M5")
    out.append("M30")
    return "\n".join(out) + "\n"
