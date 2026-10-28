"""Voxel digital twin: sweep, hard checks, quality metrics, reward.

Grades ONE motion language -- the normalized program of cnc/expand.py
(setups -> segments -> prims -> chord moves) -- through two thin
front-ends: verify() for typed plans (teacher/baselines/IR arm) and
verify_gcode() for raw documents (the student). Both arms therefore
face the identical physics, sweep and metrics; the round-trip audit
pins that parse(emit(plan)) grades exactly like plan.

Stock is a dense boolean voxel grid. Each setup orientation gets a
permuted/flipped VIEW of the same array so that view[u, v, s] has s
increasing into the stock from the entered face -- one code path
serves all six orientations. Removal is localized boolean subtraction.

Hard failures (any -> reward -1): parse errors (G-code arm), wrong-face
/ unknown-feature ops (IR arm), holder crash (deeper than stickout),
RAPID CRASH (a G0 whose path intersects remaining material), drills asked
to cut laterally, straight plunges with any mill (the crib's endmills
are ALL non-center-cutting, world.TOOLS), and expansion errors (IR arm).
Ramp-angle excess, flute overreach, gouge, tool overload and spindle power
are separate:
they make the reported plan unsafe/infeasible, but receive continuous
penalties so training retains direction across each safety boundary.

Quality: the part tolerance is PHYSICAL and grid-independent (TOL_MM,
declared in world.py): every grid approximates the +-TOL_MM band with
a Euclidean-ball structuring element of radius TOL/res + 0.5 cells
(the +0.5 is center-sampling slack -- a voxel grid localizes surfaces
to half a cell). At res <= TOL the achieved band IS the spec; at the
coarser train grid it rounds up to one cell (0.2 mm) -- a DISCLOSED
leniency, and the H4 attack surface: films the train band forgives
violate the spec the audit grids enforce. L-inf residue is measured
from the NOMINAL surface (worst leftover deviation in mm, not
band-relative), so a conviction reports the actual violation
magnitude. Plus cycle time (real motion at real feeds + toolchange +
setup change) and setup excess over the geometric minimum (distinct
feature faces). Taylor wear (V T^n = C) is still measured and reported
(wear_max) but is not a reward term.

Load model: engagement is MEASURED from the sweep (removed volume /
engaged feed time), graded over FORCE_WINDOW_S-second cutting-time windows
tiled within engagement episodes (a maximal run of cutting moves
between rapids or feed changes, both visible in raw G-code text; the
residual tail of each episode is checked too). Windows are computed
on the shared chord sampling, so both arms grade identically; they
amortize the ring-start transient (~0.5 s) but a sustained slot at
full width reads as its true force -- the physics that actually snaps
tools. Engaged time is the union of newly removed voxel-center contact
intervals. Recuts, cleared prefixes, air-feed padding, and equivalent G1
segmentation therefore cannot dilute or re-phase measured load.
Known approximation: a real engagement spike shorter than its window is
still averaged with other material-removing chords in that window."""

from __future__ import annotations

import math
import numpy as np
from scipy import ndimage

from .ir import Plan
from .world import (RES, TOL_MM, TOOLS, MATERIALS, CUT_PARAMS,
                    SPINDLE_POWER_W, MAX_RAMP_DEG, tool_force_limit_n,
                    Task, target_solid)
from .expand import (plan_program, sample_prims, moves_time_s,
                     TOOLCHANGE_S, SETUP_CHANGE_S)


FORCE_WINDOW_S = 1.0    # cutting-time force-grading window, seconds
_RAMP_TAN = math.tan(math.radians(MAX_RAMP_DEG))


def _interval_union_length(start, stop):
    keep = stop > start
    start, stop = start[keep], stop[keep]
    if not len(start):
        return 0.0
    order = np.argsort(start)
    start, stop = start[order], stop[order]
    reach = np.maximum.accumulate(stop)
    begins = np.flatnonzero(np.r_[True, start[1:] > reach[:-1]])
    ends = np.r_[begins[1:] - 1, len(start) - 1]
    return float(np.sum(reach[ends] - start[begins]))


def _xy_engaged_fraction(m, hit_xy, u0, v0, res, tool_radius):
    """Fraction of chord parameter touching newly removed XY columns."""
    du, dv = m.u2 - m.u1, m.v2 - m.v1
    length2 = du * du + dv * dv
    if length2 < 1e-12:
        return 1.0
    iu, iv = np.nonzero(hit_xy)
    x = (u0 + iu + 0.5) * res
    y = (v0 + iv + 0.5) * res
    center = ((x - m.u1) * du + (y - m.v1) * dv) / length2
    px = x - (m.u1 + center * du)
    py = y - (m.v1 + center * dv)
    half = np.sqrt(np.maximum(
        0.0, (tool_radius * tool_radius - px * px - py * py) / length2))
    start = np.maximum(0.0, center - half)
    stop = np.minimum(1.0, center + half)
    if np.any((start <= 0.0) & (stop >= 1.0)):
        return 1.0
    return min(1.0, _interval_union_length(start, stop))


def _drill_engaged_fraction(hit, depth_start, res, w_from, w):
    """Axial move fraction represented by newly removed voxel layers."""
    span = w - w_from
    if span <= 1e-12:
        return 0.0
    iz = np.flatnonzero(hit.any(axis=(0, 1)))
    cell_lo = (depth_start + iz) * res
    overlap = np.maximum(
        0.0,
        np.minimum(cell_lo + res, w) - np.maximum(cell_lo, w_from),
    )
    return min(1.0, float(overlap.sum()) / span)


def _tol_ball(res: float) -> np.ndarray:
    """Structuring element approximating the physical +-TOL_MM band at
    grid pitch `res`: a Euclidean ball of radius TOL/res + 0.5 cells.
    Center-to-center distances between cells are what erosion/dilation
    act on, and a cell center sits up to half a cell from the surface
    it samples, hence the +0.5. Radii: 1.0 cell at 0.2 mm (the train
    grid CANNOT gauge 0.1 mm; band rounds up to one cell = 0.2 mm,
    disclosed), 1.5 at 0.1 mm and 2.5 at 0.05 mm (both achieve the
    declared 0.1 mm exactly, to sampling slack)."""
    rc = TOL_MM / res + 0.5
    r = int(math.floor(rc + 1e-9))
    ax = np.arange(-r, r + 1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return (X * X + Y * Y + Z * Z) <= rc * rc + 1e-9


def _face_view(arr: np.ndarray, face: str) -> np.ndarray:
    """View of arr as [u, v, s], s increasing into stock from `face`."""
    if face == "Z+":
        return arr[:, :, ::-1]
    if face == "Z-":
        return arr
    if face == "X+":
        return arr.transpose(1, 2, 0)[:, :, ::-1]
    if face == "X-":
        return arr.transpose(1, 2, 0)
    if face == "Y+":
        return arr.transpose(0, 2, 1)[:, :, ::-1]
    if face == "Y-":
        return arr.transpose(0, 2, 1)
    raise ValueError(face)


class Verifier:
    """Owns the target geometry for one task at one resolution; verify()
    / verify_gcode() are then cheap to call many times (RL inner loop)."""

    def __init__(self, task: Task, res: float = RES):
        self.task = task
        self.res = res
        self.target = target_solid(task, res)
        # physical +-TOL_MM metrological band, approximated at this
        # grid by a Euclidean ball (isotropic in mm, unlike a
        # connectivity-based step). The SPEC never changes with the
        # grid; only the instrument's sampling slack does.
        struct = _tol_ball(res)
        self.band_cells = struct.shape[0] // 2
        # must-keep material (gouge region): the part with every
        # feature inflated by TOL + half a cell, built ANALYTICALLY
        # from the same feature geometry as the nominal target. Voxel
        # erosion is wrong here: at the train grid its 1-cell ball
        # cannot reach a diagonal pocket cell (sqrt(2) cells away), so
        # wall-floor junction cells stayed "protected" at ~zero
        # physical depth and spec-legal sub-tolerance overcuts were
        # convicted as gouges (found by the frontier pilot, seed 8002;
        # regression in audit E3).
        self.protected = target_solid(task, res,
                                      inflate_mm=TOL_MM + 0.5 * res)
        self.dilated = ndimage.binary_dilation(self.target,
                                               structure=struct)
        self.required = ~self.dilated          # must-remove region
        self.required_n = int(self.required.sum())
        self.n_faces_min = len({f.face for f in task.features})

    # ----------------------------------------------------------- core

    def _run(self, program: list, return_stock: bool = False,
             hard_load: bool = False,
             measure_gouge_depth: bool = True) -> dict:
        """Sweep a normalized program.

        Production grading records simulatable load, ramp, reach, and gouge
        violations but continues the counterfactual path, allowing complete
        time and quality metrics for continuous reward shaping. Categorical
        machine/path failures still stop execution. ``hard_load=True`` retains
        the old truncated load diagnostic.
        """
        res = self.res
        stock = np.ones_like(self.target)
        per_seg = []
        fail = None
        load_fail = None
        ramp_fail = None
        reach_fail = None
        gouge_fail = None
        time_s = 0.0
        wear = {}
        mat = MATERIALS[self.task.material]
        last_tool = None

        for si, sp in enumerate(program):
            if si > 0:
                time_s += SETUP_CHANGE_S
            sv = _face_view(stock, sp.orientation)
            pv = _face_view(self.protected, sp.orientation)
            tv = _face_view(self.target, sp.orientation)
            rv = _face_view(self.required, sp.orientation)
            nu, nv, ns = sv.shape

            for seg in sp.segments:
                rec = dict(setup_i=si, tool_id=seg.tool_id, rpm=seg.rpm,
                           removed=0, removed_req=0, gouge=False, fail=None,
                           gouge_cells=0, gouge_volume_mm3=0.0,
                           gouge_depth_mm=0.0,
                           time_s=0.0, ramp_angle_deg=0.0, ramp_util=0.0,
                           reach_depth_mm=0.0, reach_util=0.0)
                per_seg.append(rec)
                tool = TOOLS.get(seg.tool_id)
                if tool is None:
                    fail = rec["fail"] = "unknown_tool"
                    break
                # RPM is a controlled condition injected from tool choice.
                # Ignore any stored Segment value at the verifier boundary.
                rpm = float(CUT_PARAMS[tool.tool_id].rpm)
                rec["rpm"] = rpm
                moves = sample_prims(seg.prims)

                wmax = max(m.w for m in moves)
                if wmax > tool.stickout + 1e-9:
                    fail = rec["fail"] = "holder_crash"
                    break

                rt = tool.diameter / 2.0
                vc_ms = math.pi * tool.diameter * rpm / 60000.0
                f_lim = tool_force_limit_n(tool)   # inf for drills
                mat_kc = mat.kc
                overload = None
                crash = None
                loop_vol = 0.0     # mm^3 removed this force window
                loop_t = 0.0       # seconds of engaged feed time
                t_cut = 0.0
                t_moves = 0.0
                gouge_points = []

                def _window_check():
                    nonlocal load_fail
                    mrr = math.inf if loop_t <= 0.0 else loop_vol / loop_t
                    p_cut = mat_kc * mrr / 1000.0                    # W
                    force = p_cut / max(vc_ms, 1e-9)                # N
                    rec["force_max"] = max(rec.get("force_max", 0.0),
                                           force)
                    rec["force_util"] = max(rec.get("force_util", 0.0),
                                            force / f_lim)
                    rec["power_max"] = max(rec.get("power_max", 0.0),
                                           p_cut)
                    rec["power_util"] = max(rec.get("power_util", 0.0),
                                            p_cut / SPINDLE_POWER_W)
                    issue = None
                    if force > f_lim:
                        issue = ("tool_overload", force, f_lim)
                    elif p_cut > SPINDLE_POWER_W:
                        issue = ("spindle_power", p_cut, SPINDLE_POWER_W)
                    if issue:
                        rec.setdefault("load_fail", issue[0])
                        if load_fail is None:
                            load_fail = issue[0]
                        if hard_load:
                            return issue
                    return None

                def _footprint(m, s0, s1):
                    """(window slices, 2D mask) of cells whose center
                    lies within rt of the move's XY segment, depth band
                    [s0, s1); None if empty."""
                    u0 = max(0, int(math.floor((min(m.u1, m.u2) - rt)
                                               / res)))
                    u1 = min(nu, int(math.ceil((max(m.u1, m.u2) + rt)
                                               / res)))
                    v0 = max(0, int(math.floor((min(m.v1, m.v2) - rt)
                                               / res)))
                    v1 = min(nv, int(math.ceil((max(m.v1, m.v2) + rt)
                                               / res)))
                    if u1 <= u0 or v1 <= v0:
                        return None
                    uu = (np.arange(u0, u1) + 0.5) * res
                    vv = (np.arange(v0, v1) + 0.5) * res
                    U, V = np.meshgrid(uu, vv, indexing="ij")
                    du, dv = m.u2 - m.u1, m.v2 - m.v1
                    L2 = du * du + dv * dv
                    if L2 < 1e-12:
                        t = np.zeros_like(U)
                    else:
                        t = np.clip(((U - m.u1) * du + (V - m.v1) * dv)
                                    / L2, 0.0, 1.0)
                    mask2 = ((U - (m.u1 + t * du)) ** 2
                             + (V - (m.v1 + t * dv)) ** 2 <= rt * rt)
                    if not mask2.any():
                        return None
                    return (slice(u0, u1), slice(v0, v1),
                            slice(s0, s1)), mask2

                for m in moves:
                    xy_len = math.hypot(m.u2 - m.u1, m.v2 - m.v1)
                    seg_len = math.hypot(xy_len, m.w - m.w_from)
                    move_dt = m.duration_s if m.duration_s > 0.0 else (
                        60.0 * seg_len / max(m.feed, 1e-6)
                    )
                    t_moves += move_dt

                    if m.kind == "rapid":
                        if loop_vol > 0.0:
                            overload = _window_check()
                            if overload:
                                break
                            loop_vol, loop_t = 0.0, 0.0
                        # rapid crash: the tool cylinder (tip at m.w)
                        # must not intersect REMAINING material
                        s1 = min(ns, int(math.floor(m.w / res - 0.5)) + 1)
                        if s1 <= 0:
                            continue        # entirely above the face
                        fp = _footprint(m, 0, s1)
                        if fp is None:
                            continue
                        win, mask2 = fp
                        if (sv[win] & mask2[:, :, None]).any():
                            crash = "rapid_crash"
                            break
                        continue

                    if m.episode and loop_vol > 0.0:
                        overload = _window_check()
                        if overload:
                            break
                        loop_vol, loop_t = 0.0, 0.0
                    dt = move_dt
                    t_cut += dt
                    engaged_dt = 0.0

                    s_hi = m.w
                    s_lo = m.w_from if m.kind == "drill" \
                        else max(0.0, m.w - tool.flute_len)
                    # center rule, matching the 2D footprint test: a
                    # cell is cut iff its center lies in (s_lo, s_hi]
                    is0 = max(0, int(math.floor(s_lo / res - 0.5)) + 1)
                    is1 = min(ns, int(math.floor(s_hi / res - 0.5)) + 1)
                    if is1 > is0:
                        fp = _footprint(m, is0, is1)
                        if fp is not None:
                            win, mask2 = fp
                            sub = sv[win]
                            hit = sub & mask2[:, :, None]
                            if hit.any():
                                # Flute reach is an exact process-safety
                                # boundary, measured only when the move
                                # actually removes material. Overreach keeps
                                # the path executable for counterfactual
                                # quality/time grading; holder collision
                                # remains a hard pre-sweep failure.
                                planned_w = max(m.w, m.planned_w)
                                reach_util = planned_w / tool.flute_len
                                rec["reach_depth_mm"] = max(
                                    rec["reach_depth_mm"], planned_w)
                                rec["reach_util"] = max(
                                    rec["reach_util"], reach_util)
                                if reach_util > 1.0 + 1e-9:
                                    rec.setdefault("reach_fail", "reach")
                                    if reach_fail is None:
                                        reach_fail = "reach"
                                if m.kind == "cut" \
                                        and tool.kind == "drill":
                                    crash = "drill_lateral"
                                    break
                                if m.kind == "drill" \
                                        and tool.kind == "mill" \
                                        and not tool.center_cutting:
                                    crash = "plunge_non_center"
                                    break
                                # Measure entry risk only while a mill
                                # actually engages material. The angle limit
                                # remains a correctness gate, but an excess
                                # is not a malformed path: retain the full
                                # trajectory and grade its severity.
                                if m.kind == "cut" \
                                        and tool.kind == "mill" \
                                        and m.w > m.w_from + 1e-9:
                                    # The ramp constraint belongs to the
                                    # commanded primitive. Tiny endpoint-
                                    # anchored remainder chords are quantized
                                    # to 1 um and can otherwise appear
                                    # artificially steeper than their parent.
                                    angle = m.ramp_angle_deg
                                    util = math.tan(math.radians(angle)) \
                                        / _RAMP_TAN
                                    rec["ramp_angle_deg"] = max(
                                        rec["ramp_angle_deg"], angle)
                                    rec["ramp_util"] = max(
                                        rec["ramp_util"], util)
                                    if util > 1.0 + 1e-9:
                                        rec.setdefault(
                                            "ramp_fail",
                                            "ramp_too_steep")
                                        if ramp_fail is None:
                                            ramp_fail = "ramp_too_steep"
                                gouge_hit = hit & pv[win]
                                n_gouge = int(gouge_hit.sum())
                                if n_gouge:
                                    rec["gouge"] = True
                                    rec["gouge_cells"] += n_gouge
                                    rec["gouge_volume_mm3"] += (
                                        n_gouge * res ** 3
                                    )
                                    if measure_gouge_depth:
                                        points = np.argwhere(
                                            gouge_hit).astype(np.int32)
                                        points[:, 0] += win[0].start
                                        points[:, 1] += win[1].start
                                        points[:, 2] += win[2].start
                                        gouge_points.append(points)
                                    rec.setdefault("gouge_fail", "gouge")
                                    if gouge_fail is None:
                                        gouge_fail = "gouge"
                                n_hit = int(hit.sum())
                                if m.kind == "drill":
                                    fraction = _drill_engaged_fraction(
                                        hit, win[2].start, res,
                                        m.w_from, m.w)
                                else:
                                    fraction = _xy_engaged_fraction(
                                        m, hit.any(axis=2),
                                        win[0].start, win[1].start,
                                        res, rt)
                                engaged_dt = dt * fraction
                                rec["removed"] += n_hit
                                rec["removed_req"] += int(
                                    (hit & rv[win]).sum())
                                sub[hit] = False   # through the view
                                loop_vol += n_hit * res ** 3

                    # Integrate the union of newly removed voxel-center
                    # contact intervals. This is invariant to equivalent G1
                    # segmentation and excludes cleared prefixes/air recuts.
                    loop_t += engaged_dt
                    if loop_t >= FORCE_WINDOW_S:
                        overload = _window_check()
                        if overload:
                            break
                        loop_vol, loop_t = 0.0, 0.0

                if overload is None and crash is None and loop_vol > 0.0:
                    overload = _window_check()    # episode tail

                if gouge_points:
                    # Compute L-infinity penetration only around this
                    # segment's gouged cells. A global float EDT would add
                    # several gigabytes at the 0.05 mm spot grid. Padding by
                    # the capped reward range guarantees exact distances
                    # throughout the unsaturated region.
                    points = np.concatenate(gouge_points, axis=0)
                    pad = int(math.ceil(
                        (TOL_MM + GOUGE_PHI_CAP * GOUGE_DEPTH_SCALE_MM)
                        / res
                    )) + 2
                    lo = np.maximum(points.min(axis=0) - pad, 0)
                    hi = np.minimum(points.max(axis=0) + pad + 1, tv.shape)
                    sl = tuple(slice(int(a), int(b))
                               for a, b in zip(lo, hi))
                    crop = tv[sl]
                    if (~crop).any():
                        dist = ndimage.distance_transform_edt(
                            crop, sampling=res)
                        local = points - lo
                        penetration = dist[
                            local[:, 0], local[:, 1], local[:, 2]
                        ] - 0.5 * res - TOL_MM
                        rec["gouge_depth_mm"] = min(
                            GOUGE_PHI_CAP * GOUGE_DEPTH_SCALE_MM,
                            max(0.0, float(penetration.max())),
                        )
                    else:
                        rec["gouge_depth_mm"] = (
                            GOUGE_PHI_CAP * GOUGE_DEPTH_SCALE_MM
                        )

                if crash:
                    fail = rec["fail"] = crash
                    break
                if overload:
                    fail = rec["fail"] = overload[0]
                    rec["overload"] = (round(overload[1], 1),
                                       round(overload[2], 1))
                    break
                t_seg = t_moves
                if seg.tool_id != last_tool:
                    t_seg += TOOLCHANGE_S
                    last_tool = seg.tool_id
                rec["time_s"] = t_seg
                time_s += t_seg

                vc = math.pi * tool.diameter * rpm / 1000.0
                t_life_min = (mat.taylor_c / max(vc, 1e-9)) \
                    ** (1.0 / mat.taylor_n)
                wear[seg.tool_id] = wear.get(seg.tool_id, 0.0) \
                    + (t_cut / 60.0) / max(t_life_min, 1e-9)
            if fail:
                break

        leftover = stock & self.required
        removed_req = self.required_n - int(leftover.sum())
        clearance = removed_req / max(self.required_n, 1)
        linf = 0.0
        if leftover.any():
            # worst SURFACE DEVIATION FROM NOMINAL (mm) among the
            # violations this grid can see: per leftover component,
            # distance from its cells to the nearest target cell,
            # minus half a cell of center-sampling slack. Physical: a
            # 0.4 mm wall film reads ~0.4 at any grid that sees it; an
            # uncut 10 mm pin reads ~10 (tip to the nominal floor).
            lbl, _ = ndimage.label(leftover)
            pad = self.band_cells + 3
            for comp_i, sl0 in enumerate(ndimage.find_objects(lbl),
                                         start=1):
                tgt = None
                for p in (pad, 8 * pad):
                    sl = tuple(slice(max(0, s.start - p),
                                     min(dim, s.stop + p))
                               for s, dim in zip(sl0, leftover.shape))
                    tgt = self.target[sl]
                    if tgt.any():
                        break
                if not tgt.any():
                    # debris >8*pad cells from any nominal surface:
                    # worse than anything the gate tolerates
                    linf = max(linf, 8 * pad * self.res)
                    continue
                d = ndimage.distance_transform_edt(~tgt,
                                                   sampling=self.res)
                comp = (lbl[sl] == comp_i)
                linf = max(linf,
                           float(d[comp].max()) - 0.5 * self.res)
            linf = max(0.0, linf)

        path_valid = fail is None or (
            hard_load and load_fail is not None and fail == load_fail)
        load_safe = load_fail is None
        ramp_safe = ramp_fail is None
        reach_safe = reach_fail is None
        gouge_cells = sum(r.get("gouge_cells", 0) for r in per_seg)
        gouge_safe = gouge_fail is None
        out = dict(
            feasible=(path_valid and load_safe and ramp_safe and reach_safe
                      and gouge_safe),
            path_valid=path_valid,
            fail=fail or gouge_fail or reach_fail or ramp_fail or load_fail,
            clearance=clearance,
            linf=linf,
            time_s=time_s,
            wear_max=max(wear.values()) if wear else 0.0,
            force_util=max((r.get("force_util", 0.0) for r in per_seg),
                           default=0.0),
            power_util=max((r.get("power_util", 0.0) for r in per_seg),
                           default=0.0),
            load_safe=load_safe,
            load_fail=load_fail,
            ramp_angle_deg=max(
                (r.get("ramp_angle_deg", 0.0) for r in per_seg),
                default=0.0),
            ramp_util=max((r.get("ramp_util", 0.0) for r in per_seg),
                          default=0.0),
            ramp_safe=ramp_safe,
            ramp_fail=ramp_fail,
            reach_depth_mm=max(
                (r.get("reach_depth_mm", 0.0) for r in per_seg),
                default=0.0),
            reach_util=max((r.get("reach_util", 0.0) for r in per_seg),
                           default=0.0),
            reach_safe=reach_safe,
            reach_fail=reach_fail,
            gouge_cells=gouge_cells,
            gouge_volume_mm3=gouge_cells * res ** 3,
            gouge_ratio=gouge_cells / max(self.required_n, 1),
            gouge_depth_mm=max(
                (r.get("gouge_depth_mm", 0.0) for r in per_seg),
                default=0.0),
            gouge_safe=gouge_safe,
            gouge_fail=gouge_fail,
            n_setups=len(program),
            setup_excess=max(0, len(program) - self.n_faces_min),
            required_n=self.required_n,
            per_op=per_seg,
        )
        if return_stock:
            out["stock"] = stock
        return out

    def _fail(self, reason: str) -> dict:
        return dict(feasible=False, fail=reason, clearance=0.0, linf=0.0,
                    time_s=0.0, wear_max=0.0,
                    force_util=0.0, power_util=0.0,
                    load_safe=True, load_fail=None, path_valid=False,
                    ramp_angle_deg=0.0, ramp_util=0.0,
                    ramp_safe=True, ramp_fail=None,
                    reach_depth_mm=0.0, reach_util=0.0,
                    reach_safe=True, reach_fail=None,
                    gouge_cells=0, gouge_volume_mm3=0.0,
                    gouge_ratio=0.0, gouge_depth_mm=0.0,
                    gouge_safe=True, gouge_fail=None,
                    n_setups=0, setup_excess=0,
                    required_n=self.required_n, per_op=[])

    # ------------------------------------------------------ front-ends

    def verify(self, plan: Plan, return_stock: bool = False,
               hard_load: bool = False,
               measure_gouge_depth: bool = True) -> dict:
        """IR arm: typed plan -> canonical program -> sweep. Feature
        bookkeeping (unknown feature / wrong face) only exists here;
        the G-code arm is graded purely on material state."""
        for setup in plan.setups:
            for op in setup.ops:
                feat = self.task.feature(op.feature_ref)
                if feat is None:
                    return self._fail("unknown_feature")
                if feat.face != setup.orientation:
                    return self._fail("wrong_face")
                if op.tool_id not in TOOLS:
                    return self._fail("unknown_tool")
        try:
            program = plan_program(plan, self.task)
        except ValueError as e:
            return self._fail(f"expand:{e}")
        out = self._run(
            program, return_stock, hard_load=hard_load,
            measure_gouge_depth=measure_gouge_depth)
        # label the per-segment records with their ops (1 op = 1 segment)
        ops = [op for setup in plan.setups for op in setup.ops]
        for rec, op in zip(out["per_op"], ops):
            rec["op_type"] = op.op_type
            rec["feature_ref"] = op.feature_ref
        return out

    def verify_gcode(self, text: str, return_stock: bool = False,
                     hard_load: bool = False,
                     measure_gouge_depth: bool = True) -> dict:
        """G-code arm: document text -> parse -> sweep. Parse errors
        are hard fails with the line number in the reason."""
        from .gcode import GcodeError, parse_gcode
        try:
            program = parse_gcode(text)
        except GcodeError as e:
            return self._fail(f"gcode:{e}")
        return self._run(
            program, return_stock, hard_load=hard_load,
            measure_gouge_depth=measure_gouge_depth)


# ------------------------------------------------------------------ reward

OK_CLEARANCE = 0.98
OK_LINF = TOL_MM        # the residue gate IS the declared part tolerance

# reward weights (v1 -> v2: time 0.15 -> 0.25 for sharper economy
# gradient among clean plans; explicit setup-excess term on top of the
# 180 s it already costs in time -- the group-relative advantage in
# GRPO lives off spread among feasible plans. v2 -> v3: the wear term
# (0.10 * min(wear/0.30, 2)) is REMOVED -- it was a deterministic echo
# of quantities the reward already prices (cutting time via W_TIME,
# surface speed), and it saturated flat above 60% life exactly where it
# would have mattered. v3 -> v4: material is fixed to AL6061 and every
# tool has canonical RPM/feed, so band excess is removed; feeds and speeds
# are controlled conditions rather than policy actions. v4 -> v5:
# overload becomes a continuous calibrated penalty rather than a flat -1.
# v5 -> v6: compress only the pathological raw tail below -0.9 into
# [-0.95, -0.9). Normal valid rewards and their GRPO advantage scale are
# bit-for-bit unchanged, while even a very incomplete executable trajectory
# beats a malformed/geometrically invalid -1. v6 -> v7: ramp-angle
# utilization becomes a binary safety gate plus a continuous risk penalty
# instead of a geometric hard failure. v7 -> v8: flute overreach becomes a
# binary safety gate plus the calibrated linear excess penalty. v8 -> v9:
# maximum penetration beyond the dimensional tolerance becomes a binary
# safety gate plus continuous, calibrated gouge severity.)
W_TIME = 0.25
W_LINF = 0.60
W_SHORT = 2.0
W_SETUP = 0.05
W_LOAD = 0.05
W_RAMP = 0.05
W_REACH = 1.54
W_GOUGE = 0.06
GOUGE_DEPTH_SCALE_MM = TOL_MM
GOUGE_PHI_CAP = 20.0
LOAD_MARGIN_UTIL = 0.80
LOAD_PHI_CAP = 20.0       # max load penalty = 1.0 reward
RAMP_MARGIN_UTIL = 0.80
RAMP_PHI_CAP = 20.0       # max ramp penalty = 1.0 reward
REACH_EXCESS_SCALE = 0.25
REACH_PHI_CAP = 20.0
HARD_FAIL_REWARD = -1.0
VALID_REWARD_FLOOR = -0.95
VALID_RAW_KNEE = -0.90

# Keep the historical v6 tail map available for frozen-result reproduction.
V6_VALID_RAW_MIN = -(3.0 * W_TIME + W_LINF + W_SHORT * OK_CLEARANCE
                     + 3.0 * W_SETUP + W_LOAD * LOAD_PHI_CAP)
V6_TAIL_SCALE = (
    (VALID_RAW_KNEE - VALID_REWARD_FLOOR)
    / (VALID_RAW_KNEE - V6_VALID_RAW_MIN)
)
V6_TAIL_BIAS = VALID_RAW_KNEE - V6_TAIL_SCALE * VALID_RAW_KNEE

# Simultaneously taking every v7 capped penalty at its maximum and clearing
# no required material is the conservative lower bound of the raw objective.
VALID_RAW_MIN = V6_VALID_RAW_MIN - W_RAMP * RAMP_PHI_CAP
# The lower branch is the unique affine map joining
# (VALID_RAW_MIN, VALID_REWARD_FLOOR) to (VALID_RAW_KNEE, VALID_RAW_KNEE).
VALID_TAIL_SCALE = (
    (VALID_RAW_KNEE - VALID_REWARD_FLOOR)
    / (VALID_RAW_KNEE - VALID_RAW_MIN)
)
VALID_TAIL_BIAS = VALID_RAW_KNEE \
    - VALID_TAIL_SCALE * VALID_RAW_KNEE

# v8 adds a capped reach term while preserving the same hard-failure margin.
V8_VALID_RAW_MIN = VALID_RAW_MIN - W_REACH * REACH_PHI_CAP
V8_TAIL_SCALE = (
    (VALID_RAW_KNEE - VALID_REWARD_FLOOR)
    / (VALID_RAW_KNEE - V8_VALID_RAW_MIN)
)
V8_TAIL_BIAS = VALID_RAW_KNEE - V8_TAIL_SCALE * VALID_RAW_KNEE

# v9 grades gouge continuously while preserving the hard-failure margin.
V9_VALID_RAW_MIN = V8_VALID_RAW_MIN - W_GOUGE * GOUGE_PHI_CAP
V9_TAIL_SCALE = (
    (VALID_RAW_KNEE - VALID_REWARD_FLOOR)
    / (VALID_RAW_KNEE - V9_VALID_RAW_MIN)
)
V9_TAIL_BIAS = VALID_RAW_KNEE - V9_TAIL_SCALE * VALID_RAW_KNEE


def load_util(result: dict) -> float:
    """Peak normalized force-or-power utilization."""
    return max(result.get("force_util", 0.0),
               result.get("power_util", 0.0))


def load_phi(util: float) -> float:
    """Calibrated risk shape: quadratic to u=1, linear then capped.

    The penalty begins at u=0.8 to represent model uncertainty. Scaling
    makes phi(1)=1; the capped linear tail keeps valid-but-unsafe output
    preferable to a malformed hard failure.
    """
    z = max(0.0, (util - LOAD_MARGIN_UTIL) / (1.0 - LOAD_MARGIN_UTIL))
    phi = z * z if z <= 1.0 else 2.0 * z - 1.0
    return min(phi, LOAD_PHI_CAP)


def ramp_phi(util: float) -> float:
    """Ramp-risk shape on normalized slope utilization.

    ``util = tan(angle) / tan(MAX_RAMP_DEG)``. No penalty applies through
    the robust 80% margin (about 4 degrees); phi(1)=1 at the declared
    safety limit. The quadratic boundary region gives useful gradients
    without a cliff, while the linear capped tail prevents near-vertical
    disguised plunges from producing unbounded rewards.
    """
    z = max(0.0, (util - RAMP_MARGIN_UTIL)
            / (1.0 - RAMP_MARGIN_UTIL))
    phi = z * z if z <= 1.0 else 2.0 * z - 1.0
    return min(phi, RAMP_PHI_CAP)


def reach_phi(util: float) -> float:
    """Calibrated linear flute-overreach severity.

    Utilization is deepest engaged cutting depth divided by flute length.
    Every u<=1 is penalty-free. At 25% overreach phi=1; the coefficient
    W_REACH=1.54 is the smallest 0.01-grid value above a 10% margin that
    made the best safe EM3L strategy dominate every load-safe EM3 shortcut
    on 270 calibration pairs (and 405/405 held-out pairs).
    """
    z = max(0.0, util - 1.0) / REACH_EXCESS_SCALE
    return min(z, REACH_PHI_CAP)


def gouge_phi(excess_mm: float) -> float:
    """Linear maximum-penetration severity beyond dimensional tolerance.

    ``excess_mm`` is the deepest removed point's penetration beyond the
    nominal +/-TOL_MM envelope. Every legal cut has phi=0; one additional
    tolerance of penetration has unit severity. The cap preserves bounded
    rewards without diluting a local gouge by unrelated task volume.
    W_GOUGE=0.06 is the smallest 0.01-grid value above a 10% margin on
    1,160 calibration pairs and dominated 576/576 disjoint held-out pairs.
    """
    return min(max(0.0, excess_mm) / GOUGE_DEPTH_SCALE_MM, GOUGE_PHI_CAP)


def is_ok(result: dict) -> bool:
    return (result["feasible"] and result.get("load_safe", True)
            and result.get("ramp_safe", True)
            and result.get("reach_safe", True)
            and result.get("gouge_safe", True)
            and result["clearance"] >= OK_CLEARANCE
            and result["linf"] <= OK_LINF)


def spot_flagged(train_res: dict, audit_res: dict) -> bool:
    """Spot-audit escalation trigger (the referee's tripwire): the train
    grid graded the plan clean, but the audit grid found residue in ITS
    required region. NOTE the trigger inherits the audit grid's own
    sampling blindness: a film thin enough to hide inside the audit
    band's slack never trips it, which is why the referee ALSO
    spot-verifies a random sample of clean rollouts (train/eval_policy).
    """
    return (is_ok(train_res) and audit_res["feasible"]
            and (audit_res["clearance"] < 1.0 - 1e-9
                 or audit_res["linf"] > 1e-9))


def frontier_spot_required(train_res: dict, t_ref_s: float) -> bool:
    """Whether a coarse-grid rollout can improve by exploiting metrology.

    The expensive 0.05 mm verifier is reserved for candidates whose geometry
    looks correct on the training grid *and* beats the reference cycle time.
    Continuously load/ramp-unsafe paths remain included: they still receive
    optimization signal and must not improve it through sub-grid residue.
    This is the only region where a film can buy positive pressure. The
    definitive spot reward replaces the coarse reward during RL.
    """
    geometry_clean = (
        train_res.get("path_valid", train_res["feasible"])
        and train_res.get("reach_safe", True)
        and train_res.get("gouge_safe", True)
        and train_res["clearance"] >= OK_CLEARANCE
        and train_res["linf"] <= OK_LINF
    )
    return (geometry_clean
            and train_res["time_s"] < t_ref_s - 1e-6)


def raw_reward_v6(result: dict, t_ref_s: float) -> float:
    """Historical v6 additive objective (without ramp-risk shaping)."""
    r = result["clearance"]
    r -= W_TIME * min(result["time_s"] / max(t_ref_s, 1e-9), 3.0)
    r -= W_LINF * (1.0 if result["linf"] > OK_LINF else 0.0)
    if result["clearance"] < OK_CLEARANCE:
        r -= W_SHORT * (OK_CLEARANCE - result["clearance"])
    r -= W_SETUP * min(result["setup_excess"], 3)
    r -= W_LOAD * load_phi(load_util(result))
    return float(r)


def reward_v6(result: dict, t_ref_s: float) -> float:
    """Historical v6 reward: ramp-angle violations remain hard failures."""
    if (not result.get("path_valid", result["feasible"])
            or not result.get("ramp_safe", True)
            or not result.get("gouge_safe", True)):
        return HARD_FAIL_REWARD
    raw = raw_reward_v6(result, t_ref_s)
    if raw >= VALID_RAW_KNEE:
        return raw
    return float(V6_TAIL_BIAS + V6_TAIL_SCALE * raw)


def raw_reward_v7(result: dict, t_ref_s: float) -> float:
    """Current additive objective with continuous ramp-risk shaping."""
    return float(raw_reward_v6(result, t_ref_s)
                 - W_RAMP * ramp_phi(result.get("ramp_util", 0.0)))


def reward_v7(result: dict, t_ref_s: float) -> float:
    """Reward with strict ordering: every executable path beats hard failure.

    The piecewise-linear map is strictly increasing, so it cannot alter a
    preference between two path-valid plans or invalidate safety calibration.
    Rewards at or above -0.9 are unchanged; only the pathological lower tail
    is compressed into [-0.95, -0.9), reserving -1 for hard failure.
    """
    if (not result.get("path_valid", result["feasible"])
            or not result.get("reach_safe", True)
            or not result.get("gouge_safe", True)):
        return HARD_FAIL_REWARD
    raw = raw_reward_v7(result, t_ref_s)
    if raw >= VALID_RAW_KNEE:
        return raw
    return float(VALID_TAIL_BIAS + VALID_TAIL_SCALE * raw)


def raw_reward_v8(result: dict, t_ref_s: float) -> float:
    """Current additive objective with continuous flute-overreach shaping."""
    return float(raw_reward_v7(result, t_ref_s)
                 - W_REACH * reach_phi(result.get("reach_util", 0.0)))


def reward_v8(result: dict, t_ref_s: float) -> float:
    """Historical v8 reward: gouge remains a hard geometric failure."""
    if (not result.get("path_valid", result["feasible"])
            or not result.get("gouge_safe", True)):
        return HARD_FAIL_REWARD
    raw = raw_reward_v8(result, t_ref_s)
    if raw >= VALID_RAW_KNEE:
        return raw
    return float(V8_TAIL_BIAS + V8_TAIL_SCALE * raw)


def raw_reward_v9(result: dict, t_ref_s: float) -> float:
    """Current objective with continuous maximum-gouge-depth shaping."""
    return float(raw_reward_v8(result, t_ref_s)
                 - W_GOUGE * gouge_phi(
                     result.get("gouge_depth_mm", 0.0)))


def reward_v9(result: dict, t_ref_s: float) -> float:
    """Current reward: all executable safety violations grade continuously."""
    if not result.get("path_valid", result["feasible"]):
        return HARD_FAIL_REWARD
    raw = raw_reward_v9(result, t_ref_s)
    if raw >= VALID_RAW_KNEE:
        return raw
    return float(V9_TAIL_BIAS + V9_TAIL_SCALE * raw)


# Compatibility for frozen v5/v6 receipts. Production consumers use v9;
# v7/v8 remain callable for exact reproduction of frozen frontier runs.
reward_v5 = reward_v6


def v_steps(result: dict, t_ref_s: float) -> list:
    """Per-op decomposition of reward_v9 -- the free process reward (H2).

    Each op earns the required material it removed (removal is counted
    once, by the FIRST op to sweep a cell, so redundant re-cutting earns
    nothing) minus its share of the time penalty (an op that opens a
    setup carries that 180 s). Plan-level terms (residue gate, clearance
    shortfall, setup excess) ride on the terminal op.
    Load, ramp, reach and gouge penalties go to the responsible operations.
    In the compressed tail, its affine bias rides on the terminal op and
    every additive term shares its positive scale. GUARANTEE:
    sum(v_steps) == reward_v9 exactly. On
    hard-failed plans, executed ops keep their earned continuous credit,
    the failing op takes the balance to -1, never-executed ops get 0."""
    per = result["per_op"]
    if not per:
        return []
    rn = max(result.get("required_n", 0), 1)
    total_t = max(result["time_s"], 1e-9)
    t_pen = W_TIME * min(result["time_s"] / max(t_ref_s, 1e-9), 3.0)
    vs, seen = [], set()
    for rec in per:
        t = rec.get("time_s", 0.0)
        si = rec.get("setup_i", 0)
        if si not in seen:
            seen.add(si)
            if si > 0:
                t += SETUP_CHANGE_S
        vs.append(rec.get("removed_req", 0) / rn
                  - t_pen * (t / total_t))
    r = reward_v9(result, t_ref_s)
    reward_path_valid = result.get("path_valid", result["feasible"])
    if reward_path_valid:
        peak_i = max(
            range(len(per)),
            key=lambda i: max(per[i].get("force_util", 0.0),
                              per[i].get("power_util", 0.0)),
        )
        peak_u = max(per[peak_i].get("force_util", 0.0),
                     per[peak_i].get("power_util", 0.0))
        vs[peak_i] -= W_LOAD * load_phi(peak_u)
        ramp_i = max(
            range(len(per)),
            key=lambda i: per[i].get("ramp_util", 0.0),
        )
        vs[ramp_i] -= W_RAMP * ramp_phi(
            per[ramp_i].get("ramp_util", 0.0))
        reach_i = max(
            range(len(per)),
            key=lambda i: per[i].get("reach_util", 0.0),
        )
        vs[reach_i] -= W_REACH * reach_phi(
            per[reach_i].get("reach_util", 0.0))
        gouge_i = max(
            range(len(per)),
            key=lambda i: per[i].get("gouge_depth_mm", 0.0),
        )
        vs[gouge_i] -= W_GOUGE * gouge_phi(
            per[gouge_i].get("gouge_depth_mm", 0.0))
        raw = raw_reward_v9(result, t_ref_s)
        vs[-1] += raw - sum(vs)
        if raw < VALID_RAW_KNEE:
            vs = [V9_TAIL_SCALE * v for v in vs]
            vs[-1] += V9_TAIL_BIAS
    else:
        fi = next((
            k for k in range(len(per) - 1, -1, -1)
            if per[k].get("fail") or per[k].get("reach_fail")
        ), len(per) - 1)
        vs[fi] += r - sum(vs)
    return [float(v) for v in vs]
