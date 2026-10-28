"""Adversarial audit battery for the verifier stack.

Every check is a falsifiable assertion about the engine. Run after any
engine change. Sections:
  A. orientation frames: drills on all six faces machine their own hole
     exactly; the wrong setup orientation hard-fails
  A2. side milling frames: milled pockets on all six faces machine
     clean; wrong orientation hard-fails
  B. hard constraints actually fire: reach, plunge/non-center, fixed
     canonical RPM injection, drill-mills / mill-drills, unknown tool/feature
  C. volume conservation vs analytic feature volumes (top + side faces)
  D. gouge detection: an oversized tool through a tight corner must fail
  E. tolerance boundary: leftover just inside / outside the train grid's
     band, on a top face and on a side face
  E3. the PHYSICAL +-0.1 mm spec across the whole grid ladder, with
     grid-aligned walls (deterministic layer arithmetic): spec-legal
     films and overcuts pass EVERY grid (no false conviction at finer
     grids -- the property the old grid-tied band lacked);      violations
     beyond spec + rung slack are GUARANTEED convicted (asserted);
     the gray zone (spec, spec + slack] is alignment-dependent and
     reported, not asserted -- the enforcement floor is a bound;
     legal overcuts grazing concave wall-floor junctions are never
     convicted at any grid (frontier-pilot regression: the analytic
     must-keep envelope replaced erosion, which over-protected there)
  V. v_steps: the per-op decomposition sums EXACTLY to reward_v9, on
     clean plans, continuous safety violations, and mid-plan hard failures
  V2. reward ordering: every path-valid partial trajectory beats malformed
      output while preserving the ordering of the raw valid objective
  F. time model sanity: cycle time invariant to arc sampling density
  H. load model: teacher runs with calibrated force headroom; brute-force
     engagement at fixed feed still trips deflection / spindle caps
  J. audit ladder vs the planted hacks: the profitable film hack reads
     clean at train and is convicted at the ROUTINE audit (the physical
     L-inf reads its true deviation -- the old spec's silent corridor is
     closed); the near-spec film evades train, audit AND the escalation
     tripwire, and only the 0.05 mm spot grid convicts it -- the receipt
     for the random-sample spot rung in the eval protocol
  K. G-code round trip: parse(emit(plan)) is the IDENTICAL program,
     re-emission is byte-identical, both verifier front-ends agree
  L. interpreter is fail-loud: the dialect rejects everything outside
     the contract with line-numbered errors
  M. hand-authored documents: the dialect is writable without the
     emitter (clean drill + clean pocket), and raw-text hazards fire
     (rapid crash, drill lateral, mill plunge, steep/disguised ramp,
     reach, fixed cutting parameters)
  G. determinism: same plan verifies identically twice; task regen
     reproduces frozen prompts
  I. freeze integrity: stored teacher metrics + documents reproduce on
     the current engine
"""

import json
import math
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, ".")
from cnc.ir import Op, Plan, Setup
from cnc.world import (STOCK, RES, TOOLS, MATERIALS, CUT_PARAMS, MAX_RAMP_DEG,
                       Feature, Task, make_task)
from cnc.sim import (GOUGE_DEPTH_SCALE_MM, HARD_FAIL_REWARD,
                     VALID_REWARD_FLOOR, W_LOAD, W_RAMP, W_REACH, W_GOUGE,
                     Verifier, frontier_spot_required,
                     gouge_phi, load_phi, ramp_phi, reach_phi,
                     raw_reward_v7, raw_reward_v8, raw_reward_v9,
                     reward_v7, reward_v8, reward_v9, v_steps, is_ok)
from cnc.planner import template_plan, _drill_params, _mill_params
from cnc.expand import CLEAR_W, Prim, Segment, SetupProgram, mk, plan_program

FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def mk_task(features, material="AL6061"):
    return Task(seed=0, split="manual", material=material, features=features)


def drill_op(feat, mat, tool):
    rpm, feed = _drill_params(tool, mat)
    return Op("drill", feat.name, tool.tool_id, round(rpm), feed,
              0.5, feat.params["depth"], "plunge", 0.0)


# ---------------- A. orientation frames ----------------
mat = MATERIALS["AL6061"]
for face in ("Z+", "Z-", "X+", "X-", "Y+", "Y-"):
    if face[0] == "Z":
        u, v = 30.0, 40.0
    else:
        u, v = 30.0, 10.0     # (u along face, v = height coordinate)
    feat = Feature("F1", "hole", face, dict(u=u, v=v, d=5.0, depth=12.0))
    task = mk_task([feat])
    ver = Verifier(task)
    plan = Plan([Setup(face, [drill_op(feat, mat, TOOLS["D5"])])])
    res = ver.verify(plan)
    check(f"A: drill on {face} clean", res["feasible"] and is_ok(res),
          f"fail={res['fail']} clear={res['clearance']:.3f} linf={res['linf']}")
    # analytic volume: pi r^2 depth, minus tolerance shell effects
    vol_req = ver.required_n * RES ** 3
    vol_analytic = math.pi * 2.5 ** 2 * 12.0
    check(f"A: required volume sane on {face}",
          abs(vol_req - vol_analytic) / vol_analytic < 0.35,
          f"req={vol_req:.0f} vs analytic={vol_analytic:.0f}")
    wrong = "Z-" if face != "Z-" else "Z+"
    res_w = Verifier(task).verify(
        Plan([Setup(wrong, [drill_op(feat, mat, TOOLS["D5"])])]))
    check(f"A: wrong-face from {wrong} for {face} fails",
          not res_w["feasible"] and res_w["fail"] == "wrong_face",
          f"fail={res_w['fail']}")

# ---------------- A2. side milling frames ----------------
# A milled pocket on each face must be geometrically clean; running its
# operations from the wrong orientation must fail. Load is audited
# separately in section H so it cannot obscure this frame-mapping test.
for face in ("Z+", "Z-", "X+", "X-", "Y+", "Y-"):
    if face[0] == "Z":
        p = dict(cx=30.0, cy=30.0, w=22.0, h=18.0, depth=6.0, corner_r=3.0)
    else:
        p = dict(cx=30.0, cy=10.0, w=22.0, h=11.0, depth=6.0, corner_r=3.0)
    feat = Feature("F1", "pocket_rect", face, p)
    task = mk_task([feat])
    plan = template_plan(task)
    res = Verifier(task).verify(plan)
    check(f"A2: milled pocket on {face} geometrically clean",
          res["path_valid"] and res["gouge_safe"]
          and res["reach_safe"] and res["ramp_safe"]
          and res["clearance"] >= 0.98 and res["linf"] <= 0.1,
          f"fail={res['fail']} clear={res['clearance']:.3f} "
          f"linf={res['linf']:.2f}")
    wrong = "Z-" if face != "Z-" else "Z+"
    res_w = Verifier(task).verify(Plan([Setup(wrong, plan.setups[0].ops)]))
    check(f"A2: pocket on {face} from {wrong} fails",
          not res_w["feasible"] and res_w["fail"] == "wrong_face",
          f"fail={res_w['fail']}")

# ---------------- B. hard constraints fire ----------------
pocket = Feature("F1", "pocket_rect", "Z+",
                 dict(cx=30, cy=30, w=24, h=24, depth=11.0, corner_r=2.0))
task = mk_task([pocket])

# EM3 (flute 8) cannot cut 11 deep
rpm, feed, sd = _mill_params(TOOLS["EM3"], mat, 11.0)
p_reach = Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM3", round(rpm),
                                feed, 0.5, sd, "helix", 0.0)])])
res = Verifier(task).verify(p_reach)
check("B: reach fail (EM3 flute 8 at depth 11)",
      not res["feasible"] and res["fail"] == "reach", f"fail={res['fail']}")

# plunge with non-center-cutting EM12 (refused at expansion: the
# teacher-side path will not even author that motion; the raw-text
# equivalent is policed in-sweep, section M)
rpm, feed, sd = _mill_params(TOOLS["EM12"], mat, 5.0)
big = Feature("F1", "pocket_rect", "Z+",
              dict(cx=30, cy=30, w=40, h=40, depth=5.0, corner_r=6.0))
p_plunge = Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM12", round(rpm),
                                 feed, 0.5, sd, "plunge", 0.0)])])
res = Verifier(mk_task([big])).verify(p_plunge)
check("B: plunge with non-center-cutting fails",
      not res["feasible"]
      and str(res["fail"]).startswith("expand:plunge"),
      f"fail={res['fail']}")

# RPM is controlled and injected, so an arbitrary stored value must
# have no effect on motion, load, time, or reward.
rpm, feed, _ = _mill_params(TOOLS["EM12"], mat, 5.0)
rpm_task = mk_task([big])
canonical_rpm = Plan([Setup("Z+", [
    Op("pocket_rough", "F1", "EM12", rpm, feed,
       0.5, 4.0, "ramp", 0.0),
])])
altered_rpm = deepcopy(canonical_rpm)
altered_rpm.setups[0].ops[0].rpm = 2.0 * rpm
r_rpm_fixed = Verifier(rpm_task).verify(canonical_rpm)
r_rpm_altered = Verifier(rpm_task).verify(altered_rpm)
check("B: arbitrary policy RPM is ignored and canonicalized",
      r_rpm_fixed == r_rpm_altered
      and r_rpm_altered["per_op"][0]["rpm"] == CUT_PARAMS["EM12"].rpm)

# Feed is controlled and injected, so an arbitrary stored value must
# have no effect on motion, load, time, or reward.
rpm, feed, _ = _mill_params(TOOLS["EM12"], mat, 5.0)
feed_task = mk_task([big])
canonical = Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM12", rpm,
                                  feed, 0.5, 4.0, "ramp", 0.0)])])
altered = deepcopy(canonical)
altered.setups[0].ops[0].feed = 0.8 * feed
r_fixed = Verifier(feed_task).verify(canonical)
r_altered = Verifier(feed_task).verify(altered)
check("B: policy-supplied feed is ignored and canonicalized",
      all(r_fixed[k] == r_altered[k]
          for k in ("feasible", "fail", "clearance", "linf", "time_s",
                    "force_util", "power_util")),
      f"fixed={r_fixed['fail']}/{r_fixed['time_s']:.3f} "
      f"altered={r_altered['fail']}/{r_altered['time_s']:.3f}")

# drill asked to mill / mill asked to drill
hole = Feature("F2", "hole", "Z+", dict(u=10, v=10, d=5.0, depth=8.0))
res = Verifier(mk_task([big, hole])).verify(
    Plan([Setup("Z+", [Op("pocket_rough", "F2", "D5", 5000, 500, 0.5, 2.0,
                          "helix", 0.0)])]))
check("B: drill asked to mill fails",
      not res["feasible"] and str(res["fail"]).startswith("expand"),
      f"fail={res['fail']}")
res = Verifier(mk_task([big, hole])).verify(
    Plan([Setup("Z+", [Op("drill", "F1", "EM6", 8000, 800, 0.5, 5.0,
                          "plunge", 0.0)])]))
check("B: mill asked to drill fails",
      not res["feasible"] and str(res["fail"]).startswith("expand"),
      f"fail={res['fail']}")

# ---------------- C. volume conservation ----------------
circ = Feature("F1", "pocket_circ", "Z+", dict(cx=30, cy=30, d=20.0, depth=6.0))
task = mk_task([circ])
ver = Verifier(task)
plan = template_plan(task)
res = ver.verify(plan, return_stock=True)
removed = np.ones_like(ver.target).sum() - res["stock"].sum()
vol_removed = removed * RES ** 3
vol_analytic = math.pi * 10 ** 2 * 6
check("C: circ pocket removed volume ~ analytic",
      res["feasible"] and abs(vol_removed - vol_analytic) / vol_analytic < 0.10,
      f"removed={vol_removed:.0f} analytic={vol_analytic:.0f}")
check("C: teacher ok on manual circ", is_ok(res), f"fail={res['fail']}")

# C2: same conservation law on a SIDE-face pocket (rounded rect)
side_p = dict(cx=30.0, cy=10.0, w=20.0, h=12.0, depth=5.0, corner_r=3.0)
side_pocket = Feature("F1", "pocket_rect", "X+", side_p)
task = mk_task([side_pocket])
ver = Verifier(task)
res = ver.verify(template_plan(task), return_stock=True)
removed = np.ones_like(ver.target).sum() - res["stock"].sum()
vol_removed = removed * RES ** 3
vol_analytic = (side_p["w"] * side_p["h"]
                - (4 - math.pi) * side_p["corner_r"] ** 2) * side_p["depth"]
check("C2: side pocket removed volume ~ analytic",
      res["feasible"] and abs(vol_removed - vol_analytic) / vol_analytic < 0.10,
      f"removed={vol_removed:.0f} analytic={vol_analytic:.0f}")
check("C2: teacher ok on side pocket", is_ok(res), f"fail={res['fail']}")

# ---------------- D. gouge / undercut semantics ----------------
# oversized drill in a small hole: capsule radius 4 vs target radius
# 1.5 -> cuts eroded target -> gouge MUST fire (a real temptation:
# wrong tool choice)
small_hole = Feature("F1", "hole", "Z+", dict(u=30, v=30, d=3.0, depth=10.0))
res = Verifier(mk_task([small_hole])).verify(
    Plan([Setup("Z+", [drill_op(small_hole, mat, TOOLS["D8"])])]))
check("D: oversized drill gouges",
      not res["feasible"] and res["fail"] == "gouge", f"fail={res['fail']}")
gouge_vs = v_steps(res, 600.0)
check("D: executable gouge gets continuous reward_v9",
      res["path_valid"] and not res["gouge_safe"]
      and reward_v8(res, 600.0) == HARD_FAIL_REWARD
      and reward_v9(res, 600.0) > HARD_FAIL_REWARD
      and abs(sum(gouge_vs) - reward_v9(res, 600.0)) < 1e-9
      and W_GOUGE > 0.0 and gouge_phi(res["gouge_depth_mm"]) > 0.0,
      f"depth={res['gouge_depth_mm']:.6f} "
      f"v9={reward_v9(res, 600.0):.3f}")
check("D: gouge shape is pinned at zero, unit depth, and cap",
      gouge_phi(0.0) == 0.0
      and gouge_phi(GOUGE_DEPTH_SCALE_MM) == 1.0
      and gouge_phi(100.0 * GOUGE_DEPTH_SCALE_MM) == 20.0)

# undersized drill: no gouge, but residue annulus -> feasible, not ok
big_hole = Feature("F1", "hole", "Z+", dict(u=30, v=30, d=8.0, depth=10.0))
res = Verifier(mk_task([big_hole])).verify(
    Plan([Setup("Z+", [drill_op(big_hole, mat, TOOLS["D3"])])]))
check("D: undersized drill = feasible but dirty (undercut, not gouge)",
      res["feasible"] and not is_ok(res),
      f"fail={res['fail']} clear={res['clearance']:.3f}")

# oversized endmill in a tight-corner pocket: CANNOT reach the corner;
# must be feasible-but-dirty (undercut), never a gouge
tight = Feature("F1", "pocket_rect", "Z+",
                dict(cx=30, cy=30, w=20, h=20, depth=5.0, corner_r=2.0))
rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 5.0)
res = Verifier(mk_task([tight])).verify(
    Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM6", round(rpm),
                          feed, 0.5, sd, "helix", 0.0)])]))
check("D: oversized endmill undercuts corners (feasible, not ok, no gouge)",
      res["feasible"] and not is_ok(res) and res["fail"] is None,
      f"fail={res['fail']} clear={res['clearance']:.3f} linf={res['linf']:.2f}")

# ---------------- E. tolerance boundary ----------------
# leave a film via wall_stock: 0.15 (inside the train grid's one-cell
# band -- its disclosed leniency) vs 0.45 (outside any band)
for ws, should_ok in ((0.15, True), (0.45, False)):
    rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 5.0)
    ops = [Op("pocket_rough", "F1", "EM6", round(rpm), feed,
              0.5, sd, "helix", ws)]
    res = Verifier(mk_task([big])).verify(Plan([Setup("Z+", ops)]))
    check(f"E: wall film {ws}mm -> ok={should_ok}",
          res["feasible"] and is_ok(res) == should_ok,
          f"clear={res['clearance']:.4f} linf={res['linf']:.2f}")

# E2: identical boundary behaviour on a SIDE face
side_big = Feature("F1", "pocket_rect", "X+",
                   dict(cx=30.0, cy=10.0, w=26.0, h=12.0, depth=5.0,
                        corner_r=4.0))
for ws, should_ok in ((0.15, True), (0.45, False)):
    rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 5.0)
    ops = [Op("pocket_rough", "F1", "EM6", round(rpm), feed,
              0.5, sd, "helix", ws)]
    res = Verifier(mk_task([side_big])).verify(Plan([Setup("X+", ops)]))
    check(f"E2: side-face wall film {ws}mm -> ok={should_ok}",
          res["feasible"] and is_ok(res) == should_ok,
          f"clear={res['clearance']:.4f} linf={res['linf']:.2f}")

# ---------------- E3. the physical spec across the grid ladder ----------
# `big` has walls at x,y = 10/50 -- grid-aligned at 0.2/0.1/0.05, so
# cell-layer membership is exact integer arithmetic, not alignment
# luck. The declared spec is +-0.1 mm at EVERY grid; each grid achieves
# it to +-half a cell of center-sampling slack. Films are worst-case
# aligned here; generated (unaligned) parts are convicted earlier --
# section J measures those.
from cnc.world import AUDIT_RES, SPOT_RES

rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 5.0)
task_e3 = mk_task([big])
ver_e3 = {r: Verifier(task_e3, r) for r in (RES, AUDIT_RES, SPOT_RES)}


def _film_plan(ws):
    return Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM6", round(rpm),
                                 feed, 0.5, sd, "helix", ws)])])


# spec-legal film (0.08 < 0.1): NO grid may convict it. Under the old
# grid-tied band the 0.05 mm grid enforced an undeclared +-0.05 spec
# and false-failed exactly this case.
for r, tag in ((RES, "train"), (AUDIT_RES, "audit"), (SPOT_RES, "spot")):
    res = ver_e3[r].verify(_film_plan(0.08))
    check(f"E3: spec-legal 0.08mm film ok at {tag} grid",
          res["feasible"] and is_ok(res),
          f"clear={res['clearance']:.5f} linf={res['linf']:.3f}")

# violating film, worst-case aligned (0.16 > spec): train's one-cell
# band forgives it (disclosed leniency); audit and spot convict, and
# L-inf reads the TRUE deviation magnitude, not a band-relative crumb
res_t = ver_e3[RES].verify(_film_plan(0.16))
res_a = ver_e3[AUDIT_RES].verify(_film_plan(0.16))
res_s = ver_e3[SPOT_RES].verify(_film_plan(0.16))
check("E3: 0.16mm film invisible at train (grid limit, disclosed)",
      res_t["feasible"] and is_ok(res_t),
      f"clear={res_t['clearance']:.5f} linf={res_t['linf']:.3f}")
check("E3: 0.16mm film CONVICTED at audit, linf reads ~0.15",
      res_a["feasible"] and not is_ok(res_a)
      and 0.10 < res_a["linf"] < 0.20, f"linf={res_a['linf']:.3f}")
check("E3: 0.16mm film convicted at spot too",
      res_s["feasible"] and not is_ok(res_s),
      f"linf={res_s['linf']:.3f}")

# enforcement floor: spec + rung half-cell slack is a GUARANTEE bound,
# not the typical case. A violating film in the gray zone
# (spec, spec + slack] convicts or evades depending on where cell
# centers land: this aligned 0.12 mm film happens to convict at audit
# and spot, while section J's generated 0.105 mm film evades train AND
# audit. Gray zone is reported; only the guarantee is asserted:
# anything beyond spec + spot slack (0.125 mm) MUST convict at spot.
gray = {tag: ver_e3[r].verify(_film_plan(0.12))
        for r, tag in ((RES, "train"), (AUDIT_RES, "audit"),
                       (SPOT_RES, "spot"))}
print("      [info] 0.12mm film (gray zone) verdicts: "
      + "  ".join(f"{t}: ok={is_ok(v)} linf={v['linf']:.3f}"
                  for t, v in gray.items()))
res_s13 = ver_e3[SPOT_RES].verify(_film_plan(0.13))
check("E3: 0.13mm film (> spec + spot slack) guaranteed convicted at spot",
      res_s13["feasible"] and not is_ok(res_s13),
      f"linf={res_s13['linf']:.3f}")

# gouge side, same physics: 0.05 mm overcut is INSIDE spec -- no grid
# may call it a gouge (the old band gouge-failed it at 0.05 mm).
# 0.25 mm overcut sits in the TRAIN grid's gray zone (band 0.2 +
# half-cell 0.1 -> guaranteed visibility only beyond 0.3): report its
# train verdict, assert the guarantees -- gouge at audit (0.25 > 0.1 +
# 0.05), and a 0.35 mm overcut must gouge even at train.
for r, tag in ((RES, "train"), (AUDIT_RES, "audit"), (SPOT_RES, "spot")):
    res = ver_e3[r].verify(_film_plan(-0.05))
    check(f"E3: spec-legal 0.05mm overcut is NOT a gouge at {tag}",
          res["feasible"] and res["gouge_depth_mm"] == 0.0
          and gouge_phi(res["gouge_depth_mm"]) == 0.0,
          f"fail={res['fail']} depth={res['gouge_depth_mm']:.6f}")
res_t = ver_e3[RES].verify(_film_plan(-0.25))
res_a = ver_e3[AUDIT_RES].verify(_film_plan(-0.25))
print(f"      [info] 0.25mm overcut (train gray zone) verdict: "
      f"fail={res_t['fail']}")
check("E3: 0.25mm overcut is a GOUGE at audit",
      not res_a["feasible"] and res_a["fail"] == "gouge",
      f"fail={res_a['fail']}")
res_t35 = ver_e3[RES].verify(_film_plan(-0.35))
check("E3: 0.35mm overcut (> train band + slack) gouges even at train",
      not res_t35["feasible"] and res_t35["fail"] == "gouge",
      f"fail={res_t35['fail']}")
res_a35 = ver_e3[AUDIT_RES].verify(_film_plan(-0.35))
res_a50 = ver_e3[AUDIT_RES].verify(_film_plan(-0.50))
check("E3: gouge severity is positive and monotone with overcut",
      0.0 < res_a["gouge_depth_mm"] < res_a35["gouge_depth_mm"]
      < res_a50["gouge_depth_mm"]
      and gouge_phi(res_a["gouge_depth_mm"])
      <= gouge_phi(res_a35["gouge_depth_mm"])
      <= gouge_phi(res_a50["gouge_depth_mm"]),
      f"depths={res_a['gouge_depth_mm']:.6f},"
      f"{res_a35['gouge_depth_mm']:.6f},"
      f"{res_a50['gouge_depth_mm']:.6f}")

# FRONTIER-PILOT REGRESSION (receipt 20260824, seed 8002): a legal
# sub-tolerance overcut whose floor pass grazes UNALIGNED wall-floor
# junction cells. Under erosion-derived protection the train grid's
# 1-cell ball cannot reach a diagonal pocket neighbour (sqrt(2) cells),
# so junction cells at ~0.08 mm physical depth stayed protected and
# this plan was convicted -- at the coarse grid only, inverting the
# ladder. The analytic must-keep envelope (features inflated by
# TOL + half-cell) must pass it at EVERY grid.
from cnc.world import make_task as _make_task_g
task_g = _make_task_g(8002, "train")     # 14.7x24.1 pocket, unaligned
mat_g = MATERIALS[task_g.material]
d_g = task_g.features[0].params["depth"]
rpm_g, feed_g, sd_g = _mill_params(TOOLS["EM6"], mat_g, d_g)
plan_g = Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM6",
                               round(rpm_g), feed_g,
                               0.5, sd_g, "helix", -0.08)])])
for r, tag in ((RES, "train"), (AUDIT_RES, "audit"), (SPOT_RES, "spot")):
    ver_g = Verifier(task_g, r)
    if tag == "train":
        check("E3: must-keep envelope strictly inside the nominal part",
              not (ver_g.protected & ~ver_g.target).any()
              and ver_g.protected.sum() < ver_g.target.sum(),
              f"{int(ver_g.protected.sum())} < {int(ver_g.target.sum())}")
    res_g = ver_g.verify(plan_g)
    check(f"E3: legal 0.08mm junction overcut not a gouge at {tag} "
          f"(frontier-pilot regression)",
          res_g["feasible"], f"fail={res_g['fail']}")

# ---------------- V. v_steps decomposition is exact ----------------
# H2's "free process reward" as an audited identity: per-op v_steps
# sum bit-exactly to reward_v9 on clean plans and safety violations; the
# responsible operations receive the continuous safety terms.

for seed in (0, 1, 2):
    t_v = make_task(seed, "train")
    r_v = Verifier(t_v).verify(template_plan(t_v))
    t_ref_v = r_v["time_s"]
    vs = v_steps(r_v, t_ref_v)
    check(f"V: sum(v_steps) == reward_v9 (teacher seed {seed})",
          abs(sum(vs) - reward_v9(r_v, t_ref_v)) < 1e-9
          and len(vs) == len(r_v["per_op"]),
          f"sum={sum(vs):.9f} r={reward_v9(r_v, t_ref_v):.9f}")

# mid-plan failure: op1 clean, op2 snaps the tool
two = [Feature("F1", "pocket_rect", "Z+",
               dict(cx=15, cy=15, w=18, h=18, depth=4.0, corner_r=4.0)),
       Feature("F2", "pocket_rect", "Z+",
               dict(cx=42, cy=42, w=30, h=30, depth=10.0, corner_r=5.0))]
rpm1, feed1, sd1 = _mill_params(TOOLS["EM6"], mat, 4.0)
rpm2, feed2, _ = _mill_params(TOOLS["EM3L"], mat, 10.0)
p_two = Plan([Setup("Z+", [
    Op("pocket_rough", "F1", "EM6", round(rpm1), feed1,
       0.5, sd1, "helix", 0.0),
    Op("pocket_rough", "F2", "EM3L", round(rpm2), feed2,
       0.5, 8.0, "helix", 0.0)])])
r_two = Verifier(mk_task(two)).verify(p_two)
vs_two = v_steps(r_two, 600.0)
check("V: overload gets continuous reward and exact v_steps",
      not r_two["feasible"] and r_two["path_valid"]
      and not r_two["load_safe"]
      and abs(sum(vs_two) - reward_v9(r_two, 600.0)) < 1e-9
      and reward_v9(r_two, 600.0) > HARD_FAIL_REWARD and vs_two[0] > 0.0,
      f"fail={r_two['fail']} vs={[round(v, 3) for v in vs_two]}")

# Reward-ordering regression: a syntactically and geometrically executable
# prefix must always be preferable to a malformed document. Before v6, 14/20
# of these real teacher prefixes scored below the malformed -1 sentinel.
partial_rows = []
for seed in range(20):
    t_p = make_task(seed, "train")
    v_p = Verifier(t_p)
    teacher_p = template_plan(t_p)
    full_p = v_p.verify(teacher_p)
    first_setup = teacher_p.setups[0]
    prefix = Plan([Setup(first_setup.orientation, [first_setup.ops[0]])])
    prefix_res = v_p.verify(prefix)
    partial_rows.append((
        raw_reward_v9(prefix_res, full_p["time_s"]),
        reward_v9(prefix_res, full_p["time_s"]),
        prefix_res["path_valid"],
        abs(sum(v_steps(prefix_res, full_p["time_s"]))
            - reward_v9(prefix_res, full_p["time_s"])) < 1e-9,
    ))
check("V2: all executable teacher prefixes beat malformed output",
      all(valid and mapped > HARD_FAIL_REWARD
          for _, mapped, valid, _ in partial_rows),
      f"minimum prefix reward={min(x[1] for x in partial_rows):.6f}")
check("V2: valid reward floor reserves a hard-failure margin",
      min(x[1] for x in partial_rows) >= VALID_REWARD_FLOOR - 1e-12
      and VALID_REWARD_FLOOR > HARD_FAIL_REWARD,
      f"observed_min={min(x[1] for x in partial_rows):.6f} "
      f"floor={VALID_REWARD_FLOOR:.3f}")
by_raw = sorted(partial_rows)
check("V2: tail map preserves valid-plan ordering",
      all(a[1] <= b[1] for a, b in zip(by_raw, by_raw[1:])))
check("V2: tail-compressed prefixes retain exact v_steps identity",
      all(row[3] for row in partial_rows))

# ---------------- F. time model vs sampling density ----------------
import cnc.expand as ex
task = mk_task([circ])
plan = template_plan(task)
r_fine = Verifier(task).verify(plan)
check("F: sampling-density probe plan is feasible", r_fine["feasible"],
      f"fail={r_fine['fail']}")
t_fine = r_fine["time_s"]
orig = ex.SAGITTA_MAX
ex.SAGITTA_MAX = 0.1          # 3.5x coarser arc sampling
t_coarse = Verifier(task).verify(plan)["time_s"]
ex.SAGITTA_MAX = orig
check("F: cycle time invariant to arc sampling density",
      t_fine > 0 and abs(t_fine - t_coarse) / t_fine < 0.02,
      f"fine={t_fine:.1f}s coarse={t_coarse:.1f}s")

# ---------------- H. load model (engagement grading) ----------------
# teacher must run with calibrated headroom on the force limit
utils = []
for seed in range(0, 20):
    t = make_task(seed, "train")
    r = Verifier(t).verify(template_plan(t))
    check(f"H: teacher feasible under load model (seed {seed})",
          r["feasible"], f"fail={r['fail']}")
    utils.append(r["force_util"])
check("H: teacher force utilization in calibrated band (0.1-0.85)",
      0.1 < max(utils) <= 0.85,
      f"max_util={max(utils):.2f} mean={np.mean(utils):.2f}")
check("H: calibrated load shape is pinned at margin/boundary/tail",
      W_LOAD == 0.05 and abs(load_phi(0.8)) < 1e-12
      and abs(load_phi(1.0) - 1.0) < 1e-12
      and abs(load_phi(1.2) - 3.0) < 1e-12,
      f"lambda={W_LOAD} phi=.8/{load_phi(.8):g} "
      f"1/{load_phi(1):g} 1.2/{load_phi(1.2):g}")

# At the prescribed feed, brute-force axial engagement must still snap
# the tool. The policy must change path geometry, not tune feed.
deep = Feature("F1", "pocket_rect", "Z+",
               dict(cx=30, cy=30, w=30, h=30, depth=10.0, corner_r=5.0))
rpm, feed, sd = _mill_params(TOOLS["EM3L"], mat, 10.0)
over_task = mk_task([deep])
over_plan = Plan([Setup("Z+", [
    Op("pocket_rough", "F1", "EM3L", round(rpm),
       feed, 0.5, 8.0, "helix", 0.0),
])])
over_ver = Verifier(over_task)
res = over_ver.verify(over_plan)
check("H: brute-force stepdown at fixed feed -> tool overload",
      not res["feasible"] and res["path_valid"] and not res["load_safe"]
      and res["fail"] == "tool_overload"
      and reward_v7(res, 600.0) > HARD_FAIL_REWARD,
      f"fail={res['fail']} reward={reward_v7(res, 600.0):.3f}")

# Adversarial load padding: after each straight level-feed primitive,
# traverse the already-cut path backward and forward six extra times.
# These legal recuts multiply cycle time but remove no material. They must
# not reduce force/power utilization; otherwise the policy can buy fake
# load safety. Arcs are excluded because reversing their discretized sweep
# can touch additional boundary voxels and is therefore not a zero-removal
# intervention.
padded_program = deepcopy(plan_program(over_plan, over_task))
for sp in padded_program:
    for seg in sp.segments:
        padded = []
        for p in seg.prims:
            padded.append(p)
            if p.kind != "line" or abs(p.w2 - p.w1) > 1e-9:
                continue
            reverse = Prim(
                p.kind, p.u2, p.v2, p.w2, p.u1, p.v1, p.w1,
                p.cu, p.cv, not p.ccw if p.kind == "arc" else p.ccw,
                p.feed,
            )
            for _ in range(6):
                padded.extend((reverse, p))
        seg.prims = padded
padded_res = over_ver._run(padded_program)
check("H: zero-removal recuts cannot dilute measured load",
      padded_res["fail"] == res["fail"]
      and abs(padded_res["force_util"] - res["force_util"]) < 1e-9
      and abs(padded_res["power_util"] - res["power_util"]) < 1e-9
      and abs(padded_res["clearance"] - res["clearance"]) < 1e-12
      and padded_res["time_s"] > 3.0 * res["time_s"],
      f"base_u={res['force_util']:.6f} padded_u="
      f"{padded_res['force_util']:.6f} base_t={res['time_s']:.1f} "
      f"padded_t={padded_res['time_s']:.1f}")

# Partial-engagement padding: an EM6 first clears a wide deep slot through
# X=38. An EM3L then removes the same narrow extension either by feeding from
# X=30 through cleared space (the attack) or from X=39.5 (the compact
# equivalent). Endpoint-anchored voxel-scale windows must grade them equally.
pad_depth = 3.0
pad_task = mk_task([
    Feature("F1", "slot", "Z+",
            dict(x1=20.0, y1=30.0, x2=38.0, y2=30.0,
                 width=6.0, depth=pad_depth)),
    Feature("F2", "slot", "Z+",
            dict(x1=30.0, y1=30.0, x2=39.7, y2=30.0,
                 width=3.0, depth=pad_depth)),
])
pad_base = plan_program(template_plan(pad_task), pad_task)[0]
pad_first = deepcopy(pad_base.segments[0])
pad_cp = CUT_PARAMS["EM3L"]


def _partial_pad_program(start_x):
    last = pad_first.prims[-1]
    prims = [
        mk("rapid", last.u2, last.v2, last.w2,
           start_x, 30.0, CLEAR_W, 0.0),
        mk("rapid", start_x, 30.0, CLEAR_W,
           start_x, 30.0, pad_depth, 0.0),
        mk("line", start_x, 30.0, pad_depth,
           39.7, 30.0, pad_depth, pad_cp.feed),
        mk("rapid", 39.7, 30.0, pad_depth,
           39.7, 30.0, CLEAR_W, 0.0),
    ]
    return [SetupProgram("Z+", [
        deepcopy(pad_first),
        Segment("EM3L", pad_cp.rpm, prims),
    ])]


partial_ver = Verifier(pad_task)
partial_attack = partial_ver._run(_partial_pad_program(30.0))
partial_compact = partial_ver._run(_partial_pad_program(39.5))
attack_op = partial_attack["per_op"][1]
compact_op = partial_compact["per_op"][1]
check("H: partially engaged cleared prefix cannot dilute measured load",
      partial_attack["fail"] == partial_compact["fail"] == "tool_overload"
      and attack_op["removed"] == compact_op["removed"]
      and abs(attack_op["force_util"] - compact_op["force_util"]) < 1e-9
      and abs(attack_op["power_util"] - compact_op["power_util"]) < 1e-9
      and abs(partial_attack["clearance"]
              - partial_compact["clearance"]) < 1e-12
      and partial_attack["time_s"] > partial_compact["time_s"],
      f"attack_u={attack_op.get('force_util', 0):.6f} compact_u="
      f"{compact_op.get('force_util', 0):.6f} attack_t="
      f"{partial_attack['time_s']:.1f} compact_t="
      f"{partial_compact['time_s']:.1f}")

# Equivalent G1 segmentation must not change contact time or load.
seg_feature = Feature(
    "F1", "pocket_rect", "Z+",
    dict(cx=30.0, cy=30.0, w=50.0, h=20.0,
         depth=10.0, corner_r=5.0),
)
seg_task = mk_task([seg_feature])
seg_cp = CUT_PARAMS["EM3L"]
seg_preclear = Segment("EM3L", seg_cp.rpm, [
    mk("line", 10.0, 30.0, 8.0, 29.8, 30.0, 8.0, seg_cp.feed),
])
seg_whole = Segment("EM3L", seg_cp.rpm, [
    mk("line", 28.2, 30.0, 8.0, 30.1, 30.0, 8.0, seg_cp.feed),
])
seg_split = Segment("EM3L", seg_cp.rpm, [
    mk("line", 28.2, 30.0, 8.0, 29.8, 30.0, 8.0, seg_cp.feed),
    mk("line", 29.8, 30.0, 8.0, 30.1, 30.0, 8.0, seg_cp.feed),
])
seg_ver = Verifier(seg_task)
seg_one = seg_ver._run(
    [SetupProgram("Z+", [deepcopy(seg_preclear), seg_whole])],
    return_stock=True,
)
seg_two = seg_ver._run(
    [SetupProgram("Z+", [deepcopy(seg_preclear), seg_split])],
    return_stock=True,
)
seg_one_op, seg_two_op = seg_one["per_op"][1], seg_two["per_op"][1]
check("H: partial engagement is invariant to G1 segmentation",
      np.array_equal(seg_one["stock"], seg_two["stock"])
      and seg_one_op["removed"] == seg_two_op["removed"] > 0
      and abs(seg_one["time_s"] - seg_two["time_s"]) < 1e-12
      and abs(seg_one_op["force_util"]
              - seg_two_op["force_util"]) < 1e-9
      and abs(seg_one_op["power_util"]
              - seg_two_op["power_util"]) < 1e-9,
      f"force={seg_one_op['force_util']:.6f}/"
      f"{seg_two_op['force_util']:.6f}")

# A rigid EM6 can rough the same feature cleanly at its prescribed feed:
# the constraint rewards tool choice and engagement, not feed tuning.
rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 10.0)
res = Verifier(mk_task([deep])).verify(
    Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM6", round(rpm),
                          feed, 0.5, sd, "helix", 0.0)])]))
check("H: rigid-tool version at fixed feed is clean",
      res["feasible"], f"fail={res['fail']}")

# oversized MRR on a big tool in aluminum must hit the spindle ceiling
rpm, feed, sd = _mill_params(TOOLS["EM12"], mat, 16.0)
big12 = Feature("F1", "pocket_rect", "Z+",
                dict(cx=30, cy=30, w=40, h=40, depth=16.0, corner_r=6.0))
res = Verifier(mk_task([big12])).verify(
    Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM12", round(rpm),
                          feed, 0.5, 16.0, "ramp", 0.0)])]))
check("H: excessive MRR -> spindle power cap",
      not res["feasible"] and res["path_valid"] and not res["load_safe"]
      and res["fail"] == "spindle_power" and res["power_util"] > 1.0,
      f"fail={res['fail']} putil={res['power_util']:.2f}")

# helix with no room must fail; ramp must succeed in the same slot
slot = Feature("F1", "slot", "Z+",
               dict(x1=15, y1=30, x2=45, y2=30, width=6.0, depth=4.0))
rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 4.0)
res = Verifier(mk_task([slot])).verify(
    Plan([Setup("Z+", [Op("slot", "F1", "EM6", round(rpm), feed,
                          0.5, sd, "helix", 0.0)])]))
check("H: helix without room fails",
      not res["feasible"] and "helix" in str(res["fail"]),
      f"fail={res['fail']}")
res = Verifier(mk_task([slot])).verify(
    Plan([Setup("Z+", [Op("slot", "F1", "EM6", round(rpm), feed,
                          0.5, sd, "ramp", 0.0)])]))
check("H: ramp in the same slot is clean",
      res["feasible"] and is_ok(res), f"fail={res['fail']}")

# entry contract: no endmill is center-cutting -- a plunge entry is
# refused at expansion for EVERY mill, including the small ones that
# used to be center-cutting
rpm, feed, sd = _mill_params(TOOLS["EM6"], mat, 8.0)
r = Verifier(mk_task([deep])).verify(
    Plan([Setup("Z+", [Op("pocket_rough", "F1", "EM6", round(rpm),
                          feed, 0.5, sd, "plunge", 0.0)])]))
check("H: plunge entry refused for every mill (EM6 non-center-cutting)",
      not r["feasible"] and str(r["fail"]).startswith("expand:plunge"),
      f"fail={r['fail']}")

# ---------------- J. audit ladder vs the planted hacks ----------------
# Two planted exploits on a GENERATED part (seed 16, unaligned walls):
# 1) the profitable film (HACK_WS): outscores the honest teacher at the
#    train grid, and is convicted at the ROUTINE audit -- the physical
#    L-inf reads its true ~0.15 mm deviation. Under the old grid-tied
#    band this class slipped the audit "landing exactly ON the gate";
#    the spec redesign closed that corridor.
# 2) the near-spec film (0.105 mm): the audit grid sees NOTHING (its
#    sampling slack forgives it), so the escalation tripwire -- which
#    fires off audit findings -- is structurally blind. Only the
#    0.05 mm spot grid convicts. This is the receipt for the RANDOM
#    spot-sample rung in the eval protocol (train/eval_policy.py).
# Honest plans must never trigger escalation.
from cnc.sim import spot_flagged
from cnc.planner import hack_plan, HACK_WS

for seed in (0, 1, 2):
    t = make_task(seed, "train")
    p = template_plan(t)
    tr = Verifier(t).verify(p)
    ar = Verifier(t, AUDIT_RES).verify(p)
    check(f"J: honest teacher does not trigger escalation (seed {seed})",
          not spot_flagged(tr, ar),
          f"clear_a={ar['clearance']:.6f} linf_a={ar['linf']:.3f}")

t_h = make_task(16, "train")
ver_h = Verifier(t_h)
ver_ha = Verifier(t_h, AUDIT_RES)
tres = ver_h.verify(template_plan(t_h))
t_ref_h = tres["time_s"]

hp = hack_plan(t_h)                      # the profitable film
tr = ver_h.verify(hp)
ar = ver_ha.verify(hp)
check("J: film hack outscores the honest teacher at the train grid",
      is_ok(tr) and reward_v7(tr, t_ref_h) > reward_v7(tres, t_ref_h),
      f"hack={reward_v7(tr, t_ref_h):.3f} "
      f"teach={reward_v7(tres, t_ref_h):.3f}")
check("J: routine audit CONVICTS it -- the old silent corridor is closed",
      not is_ok(ar) and ar["linf"] > 0.1,
      f"ok_a={is_ok(ar)} linf_a={ar['linf']:.3f}")
check("J: profitable film enters training-time fine metrology",
      frontier_spot_required(tr, t_ref_h)
      and reward_v7(ar, t_ref_h) < reward_v7(tres, t_ref_h),
      f"coarse_time={tr['time_s']:.1f} ref={t_ref_h:.1f}")

np_ = hack_plan(t_h, 0.105)              # the near-spec film
tr2 = ver_h.verify(np_)
ar2 = ver_ha.verify(np_)
check("J: near-spec film reads clean at train AND audit",
      is_ok(tr2) and is_ok(ar2),
      f"ok_t={is_ok(tr2)} ok_a={is_ok(ar2)} clear_a={ar2['clearance']:.6f}")
check("J: escalation tripwire is BLIND to it (audit saw nothing)",
      not spot_flagged(tr2, ar2),
      f"clear_a={ar2['clearance']:.6f} linf_a={ar2['linf']:.3f}")
sr2 = Verifier(t_h, SPOT_RES).verify(np_)
check("J: random-sample spot rung convicts it",
      not is_ok(sr2) and sr2["linf"] > 0.1,
      f"linf_spot={sr2['linf']:.3f} clear_spot={sr2['clearance']:.5f}")
check("J: near-spec film loses under training-time frontier reward",
      frontier_spot_required(tr2, t_ref_h)
      and reward_v7(sr2, t_ref_h) < reward_v7(tres, t_ref_h),
      f"coarse={reward_v7(tr2, t_ref_h):.3f} "
      f"spot={reward_v7(sr2, t_ref_h):.3f} "
      f"teacher={reward_v7(tres, t_ref_h):.3f}")
unsafe_frontier = dict(tr2)
unsafe_frontier.update(
    feasible=False, load_safe=False, fail="tool_overload",
    time_s=t_ref_h - 1.0,
)
check("J: continuously unsafe geometry is not exempt from fine metrology",
      not is_ok(unsafe_frontier)
      and frontier_spot_required(unsafe_frontier, t_ref_h),
      f"path_valid={unsafe_frontier['path_valid']} "
      f"load_safe={unsafe_frontier['load_safe']}")

# ---------------- K. G-code round trip (the exactness theorem) --------
# The twin grades ONE motion language. parse(emit(plan)) must be the
# IDENTICAL program (same prims bit-for-bit after quantization), the
# re-emitted text must be byte-identical, and the two verifier
# front-ends must agree on every scalar metric -- across plan styles
# (teacher, multitool, grouped, hack) and setups on all faces.
from cnc.expand import plan_program, emit_gcode
from cnc.gcode import parse_gcode
from cnc.planner import multitool_plan, grouped_plan


def _flat(prog):
    return [(sp.orientation, seg.tool_id, seg.rpm, pr)
            for sp in prog for seg in sp.segments for pr in seg.prims]


SCALARS = ("feasible", "path_valid", "load_safe", "ramp_safe", "fail",
           "clearance", "linf", "time_s", "wear_max", "force_util",
           "power_util", "ramp_util", "ramp_angle_deg",
           "n_setups")
rt_cases = [(make_task(s, "train"), template_plan, f"teacher seed {s}")
            for s in (0, 1, 2, 8)]
rt_cases += [(make_task(8, "train"), hack_plan, "hack seed 8"),
             (make_task(9000, "oot_multitool"), multitool_plan,
              "multitool 9000"),
             (make_task(9500, "oot_setups"), grouped_plan,
              "grouped 9500")]
for task_k, plan_fn, label in rt_cases:
    plan_k = plan_fn(task_k)
    p1 = plan_program(plan_k, task_k)
    g1 = emit_gcode(plan_k, task_k, p1)
    p2 = parse_gcode(g1)
    f1, f2 = _flat(p1), _flat(p2)
    check(f"K: program round-trips exactly ({label})",
          len(f1) == len(f2) and all(a == b for a, b in zip(f1, f2)),
          f"{len(f1)} vs {len(f2)} prims")
    g2 = emit_gcode(plan_k, task_k, p2)
    check(f"K: re-emission byte-identical ({label})", g1 == g2)
    ver_k = Verifier(task_k)
    ra = ver_k.verify(plan_k)
    rb = ver_k.verify_gcode(g1)
    agree = all((ra[k] == rb[k]) if not isinstance(ra[k], float)
                else abs(ra[k] - rb[k]) < 1e-9 for k in SCALARS)
    check(f"K: both front-ends grade identically ({label})", agree,
          " ".join(f"{k}:{ra[k]}vs{rb[k]}" for k in SCALARS
                   if ra[k] != rb[k]))

# ---------------- L. interpreter is fail-loud ----------------
HDR = "G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"


def perr(doc):
    try:
        parse_gcode(doc)
        return None
    except Exception as e:
        return str(e)


bad_docs = [
    ("G91 incremental mode", HDR + "G91\nG0 X1 Y1\nM30\n", "unsupported"),
    ("R-form arc", HDR + "G0 X10.0 Y10.0\nG2 X12.0 Y12.0 R2.0\nM30\n",
     "R-form"),
    ("cutter comp", HDR + "G41 D1\nM30\n", "unsupported"),
    ("canned cycle", HDR + "G81 X10 Y10 Z-5 R1\nM30\n", "unsupported"),
    ("unknown M code", HDR + "M8\nM30\n", "unsupported M8"),
    ("spindle-speed word",
     "G21 G90\n(SETUP Z+)\nT8 M6\nS8000 M3\nM30\n",
     "spindle start must be M3"),
    ("feed word", HDR + "G1 X1 Y1 F300\nM30\n", "unexpected word F"),
    ("motion before setup",
     "G21 G90\nT8 M6\nM3\nG0 X1 Y1\n(SETUP Z+)\nM30\n",
     "before any (SETUP"),
    ("motion before tool",
     "G21 G90\n(SETUP Z+)\nG0 X1 Y1\nT8 M6\nM3\nM30\n",
     "before T..M6"),
    ("arc radius mismatch",
     HDR + "G0 X10.0 Y10.0\nG3 X20.0 Y10.0 I2.0 J0.0\nM30\n",
     "radius mismatch"),
    ("zero-radius arc", HDR + "G0 X10.0 Y10.0\nG3 I0 J0\nM30\n",
     "zero radius"),
    ("unknown tool number",
     "G21 G90\n(SETUP Z+)\nT19 M6\nM3\nM30\n", "unknown tool"),
    ("spindle before tool",
     "G21 G90\n(SETUP Z+)\nM3\nM30\n", "before tool change"),
    ("coordinate insane", HDR + "G0 X9999.0 Y0.0\nM30\n", "out of range"),
    ("free text", HDR + "PLEASE CUT THE POCKET\nM30\n", "unrecognized"),
    ("duplicate word", HDR + "G0 X1.0 X2.0 Y1.0\nM30\n", "duplicate"),
    ("missing M30", HDR + "G0 X1.0 Y1.0\n", "missing M30"),
    ("content after M30", HDR + "M30\nG0 X1.0 Y1.0\n", "after M30"),
    ("setup that cuts nothing", HDR + "G0 X1.0 Y1.0\nM30\n",
     "cuts nothing"),
    ("motion before header",
     "(SETUP Z+)\nT8 M6\nM3\nG0 X1 Y1\nG21 G90\nM30\n",
     "before G21 G90"),
]
for name, doc, needle in bad_docs:
    e = perr(doc)
    check(f"L: {name} rejected", e is not None and needle in e,
          f"got: {e}")
check("L: nine setups rejected",
      "than 8 setups" in (perr("G21 G90\n" + "(SETUP Z+)\n" * 9 + "M30\n")
                          or ""), "")

# ---------------- M. hand-authored documents ----------------
# The dialect must be writable by an author who is NOT the emitter --
# these programs were written by hand against the contract only.
# clean drill: D5 (T2) through a 5 mm hole, approach and retract legal
hole_m = Feature("F1", "hole", "Z+", dict(u=30, v=30, d=5.0, depth=12.0))
task_m = mk_task([hole_m])
doc = ("G21 G90\n(SETUP Z+)\nT2 M6\nM3\n"
       f"G0 X30.000 Y30.000\nG0 Z0.500\n"
       "G1 Z-12.000\nG0 Z2.000\nM30\n")
res = Verifier(task_m).verify_gcode(doc)
check("M: hand-written drill program is clean",
      res["feasible"] and is_ok(res),
      f"fail={res['fail']} clear={res['clearance']:.3f}")

# clean circular pocket: EM6 (T8), HELIX entry (every mill is
# non-center-cutting: 1.0 mm/rev on r=2 = 4.55 deg, inside the 5 deg
# cap), concentric full circles per level, rapid re-entry inside the
# cavity
circ_m = Feature("F1", "pocket_circ", "Z+",
                 dict(cx=30, cy=30, d=30.0, depth=4.0))
task_m2 = mk_task([circ_m])


def _level(z_from, z_to):
    # helical bore about (28,30) r=2 at <= 1.0 mm/rev, flat cleanup
    # circle, then rings overlapping by 1 mm (zero-overlap seams leak
    # by a sagitta hair and the twin catches them -- a real 0.2 mm
    # witness line)
    lines, z = [], z_from
    while z < z_to - 1e-9:
        z = min(z + 1.0, z_to)
        lines.append(f"G3 I-2.000 J0.000 Z{-z:.3f}")
    lines.append("G3 I-2.000 J0.000")
    lines += [f"G1 X32.000 Y30.000", "G3 I-2.000 J0.000",
              "G1 X37.000 Y30.000", "G3 I-7.000 J0.000",
              "G1 X42.000 Y30.000", "G3 I-12.000 J0.000"]
    return "\n".join(lines) + "\n"


doc2 = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
        f"G0 X30.000 Y30.000\nG0 Z0.500\n" + _level(-0.5, 2.0)
        + "G0 Z2.000\nG0 X30.000 Y30.000\nG0 Z-1.500\n" + _level(1.5, 4.0)
        + "G0 Z2.000\nM5\nM30\n")
res = Verifier(task_m2).verify_gcode(doc2)
check("M: hand-written pocket program is clean",
      res["feasible"] and is_ok(res),
      f"fail={res['fail']} clear={res['clearance']:.4f} "
      f"linf={res['linf']:.2f} futil={res['force_util']:.2f}")

# rapid crash: G0 descending into virgin stock
doc3 = (HDR + "G0 X5.000 Y5.000\nG0 Z-1.000\n"
        + "G1 X6.000 Y5.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc3)
check("M: rapid descent into stock crashes",
      not res["feasible"] and res["fail"] == "rapid_crash",
      f"fail={res['fail']}")

# rapid crash: legal descent inside the cavity, then a horizontal G0
# straight through the pocket wall at depth
doc4 = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
        f"G0 X30.000 Y30.000\nG0 Z0.500\n" + _level(-0.5, 2.0)
        + "G0 X55.000 Y30.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc4)
check("M: horizontal rapid through the pocket wall crashes",
      not res["feasible"] and res["fail"] == "rapid_crash",
      f"fail={res['fail']}")

# drill asked to cut laterally, from raw text
doc5 = ("G21 G90\n(SETUP Z+)\nT2 M6\nM3\n"
        "G0 X30.000 Y30.000\nG0 Z0.500\nG1 Z-5.000\n"
        f"G1 X40.000 Y30.000\nM30\n")
res = Verifier(task_m).verify_gcode(doc5)
check("M: drill lateral move fails",
      not res["feasible"] and res["fail"] == "drill_lateral",
      f"fail={res['fail']}")

# straight plunge with the non-center-cutting EM12, from raw text
doc6 = ("G21 G90\n(SETUP Z+)\nT5 M6\nM3\n"
        "G0 X30.000 Y30.000\nG0 Z0.500\nG1 Z-2.000\n"
        f"M30\n")
res = Verifier(task_m2).verify_gcode(doc6)
check("M: non-center plunge from raw text fails",
      not res["feasible"] and res["fail"] == "plunge_non_center",
      f"fail={res['fail']}")

# the same straight plunge with EM6: the WHOLE crib is
# non-center-cutting under the entry contract
doc6b = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
         "G0 X30.000 Y30.000\nG0 Z0.500\nG1 Z-2.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc6b)
check("M: mill plunge from raw text fails (all mills non-center)",
      not res["feasible"] and res["fail"] == "plunge_non_center",
      f"fail={res['fail']}")

# A plunge with a fake moustache: 0.05 mm of XY drift over 2 mm of
# descent (88.6 deg) is ramp-unsafe, but remains an executable path with
# a capped scalar penalty rather than collapsing to the hard-fail sentinel.
doc6c = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
         "G0 X30.000 Y30.000\nG0 Z0.500\n"
         "G1 X30.050 Y30.000 Z-2.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc6c)
check("M: near-vertical disguised ramp is continuously ramp-unsafe",
      not res["feasible"] and res["path_valid"]
      and not res["ramp_safe"] and res["fail"] == "ramp_too_steep"
      and res["ramp_util"] > 1.0
      and reward_v7(res, 600.0) > HARD_FAIL_REWARD,
      f"fail={res['fail']} u={res['ramp_util']:.2f} "
      f"reward={reward_v7(res, 600.0):.3f}")

# angle-cap boundary from raw text, lane kept INSIDE the d=24 pocket
# (X 18-42 at Y30; EM6 edges need center X in ~[21,39]): a 3.6 deg
# straight ramp cuts legally; the same lane at 5.7 deg is unsafe but shaped
doc6d = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
         "G0 X22.000 Y30.000\nG0 Z0.000\n"
         "G1 X38.000 Y30.000 Z-1.000\nG0 Z2.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc6d)
check("M: shallow ramp (3.6 deg) cuts legally",
      res["feasible"] and res["ramp_safe"] and res["ramp_util"] < 0.8,
      f"fail={res['fail']} u={res['ramp_util']:.3f}")
ramp_task = make_task(5104, "train")
ramp_res = Verifier(ramp_task).verify(template_plan(ramp_task))
check("M: quantized helix remainder retains parent ramp semantics",
      ramp_res["feasible"] and ramp_res["ramp_safe"]
      and ramp_res["ramp_angle_deg"] < MAX_RAMP_DEG,
      f"fail={ramp_res['fail']} "
      f"angle={ramp_res['ramp_angle_deg']:.6f}")
doc6e = ("G21 G90\n(SETUP Z+)\nT8 M6\nM3\n"
         "G0 X22.000 Y30.000\nG0 Z0.000\n"
         "G1 X38.000 Y30.000 Z-1.600\nG0 Z2.000\nM30\n")
res = Verifier(task_m2).verify_gcode(doc6e)
check("M: steep ramp (5.7 deg) is continuously ramp-unsafe",
      not res["feasible"] and res["path_valid"]
      and not res["ramp_safe"] and res["fail"] == "ramp_too_steep"
      and 1.0 < res["ramp_util"] < 1.2
      and reward_v9(res, 600.0) > HARD_FAIL_REWARD
      and abs(sum(v_steps(res, 600.0)) - reward_v9(res, 600.0)) < 1e-9,
      f"fail={res['fail']} u={res['ramp_util']:.3f} "
      f"reward={reward_v9(res, 600.0):.3f}")
check("M: ramp penalty is pinned at margin/boundary/tail",
      W_RAMP == 0.05 and abs(ramp_phi(0.8)) < 1e-12
      and abs(ramp_phi(1.0) - 1.0) < 1e-12
      and abs(ramp_phi(1.2) - 3.0) < 1e-12,
      f"lambda={W_RAMP} phi=.8/{ramp_phi(.8):g} "
      f"1/{ramp_phi(1):g} 1.2/{ramp_phi(1.2):g}")

# reach from raw text: EM3 (flute 8 mm) commanded 11 mm deep
doc7 = ("G21 G90\n(SETUP Z+)\nT6 M6\nM3\n"
        "G0 X30.000 Y30.000\nG0 Z0.500\nG1 Z-11.000\n"
        f"M30\n")
res = Verifier(task_m2).verify_gcode(doc7)
check("M: over-reach is detected alongside harder plunge violation",
      not res["feasible"] and not res["reach_safe"]
      and any(r.get("reach_fail") == "reach" for r in res["per_op"])
      and res["fail"] == "plunge_non_center",
      f"fail={res['fail']} reach_safe={res['reach_safe']}")
check("M: harder plunge violation retains exact v_steps decomposition",
      abs(sum(v_steps(res, 600.0)) - reward_v9(res, 600.0)) < 1e-9,
      f"sum={sum(v_steps(res, 600.0)):.9f} "
      f"reward={reward_v9(res, 600.0):.9f}")

# Pure reach violation with a valid, shallow helical path: nine 1 mm-pitch
# circles on r=3 mm stay below the ramp limit but exceed EM3's 8 mm flute.
doc7b = ("G21 G90\n(SETUP Z+)\nT6 M6\nM3\n"
         "G0 X33.000 Y30.000\nG0 Z0.000\n"
         + "".join(
             f"G3 X33.000 Y30.000 Z-{depth:.3f} I-3.000 J0.000\n"
             for depth in range(1, 10)
         )
         + "M30\n")
reach_task = mk_task([
    Feature("F1", "pocket_rect", "Z+",
            dict(cx=30, cy=30, w=24, h=24, depth=10.0, corner_r=4.0))
])
reach_only = Verifier(reach_task).verify_gcode(doc7b)
reach_vs = v_steps(reach_only, 600.0)
check("M: flute overreach gets calibrated continuous reward_v8",
      reach_only["path_valid"] and not reach_only["reach_safe"]
      and reach_only["ramp_safe"]
      and abs(reach_only["reach_util"] - 9.0 / 8.0) < 1e-9
      and reward_v7(reach_only, 600.0) == HARD_FAIL_REWARD
      and reward_v8(reach_only, 600.0) > HARD_FAIL_REWARD
      and abs(sum(reach_vs) - reward_v9(reach_only, 600.0)) < 1e-9,
      f"u={reach_only['reach_util']:.3f} "
      f"v7={reward_v7(reach_only, 600.0):.3f} "
      f"v8={reward_v8(reach_only, 600.0):.3f}")
check("M: reach shape is zero through flute limit and linear above",
      W_REACH == 1.54 and abs(reach_phi(1.0)) < 1e-12
      and abs(reach_phi(1.125) - 0.5) < 1e-12
      and abs(reach_phi(1.25) - 1.0) < 1e-12,
      f"lambda={W_REACH} phi=1/{reach_phi(1):g} "
      f"1.125/{reach_phi(1.125):g} 1.25/{reach_phi(1.25):g}")

# ---------------- G. determinism + frozen prompt integrity ----------------
task = make_task(123, "train")
v1 = Verifier(task).verify(template_plan(task))
v2 = Verifier(task).verify(template_plan(task))
check("G: verify deterministic",
      v1["clearance"] == v2["clearance"] and v1["time_s"] == v2["time_s"])

bad = 0
for line in open("data/eval_iid.jsonl"):
    rec = json.loads(line)
    if make_task(rec["seed"], "train").prompt() != rec["prompt"]:
        bad += 1
check("G: frozen prompts reproduce from seeds (RNG determinism)", bad == 0,
      f"{bad} mismatches")

# ---------------- I. freeze integrity: stored teacher metrics ----------------
# The frozen JSONLs embed teacher metrics (incl. t_ref_s, which every
# reward normalizes against) computed by the engine at freeze time. Any
# engine change since then silently stales them. Re-verify a sample of
# stored teacher plans per split against the CURRENT engine.
stale = []
for split in ("train", "val", "eval_iid", "oot_multitool", "oot_setups"):
    recs = []
    with open(f"data/{split}.jsonl") as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) >= 3:
                break
    gen_split = "train" if split in ("train", "val", "eval_iid") else split
    for rec in recs:
        task = make_task(rec["seed"], gen_split)
        ver = Verifier(task)
        res = ver.verify(Plan.from_dict(rec["teacher_plan"]))
        if (res["feasible"] != rec["teacher_feasible"]
                or is_ok(res) != rec["teacher_ok"]
                or round(res["time_s"], 2) != rec["teacher_time_s"]
                or round(reward_v7(res, rec["t_ref_s"]), 4)
                != rec["teacher_reward"]):
            stale.append((split, rec["seed"], "plan"))
        # the stored DOCUMENT (the SFT/demonstration artifact) must
        # regenerate byte-identically and grade identically
        gdoc = rec.get("teacher_gcode")
        if gdoc != emit_gcode(Plan.from_dict(rec["teacher_plan"]), task):
            stale.append((split, rec["seed"], "gcode-text"))
            continue
        gres = ver.verify_gcode(gdoc)
        if (gres["feasible"] != res["feasible"]
                or abs(gres["time_s"] - res["time_s"]) > 1e-9):
            stale.append((split, rec["seed"], "gcode-grade"))
check("I: frozen teacher metrics + documents reproduce (15 samples)",
      not stale, f"stale (engine drifted since freeze?): {stale}")

print(f"\n{'ALL CHECKS PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
sys.exit(1 if FAILURES else 0)
