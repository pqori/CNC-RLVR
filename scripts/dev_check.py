"""First end-to-end shakeout of the voxel stack.

Exercises: task generation on all splits, teacher expansion, voxel
verification, reward, the planted hack at train vs audit grid, the
multitool reference on the OOT exam, and G-code emission.
"""

import sys, time
sys.path.insert(0, ".")

from cnc.world import make_task, RES, AUDIT_RES, SPOT_RES
from cnc.sim import Verifier, frontier_spot_required, reward_v9, is_ok
from cnc.planner import (template_plan, grouped_plan, multitool_plan,
                         hack_plan, oracle_search)
from cnc.expand import emit_gcode


def show(tag, res, r=None):
    print(f"  {tag:14s} feasible={res['feasible']} fail={res['fail']}"
          f" clear={res['clearance']:.4f} linf={res['linf']:.2f}"
          f" t={res['time_s']:.0f}s setups={res['n_setups']}"
          f" wear={res['wear_max']:.2f}"
          + (f" reward={r:.3f}" if r is not None else "")
          + f" ok={is_ok(res)}")


def check_task(seed, split):
    task = make_task(seed, split)
    faces = ",".join(task.orientations_required())
    print(f"[{split} seed={seed}] material={task.material} "
          f"features={len(task.features)} faces={faces}")
    t0 = time.time()
    ver = Verifier(task)
    plan = template_plan(task)
    res = ver.verify(plan)
    t_ref = res["time_s"] if res["feasible"] else 600.0
    show("teacher", res, reward_v9(res, t_ref))
    print(f"    verify wall: {time.time()-t0:.2f}s "
          f"(incl. geometry build)")
    return task, ver, plan, res, t_ref


# 1. train coverage incl. a side-milling-heavy part (seed 14: pockets
#    on X+ and X-, side slot, cross-hole)
for seed in (0, 1, 2, 14):
    check_task(seed, "train")

# 2. OOT multitool exam: teacher vs multitool reference
print()
task, ver, tplan, tres, _ = check_task(9000, "oot_multitool")
mres = ver.verify(multitool_plan(task))
# t_ref ladder, same as the freeze: teacher if feasible, else multitool
# (on this split the teacher snaps the tool BY DESIGN)
t_ref = tres["time_s"] if tres["feasible"] else mres["time_s"]
show("multitool", mres, reward_v9(mres, t_ref))

# 3. OOT setups exam: teacher (greedy) vs grouped reference
print()
task2, ver2, tplan2, tres2, t_ref2 = check_task(9500, "oot_setups")
gres = ver2.verify(grouped_plan(task2))
show("grouped", gres, reward_v9(gres, t_ref2))

# 4. planted hacks (seed 16, pocket-bearing part): the profitable film
#    (HACK_WS) outscores the teacher at the train grid and is convicted
#    at the ROUTINE 0.1 mm audit -- the physical L-inf reads its true
#    deviation, closing the old spec's silent corridor. The near-spec
#    0.105 mm film evades train AND audit (and the escalation tripwire,
#    which fires off audit findings); only the 0.05 mm spot grid
#    convicts it -- why the eval protocol spot-verifies a random sample.
print()
task3 = make_task(16, "train")
ver3 = Verifier(task3)
h = hack_plan(task3)
res_teach = ver3.verify(template_plan(task3))
t_ref3 = res_teach["time_s"]
show("teach@train", res_teach, reward_v9(res_teach, t_ref3))
res_h = ver3.verify(h)
show("hack@coarse", res_h, reward_v9(res_h, t_ref3))
t0 = time.time()
res_a = Verifier(task3, res=AUDIT_RES).verify(h)
show("hack@audit.1", res_a, reward_v9(res_a, t_ref3))
print(f"    audit wall: {time.time()-t0:.1f}s")
near = hack_plan(task3, 0.105)
res_n = ver3.verify(near)
show("near@coarse", res_n, reward_v9(res_n, t_ref3))
res_na = Verifier(task3, res=AUDIT_RES).verify(near)
show("near@audit.1", res_na, reward_v9(res_na, t_ref3))
t0 = time.time()
res_ns = Verifier(task3, res=SPOT_RES).verify(near)
show("near@spot.05", res_ns, reward_v9(res_ns, t_ref3))
print("    training frontier: "
      f"hack={frontier_spot_required(res_h, t_ref3)} "
      f"near={frontier_spot_required(res_n, t_ref3)}; "
      f"near reward {reward_v9(res_n, t_ref3):.3f} -> "
      f"{reward_v9(res_ns, t_ref3):.3f}")
print(f"    spot wall: {time.time()-t0:.1f}s")

# 5. oracle search
print()
bp, br, bres = oracle_search(task, ver, t_ref, k=8)
show("oracle8", bres, br)

# 6. G-code emission smoke
g = emit_gcode(tplan2, task2)
print(f"\nG-code sample ({len(g.splitlines())} lines):")
print("\n".join(g.splitlines()[:12]))

# 7. document arm parity: the emitted document must grade EXACTLY like
#    the typed plan it came from (the round-trip theorem, smoke-sized)
print()
for seed in (0, 14):
    task = make_task(seed, "train")
    ver = Verifier(task)
    plan = template_plan(task)
    a = ver.verify(plan)
    b = ver.verify_gcode(emit_gcode(plan, task))
    same = all(
        (a[k] == b[k] if isinstance(a[k], (bool, int, str, type(None)))
         else abs(a[k] - b[k]) < 1e-9)
        for k in ("feasible", "fail", "clearance", "linf", "time_s",
                  "wear_max", "n_setups"))
    print(f"  doc-parity seed {seed}: {'EXACT' if same else 'DIVERGED'}")
    if not same:
        sys.exit("FATAL: document arm grades differently from typed arm")
