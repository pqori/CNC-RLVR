"""Engine cross-check: voxel twin vs independent 2.5D reference.

Two batteries, both demanding EXACT per-column agreement:
  1. Generated Z+-only parts (teacher + hack plans) -- the original
     subfamily check on the benchmark distribution.
  2. Manual single-face parts on ALL SIX faces (pocket + slot + hole,
     teacher + hack plans) -- validates side milling and the
     transpose/flip frame mapping in sim._face_view.
Any mismatch is a bug in one of the engines.
"""

import sys
import numpy as np
sys.path.insert(0, ".")

from cnc.world import Feature, Task, make_task
from cnc.sim import Verifier, _face_view
from cnc.planner import template_plan, multitool_plan, hack_plan
from cnc.ref25d import heightfield_remaining

checked = 0
worst = 0


def compare(task, plan, face, tag):
    global checked, worst
    ver = Verifier(task)
    vres = ver.verify(plan, return_stock=True)
    if not vres["feasible"]:
        print(f"{tag}: infeasible ({vres['fail']}) -- skipped")
        return
    vox_cols = _face_view(vres["stock"], face).sum(axis=2)
    ref_cols = heightfield_remaining(task, plan, face=face)
    diff = np.abs(vox_cols.astype(int) - ref_cols.astype(int))
    checked += 1
    if diff.max() > 0:
        worst = max(worst, int(diff.max()))
        n_bad = int((diff > 0).sum())
        print(f"MISMATCH {tag}: {n_bad} columns differ, "
              f"worst {diff.max()} cells")


# ---- 1. generated Z+-only parts from the benchmark distribution ----
for split, seeds, planners in [
    ("train", range(0, 120), (template_plan, hack_plan)),
    ("oot_multitool", range(9000, 9010), (template_plan, multitool_plan)),
]:
    for seed in seeds:
        task = make_task(seed, split)
        if task.orientations_required() != ["Z+"]:
            continue
        for planner in planners:
            compare(task, planner(task), "Z+",
                    f"seed {seed} {planner.__name__}")

# ---- 2. manual single-face parts on all six faces ----
def single_face_task(face):
    if face[0] == "Z":
        feats = [
            Feature("F1", "pocket_rect", face, dict(
                cx=22.0, cy=30.0, w=20.0, h=18.0, depth=6.0, corner_r=3.0)),
            Feature("F2", "slot", face, dict(
                x1=36.0, y1=46.0, x2=52.0, y2=46.0, width=6.0, depth=4.0)),
            Feature("F3", "hole", face, dict(u=45.0, v=14.0, d=5.0,
                                             depth=10.0)),
        ]
    else:
        feats = [
            Feature("F1", "pocket_rect", face, dict(
                cx=20.0, cy=10.0, w=18.0, h=10.0, depth=6.0, corner_r=3.0)),
            Feature("F2", "slot", face, dict(
                x1=34.0, y1=5.0, x2=50.0, y2=5.0, width=4.0, depth=4.0)),
            Feature("F3", "hole", face, dict(u=45.0, v=14.0, d=5.0,
                                             depth=12.0)),
        ]
    return Task(seed=0, split="manual", material="AL6061", features=feats)


for face in ("Z+", "Z-", "X+", "X-", "Y+", "Y-"):
    task = single_face_task(face)
    for planner in (template_plan, hack_plan):
        compare(task, planner(task), face, f"{face} {planner.__name__}")

print(f"\n{checked} plan/engine comparisons; "
      f"{'ALL EXACT' if worst == 0 else f'WORST DIFF {worst} cells'}")
sys.exit(0 if worst == 0 else 1)
