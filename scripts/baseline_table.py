"""Scripted-baseline sweep: the reference table before any training.

For each split: teacher feasibility/ok rate and, on the OOT exams, the
gap between the teacher and the reference strategy it never demonstrates
(multitool / grouped setups). Also oracle stepdown/stepover search on a
subset: how much of the gap is engagement geometry vs composition.
"""

import sys, time
import numpy as np
sys.path.insert(0, ".")

from cnc.world import make_task
from cnc.sim import Verifier, reward_v9, is_ok
from cnc.planner import (template_plan, grouped_plan, multitool_plan,
                         oracle_search)

SPLITS = [
    ("train",         range(0, 40),      template_plan, None),
    ("eval_iid",      range(8000, 8040), template_plan, None),
    ("oot_multitool", range(9000, 9020), template_plan, multitool_plan),
    ("oot_setups",    range(9500, 9520), template_plan, grouped_plan),
]

t_start = time.time()
for split, seeds, teacher_fn, ref_fn in SPLITS:
    rows = []
    for seed in seeds:
        task = make_task(seed, split)
        ver = Verifier(task)
        tres = ver.verify(teacher_fn(task))
        rres = ver.verify(ref_fn(task)) if ref_fn is not None else None
        # time reference: same ladder as the freeze -- teacher if
        # feasible, else the reference strategy (oot_multitool: the
        # teacher snaps the tool, the multitool plan sets the scale)
        if tres["feasible"]:
            t_ref = tres["time_s"]
        elif rres is not None and rres["feasible"]:
            t_ref = rres["time_s"]
        else:
            t_ref = 600.0
        row = dict(seed=seed, t_feas=tres["feasible"], t_ok=is_ok(tres),
                   t_time=tres["time_s"], t_setups=tres["n_setups"],
                   t_reward=reward_v9(tres, t_ref), t_fail=tres["fail"])
        if rres is not None:
            row.update(r_feas=rres["feasible"], r_ok=is_ok(rres),
                       r_time=rres["time_s"], r_setups=rres["n_setups"],
                       r_reward=reward_v9(rres, t_ref), r_fail=rres["fail"])
        if seed % 4 == 0 and split.startswith("oot"):
            _, o_r, o_res = oracle_search(task, ver, t_ref, k=16, seed=seed)
            row.update(o_reward=o_r, o_time=o_res["time_s"],
                       o_feas=o_res["feasible"], o_ok=is_ok(o_res))
        rows.append(row)

    n = len(rows)
    feas = sum(r["t_feas"] for r in rows)
    ok = sum(r["t_ok"] for r in rows)
    print(f"\n=== {split} (n={n}) ===")
    print(f"teacher: feasible {feas}/{n}  ok {ok}/{n}  "
          f"mean_reward {np.mean([r['t_reward'] for r in rows]):.3f}  "
          f"mean_setups {np.mean([r['t_setups'] for r in rows]):.2f}")
    fails = [(r["seed"], r["t_fail"]) for r in rows if not r["t_feas"]]
    if fails:
        print(f"  teacher failures: {fails}")
    if ref_fn is not None:
        print(f"reference: ok {sum(r['r_ok'] for r in rows)}/{n}  "
              f"mean_reward {np.mean([r['r_reward'] for r in rows]):.3f}  "
              f"mean_setups {np.mean([r['r_setups'] for r in rows]):.2f}")
        ok_pairs = [r for r in rows if r["t_feas"] and r.get("r_feas")]
        if ok_pairs:
            regret = np.mean([r["t_time"] / r["r_time"] - 1.0
                              for r in ok_pairs])
            print(f"teacher time regret vs reference: {100*regret:.0f}%")
        rfails = [(r["seed"], r.get("r_fail")) for r in rows
                  if not r.get("r_feas")]
        if rfails:
            print(f"  reference failures: {rfails}")
        oracle_rows = [r for r in rows if "o_reward" in r]
        if oracle_rows:
            print(f"oracle16 (stepdown/stepover search, "
                  f"subset n={len(oracle_rows)}): "
                  f"mean_reward {np.mean([r['o_reward'] for r in oracle_rows]):.3f}  "
                  f"oracle feasible on "
                           f"{sum(1 for r in oracle_rows if r['o_feas'])}"
                  f"/{len(oracle_rows)}")
            fpairs = [r for r in oracle_rows if r["t_feas"]]
            if fpairs:
                o_regret = np.mean([r["t_time"] / max(r["o_time"], 1e-9) - 1.0
                                    for r in fpairs])
                print(f"teacher regret vs oracle: {100*o_regret:.0f}%")

print(f"\ntotal wall: {time.time()-t_start:.0f}s")
