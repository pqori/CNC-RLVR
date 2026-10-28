"""CPU-only calibration of the continuous gouge penalty.

The paired intervention compares a conventional rough-plus-finish pocket
against one-pass shortcuts that omit the finish pass and intentionally cut
past the nominal wall.  For each geometry, the best safe strategy must
weakly dominate every quality-complete, load-safe gouging shortcut:

    lambda >= max_short max(0, base_short - best_safe) / phi(gouge_depth_mm)

The base reward is reward_v8 before the gouge term. Calibration and held-out
geometries are disjoint. The selected coefficient is the smallest 0.01-grid
value above a 10% margin on the worst calibration pair.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from cnc.ir import Op, Plan, Setup
from cnc.sim import (GOUGE_DEPTH_SCALE_MM, GOUGE_PHI_CAP, OK_CLEARANCE,
                     OK_LINF, W_GOUGE, Verifier, gouge_phi, raw_reward_v8)
from cnc.world import CUT_PARAMS, RES, Feature, Task


STEPDOWNS = (0.45, 0.75, 1.05, 1.35)
STEPOVERS = (0.30, 0.50, 0.70, 0.90)
CAL_A_DEPTHS = (3.0, 5.0, 7.0, 9.0)
CAL_A_SIZES = ((20.0, 18.0), (28.0, 24.0), (36.0, 30.0))
CAL_B_DEPTHS = (4.0, 6.0, 8.0, 10.0)
CAL_B_SIZES = ((22.0, 20.0), (31.0, 26.0), (40.0, 34.0))
HELD_DEPTHS = (3.5, 5.5, 7.5, 9.5)
HELD_SIZES = ((25.0, 21.0), (34.0, 28.0), (43.0, 36.0))
OVERCUTS = (0.25, 0.35, 0.50, 0.80)


def plan_for(stepdown: float, stepover: float,
             overcut_mm: float | None) -> Plan:
    fixed = CUT_PARAMS["EM6"]
    if overcut_mm is None:
        ops = [
            Op("pocket_rough", "F1", "EM6", fixed.rpm, fixed.feed,
               stepover, stepdown, "helix", 0.30),
            Op("pocket_finish", "F1", "EM6", fixed.rpm, fixed.feed,
               stepover, stepdown, "helix", 0.0),
        ]
    else:
        ops = [
            Op("pocket_rough", "F1", "EM6", fixed.rpm, fixed.feed,
               stepover, stepdown, "helix", -overcut_mm),
        ]
    return Plan([Setup("Z+", ops)])


def quality_complete(result: dict) -> bool:
    return (result["path_valid"]
            and result["load_safe"]
            and result["ramp_safe"]
            and result["reach_safe"]
            and result["clearance"] >= OK_CLEARANCE
            and result["linf"] <= OK_LINF)


def case_record(depth: float, width: float, height: float,
                case_i: int) -> dict | None:
    feature = Feature(
        "F1", "pocket_rect", "Z+",
        dict(cx=30.0, cy=30.0, w=width, h=height, depth=depth,
             corner_r=4.0),
    )
    task = Task(seed=300_000 + case_i, split="gouge_calibration",
                material="AL6061", features=[feature])
    verifier = Verifier(task)
    safe = []
    unsafe = []
    depth_by_overcut = {}

    for stepdown in STEPDOWNS:
        for stepover in STEPOVERS:
            result = verifier.verify(plan_for(stepdown, stepover, None))
            if quality_complete(result) and result["gouge_safe"]:
                safe.append({
                    "stepdown": stepdown,
                    "stepover": stepover,
                    "result": result,
                })
            for overcut in OVERCUTS:
                measure_depth = overcut not in depth_by_overcut
                result = verifier.verify(
                    plan_for(stepdown, stepover, overcut),
                    measure_gouge_depth=measure_depth)
                if not result["gouge_safe"]:
                    if measure_depth:
                        depth_by_overcut[overcut] = \
                            result["gouge_depth_mm"]
                    else:
                        result["gouge_depth_mm"] = \
                            depth_by_overcut[overcut]
                if quality_complete(result) and not result["gouge_safe"]:
                    unsafe.append({
                        "stepdown": stepdown,
                        "stepover": stepover,
                        "overcut_mm": overcut,
                        "result": result,
                    })

    if not safe or not unsafe:
        return None

    t_ref = min(c["result"]["time_s"] for c in safe)
    for candidate in (*safe, *unsafe):
        candidate["base"] = raw_reward_v8(candidate["result"], t_ref)
    anchor = max(safe, key=lambda c: c["base"])

    rows = []
    for candidate in unsafe:
        depth = candidate["result"]["gouge_depth_mm"]
        phi = gouge_phi(depth)
        required = max(0.0, candidate["base"] - anchor["base"]) \
            / max(phi, 1e-12)
        rows.append({
            "safe_base": anchor["base"],
            "unsafe_base": candidate["base"],
            "gouge_depth_mm": depth,
            "gouge_ratio": candidate["result"]["gouge_ratio"],
            "gouge_volume_mm3":
                candidate["result"]["gouge_volume_mm3"],
            "phi": phi,
            "required_lambda": required,
            "overcut_mm": candidate["overcut_mm"],
            "safe_stepdown": anchor["stepdown"],
            "safe_stepover": anchor["stepover"],
            "unsafe_stepdown": candidate["stepdown"],
            "unsafe_stepover": candidate["stepover"],
        })

    return {
        "case": f"d{depth:g}-{width:g}x{height:g}",
        "rows": rows,
        "required_lambda": max(r["required_lambda"] for r in rows),
    }


def build(depths: tuple[float, ...], sizes: tuple[tuple[float, float], ...],
          start_i: int, workers: int) -> list[dict]:
    cases = []
    case_i = start_i
    for depth in depths:
        for width, height in sizes:
            cases.append((depth, width, height, case_i))
            case_i += 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = pool.map(_case_from_tuple, cases)
        return [record for record in records if record is not None]


def _case_from_tuple(args: tuple[float, float, float, int]) -> dict | None:
    return case_record(*args)


def dominance(records: list[dict], lam: float) -> tuple[int, int]:
    rows = [row for record in records for row in record["rows"]]
    passed = sum(
        row["safe_base"] >= row["unsafe_base"] - lam * row["phi"] - 1e-12
        for row in rows
    )
    return passed, len(rows)


def summarize(name: str, records: list[dict], lam: float) -> list[str]:
    case_req = np.array([r["required_lambda"] for r in records])
    pair_req = np.array([
        row["required_lambda"] for record in records for row in record["rows"]
    ])
    depths = np.array([
        row["gouge_depth_mm"] for record in records for row in record["rows"]
    ])
    lines = [
        f"{name}: {len(records)} cases, {len(pair_req)} informative pairs",
        ("  case required lambda: "
         f"median={np.median(case_req):.6f} "
         f"p90={np.percentile(case_req, 90):.6f} "
         f"max={case_req.max():.6f}"),
        ("  pair required lambda: "
         f"median={np.median(pair_req):.6f} "
         f"p99={np.percentile(pair_req, 99):.6f} "
         f"max={pair_req.max():.6f}"),
        ("  gouge excess depth mm: "
         f"min={depths.min():.6f} median={np.median(depths):.6f} "
         f"max={depths.max():.6f}"),
    ]
    for factor in (0.5, 1.0, 2.0):
        passed, total = dominance(records, factor * lam)
        lines.append(
            f"  dominance @{factor:g}x ({factor * lam:.3f}): "
            f"{passed}/{total} ({100.0 * passed / max(total, 1):.1f}%)"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/receipts/gouge_depth_lambda_calibration_20260825.txt",
    )
    parser.add_argument("--workers", type=int,
                        default=min(2, os.cpu_count() or 1),
                        help="Keep low: each verifier owns dense 3-D arrays")
    args = parser.parse_args()
    started = time.time()

    calibration = (
        build(CAL_A_DEPTHS, CAL_A_SIZES, 0, args.workers)
        + build(CAL_B_DEPTHS, CAL_B_SIZES, 10_000, args.workers)
    )
    held_out = build(HELD_DEPTHS, HELD_SIZES, 20_000, args.workers)
    if not calibration:
        raise SystemExit("no informative calibration pairs")

    empirical = max(r["required_lambda"] for r in calibration)
    selected = math.ceil((1.10 * empirical - 1e-12) / 0.01) * 0.01
    selected = max(0.01, selected)
    if abs(selected - W_GOUGE) > 1e-12:
        raise SystemExit(
            f"calibrated lambda {selected:.3f} != W_GOUGE {W_GOUGE:.3f}")
    lines = [
        "CNC-RLVR gouge-lambda calibration",
        (f"shape: phi=min({GOUGE_PHI_CAP:g}, "
         f"gouge_depth_mm/{GOUGE_DEPTH_SCALE_MM:g})"),
        "gouge_depth_mm = maximum physical penetration beyond tolerance",
        "zero penalty through the analytic tolerance envelope",
        f"calibration_grid_mm={RES}",
        f"stepdowns={STEPDOWNS}",
        f"stepovers={STEPOVERS}",
        f"overcuts_mm={OVERCUTS}",
        "selection: ceil_0.01(1.10 * calibration max)",
        f"empirical_min={empirical:.6f}",
        f"selected_lambda={selected:.3f}",
        "",
        *summarize("calibration", calibration, selected),
        "",
        *summarize("held-out synthetic validation", held_out, selected),
        "",
        f"wall_seconds={time.time() - started:.1f}",
        "No benchmark/model outputs or model APIs were used.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
