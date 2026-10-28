"""CPU-only calibration of the continuous flute-reach penalty.

The intervention is deliberately paired: an EM3 and EM3L execute the same
path family, at the same RPM/feed and engagement, on the same tight-corner
pocket.  Their diameters and cutting schedule are identical; only flute
length and force capacity differ.  EM3 is reach-unsafe above 8 mm while
EM3L is safe through 14 mm.  This isolates the reward advantage an
overreaching short/stiff tool can obtain from its higher load capacity.

For every geometry, compare the best quality-complete, load-safe EM3L
candidate against every quality-complete, load-safe overreaching EM3
candidate across the engagement grid.  This deliberately allows the stiff
short tool to take a more aggressive/faster cut than the safe long tool.
Compute the smallest lambda making the best safe candidate weakly dominate:

    lambda >= max_short max(0, base_short - best_safe) / phi(reach_util_short)

Depth/geometry cases are partitioned before selection.  The selected value
is the smallest 0.01-grid coefficient above a 10% calibration margin.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, ".")

from cnc.ir import Op, Plan, Setup
from cnc.sim import OK_CLEARANCE, OK_LINF, Verifier, raw_reward_v7
from cnc.world import CUT_PARAMS, Feature, Task


REACH_EXCESS_SCALE = 0.25
REACH_PHI_CAP = 20.0
STEPDOWNS = (0.35, 0.55, 0.75, 1.05, 1.40)
STEPovers = (0.30, 0.50, 0.70, 0.90)
CAL_DEPTHS = (8.2, 8.8, 9.4, 10.0, 10.6, 11.2)
HELD_DEPTHS = (8.5, 9.1, 9.7, 10.3, 10.9, 11.5, 12.1, 12.7, 13.3)
SIZES = ((20.0, 18.0), (26.0, 22.0), (32.0, 28.0))


def reach_phi(util: float) -> float:
    """Linear hinge: zero through boundary, unit severity at 25% excess.

    A quadratic onset has zero slope at u=1 and therefore needs a
    pathological coefficient to outweigh a meaningful shortcut at only
    2.5% overreach.  The verifier is discrete and does not consume reward
    derivatives, so the continuous linear hinge is the appropriate shape.
    """
    z = max(0.0, util - 1.0) / REACH_EXCESS_SCALE
    return min(z, REACH_PHI_CAP)


def plan_for(tool_id: str, stepdown: float, stepover: float) -> Plan:
    fixed = CUT_PARAMS[tool_id]
    return Plan([Setup("Z+", [
        Op("pocket_rough", "F1", tool_id, fixed.rpm, fixed.feed,
           stepover, stepdown, "helix", 0.3),
        Op("pocket_finish", "F1", tool_id, fixed.rpm, fixed.feed,
           stepover, stepdown, "helix", 0.0),
    ])])


def quality_complete(result: dict) -> bool:
    return (result["path_valid"]
            and result.get("ramp_safe", True)
            and result["clearance"] >= OK_CLEARANCE
            and result["linf"] <= OK_LINF)


def case_record(depth: float, width: float, height: float,
                case_i: int) -> dict | None:
    feature = Feature(
        "F1", "pocket_rect", "Z+",
        dict(cx=30.0, cy=30.0, w=width, h=height, depth=depth,
             corner_r=2.0),
    )
    task = Task(seed=200_000 + case_i, split="reach_calibration",
                material="AL6061", features=[feature])
    verifier = Verifier(task)
    candidates = []

    for stepdown in STEPDOWNS:
        for stepover in STEPovers:
            for tool_id in ("EM3L", "EM3"):
                result = verifier.verify(
                    plan_for(tool_id, stepdown, stepover))
                candidates.append({
                    "tool_id": tool_id,
                    "stepdown": stepdown,
                    "stepover": stepover,
                    "result": result,
                })

    # One common reference time is necessary for cross-engagement ranking.
    safe_results = [
        c for c in candidates
        if c["tool_id"] == "EM3L"
        and quality_complete(c["result"])
        and c["result"]["load_safe"]
        and c["result"]["reach_safe"]
    ]
    unsafe_results = [
        c for c in candidates
        if c["tool_id"] == "EM3"
        and quality_complete(c["result"])
        and c["result"]["load_safe"]
        and not c["result"]["reach_safe"]
    ]
    if not safe_results or not unsafe_results:
        return None

    # Use the fastest safe candidate as a fixed task reference, matching the
    # benchmark's reference-ladder semantics. raw_reward_v7 retains all
    # load/ramp/quality terms but deliberately excludes reach.
    t_ref = min(c["result"]["time_s"] for c in safe_results)
    for c in candidates:
        c["base"] = raw_reward_v7(c["result"], t_ref)
    anchor = max(safe_results, key=lambda c: c["base"])

    rows = []
    for c in unsafe_results:
        result = c["result"]
        util = result["reach_util"]
        phi = reach_phi(util)
        required = max(0.0, c["base"] - anchor["base"]) \
            / max(phi, 1e-12)
        rows.append({
            "stepdown": c["stepdown"],
            "stepover": c["stepover"],
            "safe_base": anchor["base"],
            "unsafe_base": c["base"],
            "reach_util": util,
            "phi": phi,
            "required_lambda": required,
            "safe_stepdown": anchor["stepdown"],
            "safe_stepover": anchor["stepover"],
            "safe_load_util": max(anchor["result"]["force_util"],
                                  anchor["result"]["power_util"]),
            "unsafe_load_util": max(result["force_util"],
                                    result["power_util"]),
        })

    return {
        "case": f"d{depth:g}-{width:g}x{height:g}",
        "depth": depth,
        "rows": rows,
        "required_lambda": max(r["required_lambda"] for r in rows),
    }


def dominance(records: list[dict], lam: float) -> tuple[int, int]:
    passed = 0
    total = 0
    for rec in records:
        for row in rec["rows"]:
            total += 1
            safe = row["safe_base"]
            unsafe = row["unsafe_base"] - lam * row["phi"]
            passed += safe >= unsafe - 1e-12
    return passed, total


def summarize(name: str, records: list[dict], lam: float) -> list[str]:
    req = np.array([r["required_lambda"] for r in records])
    pair_req = np.array([
        row["required_lambda"] for rec in records for row in rec["rows"]
    ])
    lines = [
        f"{name}: {len(records)} cases, {len(pair_req)} informative pairs",
        ("  case required lambda: "
         f"median={np.median(req):.6f} "
         f"p90={np.percentile(req, 90):.6f} "
         f"max={req.max():.6f}"),
        ("  pair required lambda: "
         f"median={np.median(pair_req):.6f} "
         f"p99={np.percentile(pair_req, 99):.6f} "
         f"max={pair_req.max():.6f}"),
    ]
    for factor in (0.5, 1.0, 2.0):
        passed, total = dominance(records, factor * lam)
        lines.append(
            f"  dominance @{factor:g}x ({factor * lam:.3f}): "
            f"{passed}/{total} ({100.0 * passed / max(total, 1):.1f}%)"
        )
    return lines


def build(depths: tuple[float, ...], start_i: int) -> list[dict]:
    records = []
    case_i = start_i
    for depth in depths:
        for width, height in SIZES:
            rec = case_record(depth, width, height, case_i)
            case_i += 1
            if rec is not None:
                records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="results/receipts/reach_lambda_calibration_20260825.txt",
    )
    args = ap.parse_args()
    t0 = time.time()

    calibration = build(CAL_DEPTHS, 0)
    held_out = build(HELD_DEPTHS, 10_000)
    if not calibration:
        raise SystemExit("no informative calibration pairs")

    empirical = max(r["required_lambda"] for r in calibration)
    selected = math.ceil((1.10 * empirical - 1e-12) / 0.01) * 0.01
    selected = max(0.01, selected)

    lines = [
        "CNC-RLVR reach-lambda calibration",
        ("shape: z=max(0,u-1)/0.25; "
         f"phi=min({REACH_PHI_CAP:g},z)"),
        "zero penalty for every u<=1; phi(1.25)=1",
        f"stepdowns={STEPDOWNS}",
        f"stepovers={STEPovers}",
        f"selection: ceil_0.01(1.10 * calibration max)",
        f"empirical_min={empirical:.6f}",
        f"selected_lambda={selected:.3f}",
        "",
        *summarize("calibration", calibration, selected),
        "",
        *summarize("held-out synthetic validation", held_out, selected),
        "",
        f"wall_seconds={time.time() - t0:.1f}",
        "No benchmark/model outputs or model APIs were used.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
