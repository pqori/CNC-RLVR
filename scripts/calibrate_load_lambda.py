"""CPU-only calibration of the continuous load-penalty coefficient.

The calibration uses paired counterfactual plans on the frozen validation
split.  Tool, path family, feed and RPM stay fixed; only axial and radial
engagement vary.  Load is measured in Verifier's diagnostic shadow mode so
an overload does not truncate the trajectory and bias its time/quality.

The reported coefficient is the smallest 0.05-grid value, with a 10% margin,
for which each observed load-unsafe candidate is dominated by a robust-safe
(u <= U0) candidate for the same task.  The final quarter of validation tasks
is held out from coefficient selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, ".")

from cnc.ir import Op, Plan, Setup
from cnc.planner import template_plan
from cnc.sim import (LOAD_MARGIN_UTIL, LOAD_PHI_CAP, OK_CLEARANCE, OK_LINF,
                     W_LOAD, Verifier, load_phi, raw_reward_v6)
from cnc.world import CUT_PARAMS, TOOLS, Feature, Task, make_task


U0 = LOAD_MARGIN_UTIL
SD_MULTS = (0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
STEPovers = (0.30, 0.50, 0.70, 0.90)


def variant(base: Plan, task, sd_mult: float, stepover: float) -> Plan:
    plan = Plan.from_dict(base.to_dict())
    for setup in plan.setups:
        for op in setup.ops:
            tool = TOOLS[op.tool_id]
            # Perturb material-removal engagement, not the fixed finishing
            # pass.  Enlarging a finish stepdown changes its entry topology
            # and would no longer be a paired roughing intervention.
            if tool.kind != "mill" or op.op_type == "pocket_finish":
                continue
            feat = task.feature(op.feature_ref)
            depth = feat.params["depth"]
            op.stepdown = round(min(depth, tool.flute_len,
                                    max(0.1, op.stepdown * sd_mult)), 2)
            op.stepover = stepover
    return plan


def quality_ok(result: dict) -> bool:
    return (result["path_valid"]
            and result["clearance"] >= OK_CLEARANCE
            and result["linf"] <= OK_LINF)


def grade(verifier: Verifier, plan: Plan, t_ref_s: float) -> dict:
    result = verifier.verify(plan)
    util = max(result["force_util"], result["power_util"])
    phi = load_phi(util)
    return {
        # Remove v6's selected load term from the raw additive objective.
        # The final affine map preserves this calibration's ordering but is
        # intentionally excluded from lambda's units.
        "base": raw_reward_v6(result, t_ref_s) + W_LOAD * phi,
        "util": util,
        "phi": phi,
        "quality_ok": quality_ok(result),
        "load_safe": util <= 1.0 + 1e-12,
        "robust_safe": util <= U0 + 1e-12,
    }


def task_record(rec: dict) -> dict | None:
    task = make_task(rec["seed"], "train")
    verifier = Verifier(task)
    base = template_plan(task)
    rows = []
    for sd_mult in SD_MULTS:
        for stepover in STEPovers:
            row = grade(verifier, variant(base, task, sd_mult, stepover),
                        rec["t_ref_s"])
            row.update(sd_mult=sd_mult, stepover=stepover)
            rows.append(row)

    robust = [r for r in rows if r["quality_ok"] and r["robust_safe"]]
    unsafe = [r for r in rows if r["quality_ok"] and not r["load_safe"]]
    safe = [r for r in rows if r["quality_ok"] and r["load_safe"]]
    if not robust or not unsafe or not safe:
        return None

    anchor = max(robust, key=lambda r: r["base"])
    required = max(
        (max(0.0, r["base"] - anchor["base"]) / r["phi"]
         for r in unsafe),
        default=0.0,
    )
    return {
        "seed": rec["seed"],
        "required_lambda": required,
        "anchor_base": anchor["base"],
        "anchor_util": anchor["util"],
        "max_unsafe_base": max(r["base"] for r in unsafe),
        "max_util": max(r["util"] for r in rows),
        "rows": rows,
    }


def synthetic_record(tool_id: str, depth: float, case_i: int) -> dict | None:
    """Targeted load probe, generated outside every benchmark split."""
    tool = TOOLS[tool_id]
    width = min(48.0, max(22.0, 4.0 * tool.diameter))
    height = width - 2.0
    feature = Feature(
        "F1", "pocket_rect", "Z+",
        dict(cx=30.0, cy=30.0, w=width, h=height, depth=depth,
             corner_r=tool.diameter / 2.0),
    )
    task = Task(seed=100_000 + case_i, split="load_calibration",
                material="AL6061", features=[feature])
    fixed = CUT_PARAMS[tool_id]
    base = Plan([Setup("Z+", [
        Op("pocket_rough", "F1", tool_id, fixed.rpm, fixed.feed,
           0.5, min(depth, 0.35 * tool.diameter), "helix", 0.3),
        Op("pocket_finish", "F1", tool_id, fixed.rpm, fixed.feed,
           0.5, min(depth, 0.70 * tool.diameter), "helix", 0.0),
    ])])
    verifier = Verifier(task)
    reference = verifier.verify(variant(base, task, 0.5, 0.5))
    if not quality_ok(reference) or not reference["load_safe"]:
        return None
    t_ref_s = reference["time_s"]

    rows = []
    for sd_mult in SD_MULTS:
        for stepover in STEPovers:
            row = grade(verifier, variant(base, task, sd_mult, stepover),
                        t_ref_s)
            row.update(sd_mult=sd_mult, stepover=stepover)
            rows.append(row)

    robust = [r for r in rows if r["quality_ok"] and r["robust_safe"]]
    unsafe = [r for r in rows if r["quality_ok"] and not r["load_safe"]]
    safe = [r for r in rows if r["quality_ok"] and r["load_safe"]]
    if not robust or not unsafe or not safe:
        return None
    anchor = max(robust, key=lambda r: r["base"])
    required = max(
        (max(0.0, r["base"] - anchor["base"]) / r["phi"]
         for r in unsafe),
        default=0.0,
    )
    return {
        "seed": f"{tool_id}-d{depth:g}",
        "required_lambda": required,
        "anchor_base": anchor["base"],
        "anchor_util": anchor["util"],
        "max_unsafe_base": max(r["base"] for r in unsafe),
        "max_util": max(r["util"] for r in rows),
        "rows": rows,
    }


def synthetic_battery() -> list[tuple[bool, dict]]:
    """Return (held_out, record) targeted cases across all endmills."""
    depths = {
        "EM3": (4.0, 6.5),
        "EM3L": (8.0, 11.0, 13.0),
        "EM6": (6.0, 10.0, 12.0),
        "EM10": (8.0, 13.0, 18.0),
        "EM12": (10.0, 15.0, 19.0),
    }
    records = []
    case_i = 0
    for tool_id, tool_depths in depths.items():
        for depth_i, depth in enumerate(tool_depths):
            result = synthetic_record(tool_id, depth, case_i)
            case_i += 1
            if result is not None:
                records.append((depth_i == len(tool_depths) - 1, result))
    return records


def dominance(records: list[dict], lam: float) -> tuple[int, int]:
    passed = 0
    for rec in records:
        safe = [r["base"] - lam * r["phi"] for r in rec["rows"]
                if r["quality_ok"] and r["load_safe"]]
        unsafe = [r["base"] - lam * r["phi"] for r in rec["rows"]
                  if r["quality_ok"] and not r["load_safe"]]
        passed += bool(safe and unsafe and max(safe) >= max(unsafe) - 1e-12)
    return passed, len(records)


def summarize(name: str, records: list[dict], lam: float) -> list[str]:
    req = np.array([r["required_lambda"] for r in records])
    lines = [
        f"{name}: {len(records)} informative tasks",
        ("  required lambda: "
         f"median={np.median(req):.4f} "
         f"p90={np.percentile(req, 90):.4f} "
         f"p99={np.percentile(req, 99):.4f} max={req.max():.4f}"),
    ]
    for factor in (0.5, 1.0, 2.0):
        passed, total = dominance(records, factor * lam)
        lines.append(
            f"  dominance @{factor:g}x ({factor * lam:.3f}): "
            f"{passed}/{total} ({100.0 * passed / max(total, 1):.1f}%)"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/val.jsonl")
    ap.add_argument("--calibration-n", type=int, default=150)
    ap.add_argument("--validation-n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--output",
                    default="results/receipts/load_lambda_calibration_20260824.txt")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")

    with open(args.input) as f:
        source = [json.loads(line) for line in f]
    requested = args.calibration_n + args.validation_n
    if requested > len(source):
        raise SystemExit(f"requested {requested} tasks, found {len(source)}")

    t0 = time.time()
    records = []
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx) as pool:
        for i, result in enumerate(
                pool.map(task_record, source[:requested], chunksize=1),
                start=1):
            if result is not None:
                records.append((i - 1, result))
            if i % 25 == 0:
                print(
                    f"graded {i}/{requested} tasks "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    calibration = [r for i, r in records if i < args.calibration_n]
    validation = [r for i, r in records if i >= args.calibration_n]
    synthetic = synthetic_battery()
    calibration += [r for held_out, r in synthetic if not held_out]
    validation += [r for held_out, r in synthetic if held_out]
    if not calibration:
        raise SystemExit("no calibration task had both robust-safe and unsafe variants")

    empirical = max(r["required_lambda"] for r in calibration)
    selected = math.ceil((1.10 * empirical - 1e-12) / 0.05) * 0.05
    selected = max(0.05, selected)

    lines = [
        "CNC-RLVR load-lambda calibration",
        f"source={args.input}",
        (f"shape: u0={U0:.2f}; phi=min({LOAD_PHI_CAP:g}, z^2) for "
         f"z<=1, else min({LOAD_PHI_CAP:g}, 2z-1)"),
        f"variants/task={len(SD_MULTS) * len(STEPovers)}",
        f"selection: ceil_0.05(1.10 * calibration max)",
        f"empirical_min={empirical:.6f}",
        f"selected_lambda={selected:.3f}",
        "",
        *summarize("calibration", calibration, selected),
    ]
    if validation:
        lines += ["", *summarize("held-out validation", validation, selected)]
    lines += [
        "",
        f"wall_seconds={time.time() - t0:.1f}",
        "No model APIs were called; this receipt is CPU-simulator only.",
    ]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
