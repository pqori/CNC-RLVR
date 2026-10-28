#!/usr/bin/env python3
"""Regrade archived G-code with the included verifier.

This reproduces the audit-plus-spot evaluation without model inference.
All candidates run at 0.2 and 0.1 mm; records marked as spot-checked in
the frozen archive additionally run at 0.05 mm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cnc.sim import Verifier, is_ok, reward_v9  # noqa: E402
from cnc.world import AUDIT_RES, RES, SPOT_RES, TOOLS, make_task  # noqa: E402


def multi_mill_setup(result: dict) -> bool:
    by_setup: dict[int, set[str]] = defaultdict(set)
    for operation in result.get("per_op", []):
        tool_id = operation.get("tool_id")
        tool = TOOLS.get(tool_id)
        if (
            tool is not None
            and tool.kind == "mill"
            and operation.get("removed", 0) > 0
        ):
            by_setup[int(operation["setup_i"])].add(tool_id)
    return any(len(tools) >= 2 for tools in by_setup.values())


def pass_at(groups: list[list[dict]], k: int, key: str) -> float | None:
    estimates = []
    for group in groups:
        n = len(group)
        if n < k:
            continue
        successes = sum(bool(row[key]) for row in group)
        miss = 0.0 if n - successes < k else (
            math.comb(n - successes, k) / math.comb(n, k)
        )
        estimates.append(1.0 - miss)
    return 100.0 * float(np.mean(estimates)) if estimates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["val", "eval_iid", "oot_multitool", "oot_setups"],
        required=True,
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="optional smoke-test limit; 0 regrades the full archive",
    )
    args = parser.parse_args()

    benchmark = {
        int(row["seed"]): row
        for row in (
            json.loads(line)
            for line in (ROOT / "data" / f"{args.split}.jsonl")
            .read_text()
            .splitlines()
        )
    }
    archived = [
        json.loads(line) for line in args.archive.read_text().splitlines()
    ]
    if args.limit:
        archived = archived[:args.limit]
    gen_split = (
        "train" if args.split in {"val", "eval_iid"} else args.split
    )

    regraded = []
    for index, row in enumerate(archived, start=1):
        seed = int(row["seed"])
        task = make_task(seed, gen_split)
        train = Verifier(task, RES).verify_gcode(row["text"])
        audit = Verifier(task, AUDIT_RES).verify_gcode(row["text"])
        final = audit
        if row.get("spot_checked", False):
            final = Verifier(task, SPOT_RES).verify_gcode(row["text"])
        t_ref = float(benchmark[seed]["t_ref_s"])
        regraded.append({
            "seed": seed,
            "candidate_index": int(row.get("candidate_index", 0)),
            "path_valid": bool(train.get("path_valid", False)),
            "ok_train": is_ok(train),
            "ok_audit": is_ok(audit),
            "ok_ladder": is_ok(audit) and is_ok(final),
            "reward_train": reward_v9(train, t_ref),
            "reward_audit": reward_v9(audit, t_ref),
            "reward_ladder": reward_v9(final, t_ref),
            "fail": train.get("fail"),
            "multi_mill_setup": multi_mill_setup(train),
            "fewer_setups": bool(
                train["feasible"]
                and train["n_setups"] < benchmark[seed]["n_setups_teacher"]
            ),
        })
        if index % 25 == 0 or index == len(archived):
            print(f"regraded {index}/{len(archived)}", flush=True)

    grouped_map: dict[int, list[dict]] = defaultdict(list)
    for row in regraded:
        grouped_map[row["seed"]].append(row)
    groups = [grouped_map[seed] for seed in sorted(grouped_map)]
    selected = [max(group, key=lambda row: row["reward_train"]) for group in groups]
    pct = lambda key: 100.0 * float(np.mean([row[key] for row in selected]))
    all_rows = [row for group in groups for row in group]

    print(f"tasks: {len(groups)}; candidates: {len(all_rows)}")
    print(
        "path-valid: "
        f"{100.0 * np.mean([row['path_valid'] for row in all_rows]):.1f}%"
    )
    print(f"final success: {pct('ok_ladder'):.1f}%")
    print(
        "mean final reward: "
        f"{np.mean([row['reward_ladder'] for row in selected]):.3f}"
    )
    print(f"two-mill proxy: {pct('multi_mill_setup'):.1f}%")
    print(f"fewer-setups proxy: {pct('fewer_setups'):.1f}%")
    failures = Counter(
        (row["fail"] or "").split(":")[0]
        for row in selected if row["fail"]
    )
    print(f"failure modes: {dict(failures)}")
    for k in range(1, max(len(group) for group in groups) + 1):
        estimate = pass_at(groups, k, "ok_ladder")
        if estimate is not None:
            print(f"pass@{k}: {estimate:.1f}%")


if __name__ == "__main__":
    main()
