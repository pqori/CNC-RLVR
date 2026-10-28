#!/usr/bin/env python3
"""Reproduce the paper's headline values from archived summaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SPLITS = [
    ("validation", "val"),
    ("IID test", "eval_iid"),
    ("OOT multi-tool", "oot_multitool"),
    ("OOT setups", "oot_setups"),
]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}_summary.json").read_text())


def value(summary: dict, metric: str):
    if metric == "path-valid %":
        return summary["valid_pct"]
    if metric == "reward":
        return summary["mean_reward_ladder"]
    if metric == "success %":
        return summary["ok_ladder_pct"]
    if metric == "gouge / n":
        return summary["failure_modes"].get("gouge", 0)
    if metric == "two-mill %":
        return summary["multi_mill_setup_pct"]
    if metric == "G-code fails":
        return summary["failure_modes"].get("gcode", 0)
    if metric == "fewer-setups %":
        return summary["fewer_setups_pct"]
    raise KeyError(metric)


def fmt(metric: str, x) -> str:
    if metric == "reward":
        return f"{x:.3f}"
    if metric in {"gouge / n", "G-code fails"}:
        return str(int(x))
    return f"{x:.1f}"


def main() -> None:
    print("Headline greedy comparison")
    print("split,metric,SFT greedy,GRPO greedy")
    for label, split in SPLITS:
        sft = load(f"sft8b_{split}_k1")
        grpo = load(f"grpo_step50_{split}_k1")
        metrics = ["path-valid %", "reward", "success %"]
        if split in {"val", "eval_iid"}:
            metrics += ["gouge / n", "two-mill %"]
        elif split == "oot_multitool":
            metrics += ["two-mill %"]
        else:
            metrics += ["G-code fails", "fewer-setups %"]
        for metric in metrics:
            print(
                f"{label},{metric},"
                f"{fmt(metric, value(sft, metric))},"
                f"{fmt(metric, value(grpo, metric))}"
            )

    print("\nSFT four-sample controls under the audit-plus-spot protocol")
    for label, split in SPLITS:
        summary = load(f"sft8b_{split}_k4")
        pass4 = summary["pass_at_k_ladder_pct"]["4"]
        search4 = summary["ok_ladder_pct"]
        print(
            f"{label}: any-of-four pass@4={pass4:.1f}%; "
            f"reward-selected search-4={search4:.1f}%"
        )


if __name__ == "__main__":
    main()
