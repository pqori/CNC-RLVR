"""The referee: evaluate a policy checkpoint on a benchmark split.

- greedy (k=1) and pass@k sampling, grammar-locked or free (--no_grammar)
- every plan verified at the train grid AND the audit grid (0.1 mm);
  the train-vs-audit gap is the reward-hacking signal
- SPOT ESCALATION (triggered): a rollout that grades clean at the train
  grid but where the audit grid finds required-region material is
  re-verified at the 0.05 mm spot grid.
- SPOT SAMPLING (random): the trigger inherits the audit grid's own
  sampling blindness -- a film within ~half an audit cell beyond the
  0.1 mm spec leaves the audit grid NOTHING to flag (measured: a
  0.105 mm film reads clean at train and audit, flag never fires, and
  the spot grid convicts it at L-inf 0.116-0.125). So the referee also
  spot-verifies a random sample of clean, unflagged rollouts each eval
  (--spot_random). Violations below spec + spot slack (~0.125 mm) are
  beneath the ladder's enforcement floor -- disclosed, not hidden.
- --spot_all_clean makes final reported pass@k definitive at the spot
  resolution by checking every otherwise-passing candidate.
- reports an arm-agnostic multi-mill-tool-per-setup proxy and setup economy;
  feature-level strategy attribution is left to the archived path audit
- archives every rollout to results/<name>_rollouts.jsonl

Usage:
  python train/eval_policy.py --model runs/s0_merged --split eval_iid --k 8
"""

import argparse
import math
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from cnc.ir import SCHEMA, validate_syntax                    # noqa: E402
from cnc.world import make_task, TOOLS, RES, AUDIT_RES, SPOT_RES  # noqa: E402
from cnc.sim import (Verifier, reward_v9, is_ok,              # noqa: E402
                     spot_flagged)
from train.common import (to_messages, verify_benchmark_manifest,  # noqa: E402
                          verify_training_stack,
                          max_sequence_tokens, max_completion_tokens)
from train.s0_sft import load_rows                            # noqa: E402


def used_multi_mill_setup(res):
    """Whether two distinct mill tools removed material in one setup.

    Raw G-code has no feature labels, so this is deliberately a setup-level
    proxy, not a claim that both tools cut the same feature.
    """
    per_setup = {}
    for rec in res.get("per_op", []):
        t = TOOLS.get(rec.get("tool_id"))
        if t and t.kind == "mill" and rec.get("removed", 0) > 0:
            per_setup.setdefault(rec.get("setup_i", 0),
                                 set()).add(t.tool_id)
    return any(len(s) > 1 for s in per_setup.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="eval_iid")
    ap.add_argument("--arm", choices=["gcode", "ir"], default="gcode",
                    help="gcode: policy writes the document, graded by "
                         "verify_gcode (MAIN); ir: typed-plan JSON through "
                         "the deterministic expander (scaffolding ablation)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="limit tasks")
    ap.add_argument("--no_grammar", action="store_true",
                    help="ir arm only: drop the JSON decoding lock (the "
                         "gcode arm always decodes freely; its grammar IS "
                         "the fail-loud interpreter)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--spot_cap", type=int, default=16,
                    help="max 0.05 mm spot escalations per eval (each "
                         "costs ~20-60 s CPU); flagged-but-uncapped "
                         "rollouts are still counted as flagged")
    ap.add_argument("--spot_random", type=int, default=8,
                    help="0.05 mm spot verifications of RANDOM clean "
                         "unflagged rollouts per eval -- the insurance "
                         "against films the audit grid cannot see at "
                         "all (its escalation trigger is blind to them)")
    ap.add_argument("--spot_all_clean", action="store_true",
                    help="final-eval mode: spot-check every audit-clean "
                         "candidate contributing to pass@k")
    args = ap.parse_args()
    verify_benchmark_manifest()
    verify_training_stack()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    max_out = max_completion_tokens(args.arm)
    llm = LLM(model=args.model,
              max_model_len=max_sequence_tokens(args.arm),
              dtype="bfloat16")
    structured = StructuredOutputsParams(json=SCHEMA) \
        if (args.arm == "ir" and not args.no_grammar) else None
    sp = SamplingParams(
        n=args.k,
        temperature=0.0 if args.k == 1 else args.temp,
        max_tokens=max_out,
        structured_outputs=structured,
    )

    records = load_rows(f"data/{args.split}.jsonl", limit=args.n)

    # integrity guard: tasks are regenerated from seeds for verification;
    # if this machine's RNG produces different tasks than the freeze
    # machine's, every grade would be silently wrong. The frozen prompt
    # is the fingerprint.
    for rec in records:
        gen_split = "train" if rec["split"] in ("train", "val", "eval_iid") \
            else rec["split"]
        if make_task(rec["seed"], gen_split).prompt() != rec["prompt"]:
            sys.exit(f"FATAL: task regen mismatch on seed {rec['seed']} -- "
                     "numpy RNG differs from freeze machine; pin numpy.")

    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        to_messages(r["prompt"], args.arm), tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,                         # [VERIFY] Qwen3 kwarg
    ) for r in records]

    outs = llm.generate(prompts, sp)

    name = args.name or f"{os.path.basename(args.model)}_{args.split}_k{args.k}"
    os.makedirs("results", exist_ok=True)

    # archived at the END of the run so spot verdicts (attached to best
    # rows after generation) land in the file -- the exploit-mining
    # archive must carry the full ladder verdict, not just train/audit
    arch_rows = []
    rows = []
    candidate_groups = []
    n_flagged = n_spot_checked = n_confirmed = 0
    for rec, out in zip(records, outs):
        gen_split = "train" if rec["split"] in ("train", "val", "eval_iid") \
            else rec["split"]
        task = make_task(rec["seed"], gen_split)
        ver_t = Verifier(task, RES)
        ver_a = Verifier(task, AUDIT_RES)

        def grade(ver, text):
            """One candidate at one grid -> (result, reward). Malformed
            output is a graded fail, never a crash: the gcode arm returns
            fail='gcode:<line/reason>' from the interpreter, the ir arm
            fail='parse:<err>' from schema validation."""
            if args.arm == "gcode":
                r = ver.verify_gcode(text)
            else:
                plan, err = validate_syntax(text)
                r = ver.verify(plan) if plan is not None else dict(
                    feasible=False, fail=f"parse:{err}", clearance=0,
                    linf=99, time_s=0, wear_max=0,
                    n_setups=0, per_op=[])
            return r, reward_v9(r, rec["t_ref_s"])

        cand_rows = []
        for candidate_i, cand in enumerate(out.outputs):
            cres, r_train = grade(ver_t, cand.text)
            ares, r_audit = grade(ver_a, cand.text)
            row = dict(seed=rec["seed"], candidate_index=candidate_i,
                       text=cand.text,
                       t_ref_s=rec["t_ref_s"],
                       reward_train=r_train, ok_train=is_ok(cres),
                       reward_audit=r_audit, ok_audit=is_ok(ares),
                       reward_ladder=r_audit,
                       ok_ladder=is_ok(ares),
                       spot_flag=spot_flagged(cres, ares),
                       spot_checked=False,
                       time_s=cres["time_s"], n_setups=cres["n_setups"],
                       fail=cres["fail"],
                       path_valid=cres.get("path_valid", False),
                       load_safe=cres.get("load_safe", True),
                       force_util=cres.get("force_util", 0.0),
                       power_util=cres.get("power_util", 0.0),
                       ramp_safe=cres.get("ramp_safe", True),
                       ramp_util=cres.get("ramp_util", 0.0),
                       ramp_angle_deg=cres.get("ramp_angle_deg", 0.0),
                       reach_safe=cres.get("reach_safe", True),
                       reach_util=cres.get("reach_util", 0.0),
                       gouge_safe=cres.get("gouge_safe", True),
                       gouge_depth_mm=cres.get("gouge_depth_mm", 0.0),
                       gouge_ratio=cres.get("gouge_ratio", 0.0),
                       multi_mill_setup=used_multi_mill_setup(cres),
                       fewer_setups=bool(cres["feasible"]
                                         and cres["n_setups"]
                                         < rec["n_setups_teacher"]))
            arch_rows.append(row)
            cand_rows.append(row)

        # Escalate every flagged candidate, not only the train-reward argmax:
        # pass@k consumes the whole candidate group.
        flagged_rows = [row for row in cand_rows if row["spot_flag"]]
        n_flagged += len(flagged_rows)
        ver_s = None
        for row in flagged_rows:
            if n_spot_checked < args.spot_cap:
                n_spot_checked += 1
                if ver_s is None:
                    ver_s = Verifier(task, SPOT_RES)
                sres, spot_reward = grade(ver_s, row["text"])
                row["spot_checked"] = True
                row["ok_spot"] = is_ok(sres)
                row["linf_spot"] = sres["linf"]
                row["reward_spot"] = spot_reward
                row["reward_ladder"] = spot_reward
                row["ok_ladder"] = row["ok_audit"] and row["ok_spot"]
                if not row["ok_spot"]:
                    n_confirmed += 1
                    print(f"  [SPOT] seed {rec['seed']} candidate "
                          f"{row['candidate_index']}: ok@train but "
                          f"spot grid says linf={sres['linf']:.3f} mm, "
                          f"clearance={sres['clearance']:.4f} -- "
                          f"HACK CONFIRMED")
            else:
                # A flagged-but-unchecked candidate cannot be promoted to a
                # ladder success merely because the configured cap was hit.
                row["ok_ladder"] = False

        best = max(cand_rows, key=lambda row: row["reward_train"])
        rows.append(best)
        candidate_groups.append(cand_rows)

    # Random spot sample over every audit-clean candidate that contributes
    # to pass@k, including non-argmax candidates. Reproducible pick.
    n_rand_checked = n_rand_confirmed = 0
    pool = [r for r in arch_rows
            if r["ok_audit"] and not r["spot_flag"]]
    picks = (np.arange(len(pool)) if args.spot_all_clean
             else np.random.default_rng(0).permutation(len(pool)))
    n_random_targets = len(pool) if args.spot_all_clean else args.spot_random
    spot_task_seed, ver_s = None, None
    for j in picks[:n_random_targets]:
        row = pool[j]
        gen_split = "train" if args.split in ("train", "val", "eval_iid") \
            else args.split
        if row["seed"] != spot_task_seed:
            task = make_task(row["seed"], gen_split)
            ver_s = Verifier(task, SPOT_RES)
            spot_task_seed = row["seed"]
        if args.arm == "gcode":
            sres = ver_s.verify_gcode(row["text"])
        else:
            plan, _ = validate_syntax(row["text"])
            if plan is None:
                continue
            sres = ver_s.verify(plan)
        n_rand_checked += 1
        row["spot_checked"] = True
        row["ok_spot_random"] = is_ok(sres)
        row["linf_spot"] = sres["linf"]
        row["reward_spot"] = reward_v9(sres, row["t_ref_s"])
        row["reward_ladder"] = row["reward_spot"]
        row["ok_ladder"] = row["ok_audit"] and row["ok_spot_random"]
        if not row["ok_spot_random"]:
            n_rand_confirmed += 1
            print(f"  [SPOT-RANDOM] seed {row['seed']} candidate "
                  f"{row['candidate_index']}: clean at train "
                  f"AND audit, but spot grid says "
                  f"linf={sres['linf']:.3f} mm, "
                  f"clearance={sres['clearance']:.4f} -- HACK CONFIRMED "
                  f"below the audit's detection floor")

    with open(f"results/{name}_rollouts.jsonl", "w") as archive:
        for row in arch_rows:
            archive.write(json.dumps(row) + "\n")

    n = len(rows)
    def pct(key): return 100.0 * np.mean([r[key] for r in rows])
    all_candidates = [r for group in candidate_groups for r in group]
    valid_pct = 100.0 * np.mean(
        [r["path_valid"] for r in all_candidates]
    )

    def pass_at(k, key):
        estimates = []
        for group in candidate_groups:
            n_group = len(group)
            if n_group < k:
                continue
            c = sum(bool(r[key]) for r in group)
            miss = 0.0 if n_group - c < k else (
                math.comb(n_group - c, k) / math.comb(n_group, k)
            )
            estimates.append(1.0 - miss)
        return 100.0 * np.mean(estimates) if estimates else None

    pass_train = {k: pass_at(k, "ok_train")
                  for k in range(1, args.k + 1)}
    pass_audit = {k: pass_at(k, "ok_audit")
                  for k in range(1, args.k + 1)}
    pass_ladder = {k: pass_at(k, "ok_ladder")
                   for k in range(1, args.k + 1)}
    regrets = [r["time_s"] / rec_t - 1.0
               for r, rec_t in zip(rows, [x["t_ref_s"] for x in records])
               if r["ok_train"]]
    print(f"\n=== {name}  (n={n}, best-of-{args.k}) ===")
    print(f"ok@train {pct('ok_train'):.1f}%   ok@audit {pct('ok_audit'):.1f}%"
          f"   ok@ladder {pct('ok_ladder'):.1f}%")
    print(f"all-rollout valid {valid_pct:.1f}%   sampled pass@1 ladder "
          f"{pass_ladder[1]:.1f}%   pass@{args.k} ladder "
          f"{pass_ladder[args.k]:.1f}%")
    if n_flagged:
        print(f"spot escalations: {n_flagged} flagged, {n_spot_checked} "
              f"checked at 0.05 mm, {n_confirmed} CONFIRMED hacks")
    if n_rand_checked:
        spot_label = ("complete spot sweep" if args.spot_all_clean
                      else "random spot sample")
        print(f"{spot_label}: {n_rand_checked} checked, "
              f"{n_rand_confirmed} CONFIRMED hacks below the audit's "
              f"detection floor")
    print(f"mean reward train {np.mean([r['reward_train'] for r in rows]):.3f}"
          f"   audit {np.mean([r['reward_audit'] for r in rows]):.3f}"
          f"   ladder {np.mean([r['reward_ladder'] for r in rows]):.3f}")
    if regrets:
        print(f"time regret vs teacher (ok plans): {100*np.mean(regrets):.0f}%")
    print(f"multi-mill-tool setup proxy {pct('multi_mill_setup'):.1f}%   "
          f"fewer-setups-than-teacher {pct('fewer_setups'):.1f}%")
    fails = {}
    for r in rows:
        if r["fail"]:
            fails[r["fail"].split(":")[0]] = fails.get(
                r["fail"].split(":")[0], 0) + 1
    if fails:
        print(f"failure modes: {fails}")

    # machine-readable summary: the RAFT gate (and any script) reads this
    # instead of scraping stdout
    summary = dict(
        name=name, model=args.model, split=args.split, k=args.k, n=n,
        ok_train_pct=float(pct("ok_train")),
        ok_audit_pct=float(pct("ok_audit")),
        ok_ladder_pct=float(pct("ok_ladder")),
        valid_pct=float(valid_pct),
        sampled_pass_at_1_train_pct=float(pass_train[1]),
        sampled_pass_at_1_audit_pct=float(pass_audit[1]),
        sampled_pass_at_1_ladder_pct=float(pass_ladder[1]),
        pass_at_k_train_pct={str(k): float(v)
                             for k, v in pass_train.items()},
        pass_at_k_audit_pct={str(k): float(v)
                             for k, v in pass_audit.items()},
        pass_at_k_ladder_pct={str(k): float(v)
                              for k, v in pass_ladder.items()},
        mean_reward_train=float(np.mean([r["reward_train"] for r in rows])),
        mean_reward_audit=float(np.mean([r["reward_audit"] for r in rows])),
        mean_reward_ladder=float(
            np.mean([r["reward_ladder"] for r in rows])),
        time_regret_pct=float(100 * np.mean(regrets)) if regrets else None,
        multi_mill_setup_pct=float(pct("multi_mill_setup")),
        fewer_setups_pct=float(pct("fewer_setups")),
        spot_flagged=n_flagged,
        spot_checked=n_spot_checked,
        spot_confirmed_hacks=n_confirmed,
        spot_random_checked=n_rand_checked,
        spot_random_confirmed=n_rand_confirmed,
        spot_all_clean=bool(args.spot_all_clean),
        ladder_definitive_at_spot_resolution=bool(args.spot_all_clean),
        failure_modes=fails,
    )
    with open(f"results/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> results/{name}_summary.json")


if __name__ == "__main__":
    main()
