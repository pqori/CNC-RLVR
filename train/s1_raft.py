"""S1: RAFT -- reward-ranked fine-tuning. RL in slow motion.

Per round: sample K plans per fresh task from the current policy, keep
the argmax-reward plan iff feasible AND clean, SFT one epoch on the
keepers, re-evaluate on val. Prints the go/no-go gate: >= +8pp ok% over
S0 on val says verifier signal moves the policy.

Fresh task seeds per round (20000 + 2000*round) -- never benchmark
seeds. Rewards are normalized by the TEACHER's time on each task.

Each stage (sampling / SFT / eval) runs in its own subprocess so vLLM
and trainer VRAM are fully released between stages (a 48 GB box cannot
hold both at once).

Usage:
  python train/s1_raft.py --model runs/s0_merged --base_ok 0.62 --rounds 3
  (internal)  python train/s1_raft.py --sample_only ... --out keepers.jsonl
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
from cnc.ir import SCHEMA, validate_syntax                    # noqa: E402
from cnc.world import make_task                               # noqa: E402
from cnc.parverify import ParallelVerifier                    # noqa: E402
from train.common import (to_messages, max_sequence_tokens,  # noqa: E402
                          max_completion_tokens,
                          verify_training_stack)


def sample_only(model_path, seed0, n_tasks, k, out_path, arm, temp=1.0):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=model_path,
              max_model_len=max_sequence_tokens(arm),
              dtype="bfloat16")
    structured = StructuredOutputsParams(json=SCHEMA) if arm == "ir" else None
    sp = SamplingParams(n=k, temperature=temp,
                        max_tokens=max_completion_tokens(arm),
                        structured_outputs=structured)
    tok = llm.get_tokenizer()
    tasks = [make_task(s, "train") for s in range(seed0, seed0 + n_tasks)]
    prompts = [tok.apply_chat_template(
        to_messages(t.prompt(), arm), tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,                          # [VERIFY] Qwen3
    ) for t in tasks]
    outs = llm.generate(prompts, sp)

    # verify all candidates across CPU cores, then group-argmax
    seeds_flat, texts_flat = [], []
    for task, out in zip(tasks, outs):
        for cand in out.outputs:
            seeds_flat.append(task.seed)
            texts_flat.append(cand.text)
    # Reward-ranked self-training faces the same metrology incentive as
    # GRPO: faster coarse-clean candidates receive definitive spot reward.
    pv = ParallelVerifier(mode=arm, frontier_spot=True)
    results = pv.verify_batch(seeds_flat, texts_flat)
    pv.close()

    n_ok = 0
    with open(out_path, "w") as f:
        idx = 0
        for task, out in zip(tasks, outs):
            cand_res = results[idx:idx + len(out.outputs)]
            texts = texts_flat[idx:idx + len(out.outputs)]
            idx += len(out.outputs)
            best_i = max(range(len(cand_res)),
                         key=lambda i: cand_res[i]["reward"])
            if cand_res[best_i]["ok"]:
                n_ok += 1
                if arm == "gcode":
                    # the keeper IS the raw document the policy wrote
                    content = texts[best_i]
                else:
                    plan, _ = validate_syntax(texts[best_i])
                    content = json.dumps(plan.to_dict(),
                                         separators=(",", ":"))
                f.write(json.dumps({
                    "messages": to_messages(task.prompt(), arm)
                    + [{"role": "assistant", "content": content}]})
                    + "\n")
    print(f"SAMPLE_DONE keep_rate={n_ok/len(tasks):.4f} keepers={n_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged S0 checkpoint")
    ap.add_argument("--base_ok", type=float, default=None,
                    help="S0 val ok fraction (0-1); measured if omitted")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tasks_per_round", type=int, default=1000)
    ap.add_argument("--arm", choices=["gcode", "ir"], default="gcode")
    # internal subprocess mode
    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--n_tasks", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    verify_training_stack()

    if args.sample_only:
        sample_only(args.model, args.seed0, args.n_tasks, args.k, args.out,
                    args.arm)
        return

    def run(cmd):
        print("+", " ".join(cmd), flush=True)
        return subprocess.run(cmd, check=True, capture_output=True,
                              text=True).stdout

    def val_ok(model_path, name):
        out = run([sys.executable, "train/eval_policy.py", "--model",
                   model_path, "--split", "val", "--k", "1",
                   "--arm", args.arm, "--name", name])
        print(out)
        with open(f"results/{name}_summary.json") as f:
            return json.load(f)["ok_train_pct"] / 100.0

    base_ok = args.base_ok
    if base_ok is None:
        base_ok = val_ok(args.model, "s0_val")
        print(f"[S0 baseline] val ok {100*base_ok:.1f}%")

    current = args.model
    for rnd in range(args.rounds):
        out_dir = f"runs/raft_r{rnd}"
        os.makedirs(out_dir, exist_ok=True)
        keepers = f"{out_dir}/keepers.jsonl"
        print(run([sys.executable, __file__, "--sample_only",
                   "--model", current, "--arm", args.arm,
                   "--seed0", str(20000 + 2000 * rnd),
                   "--n_tasks", str(args.tasks_per_round),
                   "--k", str(args.k), "--out", keepers]))
        run([sys.executable, "train/s0_sft.py", "--model", current,
             "--data", keepers, "--out", out_dir, "--arm", args.arm,
             "--epochs", "1", "--lr", "5e-5"])
        run([sys.executable, "train/merge_lora.py", "--base", current,
             "--adapter", out_dir, "--out", out_dir + "_merged"])
        current = out_dir + "_merged"

        ok_now = val_ok(current, f"raft_r{rnd}_val")
        delta = 100 * (ok_now - base_ok)
        print(f"[GATE] round {rnd}: val ok {100*ok_now:.1f}% "
              f"({delta:+.1f}pp vs S0; gate >= +8pp "
              f"{'PASS' if delta >= 8 else 'not yet'})")


if __name__ == "__main__":
    main()
