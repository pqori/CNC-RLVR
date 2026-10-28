"""Frontier-model baseline: zero-shot / best-of-K on the benchmark.

Runs from a laptop (needs ANTHROPIC_API_KEY or OPENAI_API_KEY),
independent of AWS.

Arms match the local policies:
  --arm gcode (MAIN): the model writes the G-code document as plain
      text under the same system prompt the student trains on.
      Markdown code fences are stripped (API models add them); the
      fail-loud interpreter grades everything else. Parse failures
      score -1 and are reported (they are data).
  --arm ir: forced tool-use with the plan JSON schema -- the closest
      equivalent of the grammar lock the local IR arm gets.

Usage:
  python llm_policy.py --split eval_iid --k 1 --n 50
  python llm_policy.py --split oot_multitool --k 8 --arm ir
  python llm_policy.py --provider openai --model gpt-5.6-sol \
      --reasoning-effort none --split oot_multitool --n 10
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from cnc.ir import SCHEMA, validate_syntax
from cnc.world import make_task, RES, AUDIT_RES
from cnc.sim import Verifier, reward_v9, is_ok
from train.common import SYSTEM_GCODE, SYSTEM_IR
from train.eval_policy import used_multi_mill_setup


def strip_fences(text: str) -> str:
    """API models wrap code in markdown fences; the dialect forbids
    them. Extract the first fenced block if present, else pass through."""
    m = re.search(r"```(?:gcode|nc|text)?\s*\n(.*?)```", text,
                  flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_iid")
    ap.add_argument("--arm", choices=["gcode", "ir"], default="gcode")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=None,
                    help="first frozen split seed to evaluate")
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    default="anthropic")
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--reasoning-effort",
                    choices=["none", "low", "medium", "high", "xhigh", "max"],
                    default=None)
    ap.add_argument("--name", default=None,
                    help="output basename override (preserves prior runs)")
    # anthropic SDK >= 1.0 removed client-side temperature/top_p:
    # sampling always uses API defaults. k=1 is therefore the model's
    # default sampling, not strict greedy; best-of-K diversity comes
    # from K independent calls. Disclosed in the paper's protocol.
    args = ap.parse_args()

    # Generous per-request ceiling + one retry: a stuck connection fails
    # that record (API_ERROR row), never the run.
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(timeout=1800.0, max_retries=1)
    else:
        if args.arm != "gcode":
            raise SystemExit("OpenAI provider currently supports --arm gcode")
        from openai import OpenAI
        client = OpenAI(timeout=1800.0, max_retries=1)

    records = [json.loads(l) for l in open(f"data/{args.split}.jsonl")]
    if args.seed_start is not None:
        records = [r for r in records if r["seed"] >= args.seed_start]
    if args.n:
        records = records[:args.n]

    tool = {"name": "emit_plan",
            "description": "Emit the machining plan.",
            "input_schema": SCHEMA}
    system_prompt = SYSTEM_GCODE if args.arm == "gcode" else SYSTEM_IR
    prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
    print(f"system prompt sha256 {prompt_sha}", flush=True)

    os.makedirs("results", exist_ok=True)
    protocol = (args.reasoning_effort or
                ("adaptive" if args.provider == "anthropic" else "default"))
    name = args.name or (
        f"frontier_{args.provider}_{args.model.replace('/', '_')}_"
        f"{protocol}_{args.arm}_{args.split}_k{args.k}"
    )
    # line-buffered: progress must be visible on disk while running
    archive = open(f"results/{name}_rollouts.jsonl", "w", buffering=1)

    rows = []
    for i, rec in enumerate(records):
        gen_split = "train" if rec["split"] in ("train", "val", "eval_iid") \
            else rec["split"]
        task = make_task(rec["seed"], gen_split)
        ver_t, ver_a = Verifier(task, RES), Verifier(task, AUDIT_RES)
        best = None
        for _ in range(args.k):
            usage_input = usage_output = usage_reasoning = 0
            try:
                if args.provider == "openai":
                    effort = args.reasoning_effort or "none"
                    with client.responses.stream(
                        model=args.model,
                        instructions=system_prompt,
                        input=rec["prompt"],
                        reasoning={"effort": effort},
                        max_output_tokens=32000,
                        store=False,
                    ) as stream:
                        response = stream.get_final_response()
                    text = strip_fences(response.output_text)
                    stop = response.status
                    if response.usage:
                        usage_input = response.usage.input_tokens
                        usage_output = response.usage.output_tokens
                        details = response.usage.output_tokens_details
                        usage_reasoning = (
                            details.reasoning_tokens if details else 0)
                elif args.arm == "gcode":
                    # fable-5 thinking is adaptive and CANNOT be
                    # disabled (API rejects thinking.type.disabled).
                    # Protocol: frontier runs as-shipped; the 32k cap
                    # leaves room for the document after thinking --
                    # at 16k, 4/20 pilot tasks spent the whole budget
                    # thinking and returned EMPTY documents (receipt
                    # 20260824). stop_reason is archived per row.
                    # STREAMING: thinking + a long document can take
                    # >5 min per response; a non-streaming read either
                    # trips the client timeout or the SDK's long-
                    # request guard. The stream is accumulated, so
                    # downstream handling is unchanged.
                    with client.messages.stream(
                        model=args.model, max_tokens=32000,
                        system=system_prompt,
                        messages=[{"role": "user",
                                   "content": rec["prompt"]}],
                    ) as stream:
                        msg = stream.get_final_message()
                    text = strip_fences("".join(
                        b.text for b in msg.content if b.type == "text"))
                    usage_input = msg.usage.input_tokens
                    usage_output = msg.usage.output_tokens
                else:
                    with client.messages.stream(
                        model=args.model, max_tokens=4096,
                        system=system_prompt,
                        messages=[{"role": "user",
                                   "content": rec["prompt"]}],
                        tools=[tool],
                        tool_choice={"type": "tool", "name": "emit_plan"},
                    ) as stream:
                        msg = stream.get_final_message()
                    blocks = [b for b in msg.content if b.type == "tool_use"]
                    text = json.dumps(blocks[0].input) if blocks else ""
                    usage_input = msg.usage.input_tokens
                    usage_output = msg.usage.output_tokens
                if args.provider == "anthropic":
                    stop = msg.stop_reason
            except Exception as e:
                text = f"API_ERROR: {e}"
                stop = "api_error"

            if args.arm == "gcode":
                res = ver_t.verify_gcode(text)
                ares = ver_a.verify_gcode(text)
            else:
                plan, err = validate_syntax(text)
                if plan is None:
                    res = ares = dict(feasible=False, fail=f"parse:{err}",
                                      clearance=0.0, linf=99.0, time_s=0.0,
                                      wear_max=0.0,
                                      n_setups=0, per_op=[])
                else:
                    res, ares = ver_t.verify(plan), ver_a.verify(plan)
            row = dict(seed=rec["seed"], text=text, stop=stop,
                       provider=args.provider, model=args.model,
                       reasoning_effort=protocol,
                       system_prompt_sha256_12=prompt_sha,
                       reward_version="v9",
                       input_tokens=usage_input,
                       output_tokens=usage_output,
                       reasoning_tokens=usage_reasoning,
                       reward_train=reward_v9(res, rec["t_ref_s"]),
                       ok_train=is_ok(res),
                       reward_audit=reward_v9(ares, rec["t_ref_s"]),
                       ok_audit=is_ok(ares), fail=res["fail"],
                       path_valid=res.get("path_valid", False),
                       load_safe=res.get("load_safe", True),
                       ramp_safe=res.get("ramp_safe", True),
                       force_util=res.get("force_util", 0.0),
                       power_util=res.get("power_util", 0.0),
                       ramp_util=res.get("ramp_util", 0.0),
                       ramp_angle_deg=res.get("ramp_angle_deg", 0.0),
                       reach_safe=res.get("reach_safe", True),
                       reach_util=res.get("reach_util", 0.0),
                       gouge_safe=res.get("gouge_safe", True),
                       gouge_depth_mm=res.get("gouge_depth_mm", 0.0),
                       gouge_ratio=res.get("gouge_ratio", 0.0),
                       multi_mill_setup=used_multi_mill_setup(res),
                       n_setups=res["n_setups"], time_s=res["time_s"])
            archive.write(json.dumps(row) + "\n")
            if best is None or row["reward_train"] > best["reward_train"]:
                best = row
        rows.append(best)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(records)} done "
                  f"(running ok {100*np.mean([r['ok_train'] for r in rows]):.0f}%)",
                  flush=True)
    archive.close()

    n = len(rows)
    def pct(key): return 100.0 * np.mean([r[key] for r in rows])
    print(f"\n=== {name} (n={n}, best-of-{args.k}) ===")
    print(f"ok@train {pct('ok_train'):.1f}%  ok@audit {pct('ok_audit'):.1f}%")
    print(f"mean reward train "
          f"{np.mean([r['reward_train'] for r in rows]):.3f}")
    print(f"multi-mill-tool setup proxy {pct('multi_mill_setup'):.1f}%")
    fails = {}
    for r in rows:
        if r["fail"]:
            fails[r["fail"].split(":")[0]] = fails.get(
                r["fail"].split(":")[0], 0) + 1
    print(f"failure modes: {fails}")


if __name__ == "__main__":
    main()
