"""Generate and freeze the benchmark.

Splits (seed ranges are part of the spec):
  train        seeds    0-1999   SFT pairs; kept only if teacher is ok
  val          seeds 5000-5199   gate metrics during training
  eval_iid     seeds 8000-8199   the exam
  oot_multitool seeds 9000-9099  the hard exam, axis 1
  oot_setups   seeds 9500-9599   the hard exam, axis 2

Each record: seed, split, material, prompt text (byte-frozen), teacher
plan JSON, the teacher G-CODE DOCUMENT (the SFT/demonstration artifact
of the main arm; byte-frozen, round-trip audited), teacher verify
metrics, and t_ref_s -- the time reference all rewards normalize
against. t_ref comes from the first FEASIBLE plan in the reference
ladder teacher -> multitool -> grouped: on oot_multitool the teacher's
single-tool attempt snaps the slender EM3L under the load model (that
is the exam), so the multitool reference supplies the time scale
there. t_ref_source records which rung was used.

Also prints per-split document-length stats (lines and approximate
tokens at ~3.5 chars/token) -- these set the RL/SFT context budgets.

Writes data/<split>.jsonl + data/benchmark.sha256. Refuses to overwrite.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from cnc.world import make_task
from cnc.sim import Verifier, reward_v9, is_ok
from cnc.expand import emit_gcode
from cnc.planner import template_plan, multitool_plan, grouped_plan

SPLITS = [
    ("train",         range(0, 2000)),
    ("val",           range(5000, 5200)),
    ("eval_iid",      range(8000, 8200)),
    ("oot_multitool", range(9000, 9100)),
    ("oot_setups",    range(9500, 9600)),
]
ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true",
                help="replace an existing freeze after a benchmark revision")
ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                help="parallel verifier processes (size against host RAM)")
args = ap.parse_args()
if args.workers < 1:
    ap.error("--workers must be positive")
# A freeze builds five temporary files in parallel and publishes the matching
# manifest last; concurrent invocations would still waste memory and race.
lock_path = "/tmp/cnc-rlvr-benchmark-freeze.lock"
lock_file = open(lock_path, "w")
try:
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit("another benchmark freeze is already running")
os.makedirs("data", exist_ok=True)
if os.path.exists("data/benchmark.sha256") and not args.force:
    sys.exit("data/ already frozen; refusing to regenerate (use --force "
             "only for an intentional benchmark revision)")

def freeze_record(job):
    split, gen_split, seed = job
    task = make_task(seed, gen_split)
    ver = Verifier(task)
    plan = template_plan(task)
    res = ver.verify(plan)
    ok = is_ok(res)
    if split == "train" and not ok:
        return None, ok, False, None, None, None

    try:
        gdoc = emit_gcode(plan, task)
    except ValueError:
        gdoc = None
    t_ref, t_src = res["time_s"], "teacher"
    if not res["feasible"]:
        for src, fn in (("multitool", multitool_plan),
                        ("grouped", grouped_plan)):
            rres = ver.verify(fn(task))
            if rres["feasible"]:
                t_ref, t_src = rres["time_s"], src
                break
        else:
            raise RuntimeError(
                f"{split} seed {seed}: NO scripted plan is feasible "
                f"(teacher fail={res['fail']})")

    t_ref_stored = round(t_ref, 2)
    rec = dict(
        seed=seed, split=split, material=task.material,
        prompt=task.prompt(),
        teacher_plan=plan.to_dict(),
        teacher_gcode=gdoc,
        teacher_feasible=res["feasible"],
        teacher_ok=ok,
        teacher_time_s=round(res["time_s"], 2),
        teacher_reward=round(reward_v9(res, t_ref_stored), 4),
        t_ref_s=t_ref_stored,
        t_ref_source=t_src,
        n_setups_teacher=res["n_setups"],
        orientations=task.orientations_required(),
    )
    doc_lines = gdoc.count("\n") if gdoc is not None else None
    doc_chars = len(gdoc) if gdoc is not None else None
    return rec, ok, gdoc is None, t_src, doc_lines, doc_chars


hashes = {}
pending_files = []
t0 = time.time()
ctx = mp.get_context("fork")
with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
    for split, seeds in SPLITS:
        gen_split = (
            "train" if split in ("train", "val", "eval_iid") else split
        )
        path = f"data/{split}.jsonl"
        temp_path = f"{path}.tmp.{os.getpid()}"
        n_ok = n_all = n_nodoc = 0
        ref_sources = {}
        doc_lines, doc_chars = [], []
        jobs = ((split, gen_split, seed) for seed in seeds)
        with open(temp_path, "x") as f:
            for rec, ok, nodoc, t_src, lines, chars in pool.map(
                    freeze_record, jobs, chunksize=1):
                n_all += 1
                if rec is None:
                    continue
                n_ok += ok
                n_nodoc += nodoc
                ref_sources[t_src] = ref_sources.get(t_src, 0) + 1
                if lines is not None:
                    doc_lines.append(lines)
                    doc_chars.append(chars)
                f.write(json.dumps(rec) + "\n")
        h = hashlib.sha256(open(temp_path, "rb").read()).hexdigest()
        hashes[split] = h
        pending_files.append((temp_path, path))
        ln, ch = np.array(doc_lines), np.array(doc_chars)
        print(
            f"{split}: {n_all} tasks, {n_ok} teacher-ok, "
            f"{n_nodoc} no-doc, t_ref from {ref_sources}, "
            f"sha256 {h[:12]}... ({time.time()-t0:.0f}s elapsed)",
            flush=True,
        )
        print(f"  docs: lines p50={np.percentile(ln, 50):.0f} "
              f"p95={np.percentile(ln, 95):.0f} max={ln.max()} | "
              f"~tokens p50={np.percentile(ch, 50)/3.5:.0f} "
              f"p95={np.percentile(ch, 95)/3.5:.0f} "
              f"max={ch.max()/3.5:.0f}", flush=True)

manifest_path = "data/benchmark.sha256"
manifest_temp = f"{manifest_path}.tmp.{os.getpid()}"
with open(manifest_temp, "x") as f:
    for split, h in hashes.items():
        f.write(f"{h}  data/{split}.jsonl\n")
# Publish data first and the matching manifest last. A crash during this short
# replacement window leaves a checksum mismatch (fail closed), never a
# manifest that blesses a partially written JSONL.
for temp_path, path in pending_files:
    os.replace(temp_path, path)
os.replace(manifest_temp, manifest_path)
print("FROZEN. data/benchmark.sha256 written.")
