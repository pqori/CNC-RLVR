"""S3: GRPO -- policy-gradient RL with the verifier as reward function.

Group-relative advantage: A_i = r_i - mean(group), NO std division
(Dr.GRPO). Clipped ratio, KL to the S0 reference, temperature 1.0.
Main arm (gcode): free decoding, the interpreter is the grammar and
parse failure = -1. IR ablation arm: JSON grammar-locked sampling.

Fresh task seeds (50000+) -- never benchmark seeds. Every run receives a
unique ID, immutable metadata, and its own rollout archive. Coarse-clean
candidates faster than the reference are regraded at 0.05 mm during
training, so the known sub-grid film exploit loses reward before update.

Watch during training: reward, KL, entropy, zero-variance group
fraction, and frontier-metrology rejection. Sustained KL blowout and
zero-variance collapse stop automatically; saved checkpoints receive the
full validation/audit referee.

Usage:
  python train/s3_grpo.py --model runs/s0_merged --steps 400
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MethodType

sys.path.insert(0, ".")
from cnc.ir import SCHEMA                                     # noqa: E402
from cnc.world import make_task                               # noqa: E402
from cnc.parverify import ParallelVerifier                    # noqa: E402
from train.common import (to_messages, verify_benchmark_manifest,  # noqa: E402
                          verify_training_stack,
                          max_sequence_tokens, max_completion_tokens)

GRPO_LORA_R = 64
GRPO_LORA_ALPHA = 128
DISTRIBUTED_TIMEOUT_MINUTES = 30


def disable_accelerate_output_fp32(model) -> bool:
    """Remove only Accelerate's output upcast, preserving autocast/binding."""
    from accelerate.utils.operations import ConvertOutputsToFp32

    bound_forward = model.forward
    outer = getattr(bound_forward, "__func__", None)
    fp32_wrapper = getattr(outer, "__wrapped__", None)
    if not isinstance(fp32_wrapper, ConvertOutputsToFp32):
        return False
    model.forward = MethodType(fp32_wrapper.model_forward, model)
    return True


def verify_trl_runtime_patch(receipt_path: Path) -> dict:
    """Verify installed TRL source hashes against the atomic patch receipt."""
    if not receipt_path.is_file():
        raise RuntimeError(f"missing TRL runtime patch receipt: {receipt_path}")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("patch_schema") != 2 or receipt.get(
            "patch_revision") != "trl-1.10.0-bounded-logits-v2":
        raise RuntimeError(
            "obsolete or unknown TRL runtime patch receipt: "
            f"schema={receipt.get('patch_schema')}, "
            f"revision={receipt.get('patch_revision')}"
        )
    if receipt.get("trl_version") != "1.10.0":
        raise RuntimeError(
            f"unexpected TRL patch version: {receipt.get('trl_version')}"
        )
    package_root = Path(
        importlib.metadata.distribution("trl").locate_file("")
    ).resolve()
    grpo_path = package_root / "trl/trainer/grpo_trainer.py"
    utils_path = package_root / "trl/trainer/utils.py"
    grpo_hash = hashlib.sha256(grpo_path.read_bytes()).hexdigest()
    utils_hash = hashlib.sha256(utils_path.read_bytes()).hexdigest()
    expected_grpo = receipt["temperature_division"]["after_sha256"]
    expected_utils = receipt["selective_log_softmax"]["after_sha256"]
    if grpo_hash != expected_grpo or utils_hash != expected_utils:
        raise RuntimeError(
            "installed TRL sources differ from runtime patch receipt: "
            f"grpo={grpo_hash} expected={expected_grpo}, "
            f"utils={utils_hash} expected={expected_utils}"
        )
    return {
        "trl_runtime_patch_receipt_sha256": hashlib.sha256(
            receipt_bytes).hexdigest(),
        "trl_grpo_source_sha256": grpo_hash,
        "trl_utils_source_sha256": utils_hash,
    }


def completion_starts_as_expected(text: str, arm: str) -> bool:
    """Cheap template tripwire that accepts the exact supervised prefix."""
    stripped = text.lstrip()
    if arm == "ir":
        return stripped.startswith("{")
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("(") and line.endswith(")"):
            continue
        return line.upper() == "G21 G90"
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged S0 checkpoint")
    ap.add_argument("--out", default=None,
                    help="checkpoint directory (default: run-specific path)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--save_steps", type=int, default=50,
                    help="checkpoint interval in optimizer steps")
    ap.add_argument("--num_generations", type=int, default=16)
    ap.add_argument("--per_device_batch_size", type=int, default=None,
                    help="completion-scoring microbatch per GPU; defaults "
                         "to num_generations")
    ap.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="optional run-specific cap no larger than the arm-wide limit",
    )
    ap.add_argument("--prompts_per_step", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--rng_seed", type=int, default=0,
                    help="model, sampler, and dataloader RNG seed")
    ap.add_argument("--n_tasks", type=int, default=20000)
    ap.add_argument("--run_id", default=None,
                    help="unique archive/metadata ID; autogenerated if absent")
    ap.add_argument("--resume_from_checkpoint", default=None)
    ap.add_argument("--attempt_id", default="initial",
                    help="fresh artifact lineage; required and non-initial "
                         "when resuming a checkpoint")
    ap.add_argument("--environment_lock",
                    default="results/training_environment.lock",
                    help="pip-freeze lock captured before smoke/full run")
    ap.add_argument(
        "--trl_patch_receipt",
        default="results/trl_grpo_temperature_patch.json",
        help="receipt for the bounded-memory TRL scoring patch",
    )
    ap.add_argument("--max_kl", type=float, default=1.0,
                    help="stop after sustained logged KL above this value")
    ap.add_argument("--kl_patience", type=int, default=5)
    ap.add_argument("--zero_var_patience", type=int, default=20,
                    help="stop after this many reward batches with at least "
                         "95 percent "
                         "zero-variance prompt groups")
    ap.add_argument("--no_frontier_spot", action="store_true",
                    help="UNSAFE ablation: disable 0.05 mm regrading of "
                         "coarse-clean candidates faster than reference")
    ap.add_argument("--spot_workers", type=int, default=1,
                    help="node-wide max concurrent 0.05 mm verifiers; "
                         "increase only after measuring host RAM")
    ap.add_argument("--verify_workers", type=int, default=16,
                    help="node-wide coarse verifier processes; divided "
                         "across local training ranks")
    ap.add_argument("--arm", choices=["gcode", "ir"], default="gcode")
    ap.add_argument("--no_grammar", action="store_true",
                    help="ir arm only, H3 ablation: drop the JSON lock. "
                         "The gcode arm always decodes freely -- its "
                         "grammar is the fail-loud interpreter, parse "
                         "failure = -1 reward.")
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    workers_per_rank = max(1, args.verify_workers // local_world)
    if world > 1 and args.run_id is None:
        ap.error("--run_id is required for distributed launches so every "
                 "rank uses the same immutable run identity")
    if world > 1 and args.per_device_batch_size is None:
        ap.error("--per_device_batch_size is required for distributed "
                 "launches; refusing the OOM-prone implicit default")
    if args.num_generations < 2 or args.prompts_per_step < 1:
        ap.error("num_generations must be >=2 and prompts_per_step >=1")
    per_device_batch_size = (
        args.per_device_batch_size
        if args.per_device_batch_size is not None
        else args.num_generations
    )
    arm_completion_limit = max_completion_tokens(args.arm)
    completion_token_limit = (
        args.max_completion_tokens
        if args.max_completion_tokens is not None
        else arm_completion_limit
    )
    if args.steps < 1 or args.save_steps < 1 \
            or per_device_batch_size < 1 \
            or completion_token_limit < 1 \
            or completion_token_limit > arm_completion_limit \
            or args.n_tasks < 1 or args.lr <= 0 \
            or args.beta < 0 or args.max_kl <= 0:
        ap.error("steps, save_steps, per_device_batch_size, completion "
                 "tokens, tasks, learning rate, and max KL must be positive; "
                 "the completion cap cannot exceed the arm-wide limit and "
                 "beta must be nonnegative")
    if args.spot_workers < 1 or args.verify_workers < 1 \
            or args.kl_patience < 1 \
            or args.zero_var_patience < 1:
        ap.error("worker and patience values must be positive")
    if args.verify_workers < local_world:
        ap.error("--verify_workers is a node-wide budget and must be at "
                 "least LOCAL_WORLD_SIZE")
    global_microbatch = per_device_batch_size * world
    if global_microbatch % args.num_generations:
        ap.error("per_device_batch_size * WORLD_SIZE must be divisible by "
                 "num_generations so completion groups remain intact")
    prompts_per_microbatch = global_microbatch // args.num_generations
    if args.prompts_per_step % prompts_per_microbatch:
        ap.error("prompts_per_step must be divisible by "
                 "(per_device_batch_size * WORLD_SIZE / num_generations)")
    grad_accum = args.prompts_per_step // prompts_per_microbatch
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.attempt_id):
        ap.error("--attempt_id may contain only letters, numbers, ._-")
    if args.resume_from_checkpoint and args.attempt_id == "initial":
        ap.error("resume requires an explicit non-initial --attempt_id")
    verify_training_stack()

    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    if world > 1 and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=DISTRIBUTED_TIMEOUT_MINUTES),
        )

    class RunSafetyCallback(TrainerCallback):
        """Fail closed on a sustained KL blowout when TRL logs KL."""

        def __init__(self):
            self.high_kl = stats.get("high_kl", 0)

        def on_train_begin(
                self, args_, state, control, model=None, **kwargs):
            # Trainer/Accelerate installs ConvertOutputsToFp32 during
            # _inner_training_loop, after GRPOTrainer.__init__. Remove only
            # that wrapper here, while preserving its wrapped autocast forward.
            candidates = [
                candidate for candidate in (
                    model, getattr(model, "module", None)
                )
                if candidate is not None
            ]
            for candidate in candidates:
                if disable_accelerate_output_fp32(candidate):
                    print(
                        "disabled Accelerate full-output FP32 "
                        "materialization"
                    )
                    return control
            raise RuntimeError(
                "expected Accelerate ConvertOutputsToFp32 wrapper was not "
                "present at on_train_begin"
            )

        def on_log(self, args_, state, control, logs=None, **kwargs):
            logs = logs or {}
            values = [
                float(v) for k, v in logs.items()
                if k.split("/")[-1] == "kl" and isinstance(v, (int, float))
            ]
            if not values:
                return control
            kl = max(values)
            if world > 1:
                kl_tensor = torch.tensor(
                    kl,
                    device=torch.device(
                        "cuda", torch.cuda.current_device()),
                    dtype=torch.float64,
                )
                dist.all_reduce(kl_tensor, op=dist.ReduceOp.MAX)
                kl = float(kl_tensor.item())
            self.high_kl = self.high_kl + 1 if kl > args.max_kl else 0
            stats["high_kl"] = self.high_kl
            _save_guard_state()
            if self.high_kl >= args.kl_patience:
                raise RuntimeError(
                    f"sustained KL blowout: kl={kl:.6g} exceeded "
                    f"{args.max_kl:g} for {self.high_kl} logs"
                )
            return control

    # ---- dataset: fresh prompts, seed recoverable from each row ----
    seeds = list(range(args.seed0, args.seed0 + args.n_tasks))
    ds = Dataset.from_list([
        {"prompt": to_messages(make_task(s, "train").prompt(), args.arm),
         "seed": s}
        for s in seeds])

    # ---- immutable run identity and verifier-as-reward archive ----
    os.makedirs("results", exist_ok=True)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{args.arm}_s{args.rng_seed}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        sys.exit("ERROR: --run_id may contain only letters, numbers, ._-")
    output_dir = args.out or f"runs/s3_grpo_{run_id}"
    attempt_suffix = (
        "" if args.attempt_id == "initial"
        else f"_attempt-{args.attempt_id}"
    )
    rank_suffix = f"_rank{rank}" if world > 1 else ""
    archive_path = (
        f"results/grpo_{run_id}{attempt_suffix}{rank_suffix}_rollouts.jsonl"
    )
    metadata_path = f"results/grpo_{run_id}_metadata.json"
    attempt_metadata_path = (
        f"results/grpo_{run_id}{attempt_suffix}_metadata.json"
    )
    final_path = (
        f"results/grpo_{run_id}{attempt_suffix}{rank_suffix}_final.json"
    )
    guard_path = (
        f"results/grpo_{run_id}{attempt_suffix}{rank_suffix}_guard.json"
    )
    out_path = Path(output_dir)
    preflight = None
    if is_main_process:
        error = None
        expected_archives = [
            Path(
                f"results/grpo_{run_id}{attempt_suffix}"
                f"{f'_rank{r}' if world > 1 else ''}_rollouts.jsonl"
            )
            for r in range(world)
        ]
        if any(path.exists() for path in expected_archives):
            error = "one or more attempt archives already exist"
        elif not args.resume_from_checkpoint and out_path.exists() \
                and (not out_path.is_dir() or any(out_path.iterdir())):
            error = f"output path is not empty: {output_dir}"
        elif not args.resume_from_checkpoint and Path(metadata_path).exists():
            error = f"metadata already exists: {metadata_path}"
        elif Path(attempt_metadata_path).exists():
            error = f"attempt metadata already exists: {attempt_metadata_path}"
        preflight = {"ok": error is None, "error": error}
    if world > 1:
        payload = [preflight]
        dist.broadcast_object_list(payload, src=0)
        preflight = payload[0]
    if not preflight["ok"]:
        sys.exit(f"ERROR: rank-0 launch preflight failed: "
                 f"{preflight['error']}")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        try:
            checkpoint.relative_to(out_path.resolve())
        except ValueError:
            sys.exit("ERROR: resume checkpoint must belong to this run's "
                     f"output directory: {output_dir}")
        if not (checkpoint / "trainer_state.json").is_file():
            sys.exit("ERROR: resume checkpoint has no trainer_state.json: "
                     f"{checkpoint}")
        checkpoints = [
            p for p in out_path.glob("checkpoint-*")
            if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()
        ]
        latest = max(
            checkpoints,
            key=lambda p: int(p.name.removeprefix("checkpoint-")),
            default=None,
        )
        if latest is None or checkpoint != latest.resolve():
            sys.exit("ERROR: resume must use the latest checkpoint in this "
                     f"run ({latest}); refusing a silent rollback")
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    benchmark_manifest_sha256 = verify_benchmark_manifest(
        "data/benchmark.sha256")
    environment_lock = Path(args.environment_lock)
    if not environment_lock.is_file():
        sys.exit("ERROR: missing frozen training environment: "
                 f"{environment_lock}")
    environment_lock_bytes = environment_lock.read_bytes()
    current_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    normalized_freeze = "\n".join(
        sorted(line for line in current_freeze.splitlines() if line)
    ) + "\n"
    if normalized_freeze.encode() != environment_lock_bytes:
        sys.exit("ERROR: installed packages differ from "
                 f"{environment_lock}; recreate the environment or rerun "
                 "the complete smoke sequence before launch")
    environment_lock_sha256 = hashlib.sha256(
        environment_lock_bytes).hexdigest()
    try:
        trl_patch_metadata = verify_trl_runtime_patch(
            Path(args.trl_patch_receipt))
    except (KeyError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: invalid TRL runtime patch: {exc}")
    source_hasher = hashlib.sha256()
    source_paths = [
        Path("llm_policy.py"),
        Path("requirements.txt"),
        Path("requirements-training.txt"),
    ]
    for root in (Path("cnc"), Path("train"), Path("scripts")):
        source_paths.extend(root.rglob("*.py"))
        source_paths.extend(root.rglob("*.md"))
    for path in sorted(set(source_paths)):
        source_hasher.update(str(path).encode())
        source_hasher.update(b"\0")
        source_hasher.update(path.read_bytes())
        source_hasher.update(b"\0")
    metadata = {
        "run_id": run_id,
        "attempt_id": args.attempt_id,
        "parent_checkpoint": args.resume_from_checkpoint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "model": args.model,
        "output_dir": output_dir,
        "arm": args.arm,
        "steps": args.steps,
        "save_steps": args.save_steps,
        "num_generations": args.num_generations,
        "per_device_batch_size": per_device_batch_size,
        "prompts_per_step": args.prompts_per_step,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "loss_type": "dr_grpo",
        "scale_rewards": False,
        "optimization": "lora",
        "lora_r": GRPO_LORA_R,
        "lora_alpha": GRPO_LORA_ALPHA,
        "keep_model_outputs_mixed_precision": True,
        "reward_semantics": "reward_v9+frontier_spot",
        "max_sequence_tokens": max_sequence_tokens(args.arm),
        "max_completion_tokens": completion_token_limit,
        "seed0": args.seed0,
        "n_tasks": args.n_tasks,
        "rng_seed": args.rng_seed,
        "max_kl": args.max_kl,
        "kl_patience": args.kl_patience,
        "zero_var_patience": args.zero_var_patience,
        "frontier_spot": not args.no_frontier_spot,
        "spot_workers": args.spot_workers,
        "verify_workers": args.verify_workers,
        "verify_workers_per_rank": workers_per_rank,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "git_revision": git_revision,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "environment_lock": str(environment_lock),
        "environment_lock_sha256": environment_lock_sha256,
        "trl_patch_receipt": args.trl_patch_receipt,
        **trl_patch_metadata,
        "source_tree_sha256": source_hasher.hexdigest(),
        "world_size": world,
        "local_world_size": local_world,
        "distributed_timeout_minutes": DISTRIBUTED_TIMEOUT_MINUTES,
        "argv": sys.argv,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy", "scipy", "torch", "transformers", "trl",
                "peft", "datasets", "accelerate", "vllm",
            )
        },
    }
    write_run_metadata = is_main_process and not args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        if not os.path.exists(metadata_path):
            sys.exit("ERROR: cannot resume without existing run metadata: "
                     f"{metadata_path}")
        with open(metadata_path) as f:
            previous = json.load(f)
        immutable = (
            "run_id", "model", "output_dir", "arm", "steps", "save_steps",
            "seed0", "n_tasks", "rng_seed",
            "frontier_spot", "spot_workers", "verify_workers",
            "verify_workers_per_rank", "world_size", "local_world_size",
            "distributed_timeout_minutes",
            "max_kl", "kl_patience", "zero_var_patience",
            "python", "versions",
            "source_tree_sha256",
            "benchmark_manifest_sha256", "environment_lock_sha256",
            "trl_runtime_patch_receipt_sha256",
            "trl_grpo_source_sha256", "trl_utils_source_sha256",
            "num_generations", "per_device_batch_size",
            "prompts_per_step", "gradient_accumulation_steps",
            "learning_rate", "beta", "loss_type",
            "scale_rewards", "optimization", "lora_r", "lora_alpha",
            "keep_model_outputs_mixed_precision",
            "reward_semantics",
            "max_sequence_tokens", "max_completion_tokens",
        )
        if any(previous.get(k) != metadata.get(k) for k in immutable):
            sys.exit("ERROR: resume metadata does not match current arguments")
    elif is_main_process and os.path.exists(metadata_path):
        sys.exit(f"ERROR: metadata already exists: {metadata_path}; choose "
                 "a new --run_id")

    # Sanity tripwire. IR arm with the lock ENGAGED: every completion
    # starts with "{" and parse failures are near-impossible; if the
    # lock is configured but silently ignored by TRL, crash instead of
    # training on -1 noise for hours. GCODE arm (no lock by design):
    # the S0 clone emits provenance comments followed by "G21 G90", so
    # completions must reproduce that shape and mostly parse.
    stats = {
        "n": 0, "parse_fail": 0, "reward_batch": 0,
        "cap_truncated_parse": 0,
        "bad_start": 0,
        "spot_checked": 0, "spot_rejected": 0,
        "zero_var_streak": 0, "high_kl": 0,
    }

    def _save_guard_state():
        temp_path = f"{guard_path}.tmp.{os.getpid()}"
        with open(temp_path, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        os.replace(temp_path, guard_path)

    want = "{" if args.arm == "ir" else "G21 G90 after leading comments"
    failpfx = "parse:" if args.arm == "ir" else "gcode:"

    def _tripwire(texts, results, cap_truncated):
        stats["bad_start"] += sum(
            1 for t in texts
            if not completion_starts_as_expected(t, args.arm)
        )
        bad_start = stats["bad_start"]
        n, pf = stats["n"], stats["parse_fail"]
        cap_pf = stats["cap_truncated_parse"]
        unexpected_pf = pf - cap_pf
        hard = bad_start if args.arm == "ir" else bad_start > 0.25 * n
        if hard or (n >= 8 and unexpected_pf == n) or (
                n >= 16 and unexpected_pf > 0.5 * n):
            archive.flush()
            raise RuntimeError(
                f"sampling looks broken: {bad_start}/{n} completions do "
                f"not start with '{want}', {unexpected_pf}/{n} unexpectedly "
                f"failed to parse, and {cap_pf}/{n} were expected cap "
                "truncations. "
                "Causes: TRL accepted the guided-decoding field but does "
                "not apply it (ir arm), Qwen thinking mode is on (output "
                "starts with <think>), or pervasive non-cap parse failure. "
                "Inspect "
                f"{archive_path}.")

    def reward_fn(completions, seed, **kwargs):
        # [VERIFY] TRL passes dataset columns as kwargs lists; chat-format
        # completions arrive as [{"role": ..., "content": ...}] lists.
        texts = [c[-1]["content"] if isinstance(c, list) else c
                 for c in completions]
        seed_values = [int(s) for s in seed]
        results = pv.verify_batch(seed_values, texts)
        parse_failed = [
            str(r["fail"]).startswith(failpfx) for r in results
        ]
        cap_truncated = [False] * len(results)
        if args.max_completion_tokens is not None \
                and args.max_completion_tokens < arm_completion_limit:
            completion_id_lists = kwargs.get("completion_ids")
            if completion_id_lists is None or len(
                    completion_id_lists) != len(results):
                raise RuntimeError(
                    "capped training requires exact completion_ids from TRL"
                )
            trainer = trainer_ref.get("trainer")
            tokenizer = getattr(trainer, "processing_class", None)
            if tokenizer is None:
                raise RuntimeError(
                    "capped training requires the trainer tokenizer to "
                    "classify expected truncations"
                )
            terminal_ids = {
                token_id for token_id in (
                    tokenizer.eos_token_id, tokenizer.pad_token_id
                )
                if token_id is not None
            }
            cap_truncated = [
                failed
                and len(ids) >= completion_token_limit
                and len(ids) > 0
                and ids[-1] not in terminal_ids
                for ids, failed in zip(
                    completion_id_lists, parse_failed)
            ]
        stats["n"] += len(results)
        stats["parse_fail"] += sum(parse_failed)
        stats["cap_truncated_parse"] += sum(cap_truncated)
        stats["reward_batch"] += 1
        global_step = getattr(getattr(trainer_ref.get("trainer"), "state", None),
                              "global_step", None)
        for completion_i, (sd, text, r, was_cap_truncated) in enumerate(
                zip(seed_values, texts, results, cap_truncated)):
            archive.write(json.dumps(
                dict(run_id=run_id, attempt_id=args.attempt_id,
                     parent_checkpoint=args.resume_from_checkpoint,
                     global_step=global_step,
                     reward_batch=stats["reward_batch"],
                     completion_index=completion_i, seed=int(sd),
                     cap_truncated=was_cap_truncated,
                     reward=r["reward"],
                     reward_coarse=r.get("reward_coarse"),
                     reward_spot=r.get("reward_spot"),
                     spot_checked=r.get("spot_checked", False),
                     spot_ok=r.get("spot_ok"),
                     spot_linf=r.get("spot_linf"),
                     ok=r.get("ok", False),
                     feasible=r["feasible"],
                     path_valid=r.get("path_valid", False),
                     load_safe=r.get("load_safe", True),
                     ramp_safe=r.get("ramp_safe", True),
                     reach_safe=r.get("reach_safe", True),
                     gouge_safe=r.get("gouge_safe", True),
                     force_util=r.get("force_util", 0.0),
                     power_util=r.get("power_util", 0.0),
                     ramp_util=r.get("ramp_util", 0.0),
                     ramp_angle_deg=r.get("ramp_angle_deg", 0.0),
                     reach_util=r.get("reach_util", 0.0),
                     gouge_depth_mm=r.get("gouge_depth_mm", 0.0),
                     gouge_ratio=r.get("gouge_ratio", 0.0),
                     gouge_volume_mm3=r.get("gouge_volume_mm3", 0.0),
                     fail=r["fail"], clearance=r["clearance"],
                     linf=r["linf"], time_s=r["time_s"],
                     n_setups=r["n_setups"], text=text)) + "\n")
        archive.flush()
        os.fsync(archive.fileno())
        stats["spot_checked"] += sum(
            bool(r.get("spot_checked")) for r in results
        )
        stats["spot_rejected"] += sum(
            r.get("spot_checked") and not r.get("spot_ok")
            for r in results
        )
        local_pairs = [
            (sd, float(r["reward"])) for sd, r in zip(seed_values, results)
        ]
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_pairs)
            global_pairs = [
                pair for rank_pairs in gathered for pair in rank_pairs
            ]
        else:
            global_pairs = local_pairs
        grouped = {}
        for sd, reward in global_pairs:
            grouped.setdefault(sd, []).append(reward)
        bad_cardinality = {
            sd: len(rs) for sd, rs in grouped.items()
            if len(rs) != args.num_generations
        }
        if bad_cardinality:
            raise RuntimeError(
                "distributed reward groups have unexpected cardinality "
                f"(expected {args.num_generations}): {bad_cardinality}")
        complete_groups = list(grouped.values())
        zero_var = sum(max(rs) - min(rs) < 1e-12
                       for rs in complete_groups)
        zero_frac = zero_var / max(len(complete_groups), 1) \
            if complete_groups else 0.0
        stats["zero_var_streak"] = (
            stats["zero_var_streak"] + 1 if zero_frac >= 0.95 else 0
        )
        _save_guard_state()
        if stats["zero_var_streak"] >= args.zero_var_patience:
            archive.flush()
            raise RuntimeError(
                f"reward collapse: {zero_var}/{len(complete_groups)} "
                "prompt groups "
                f"had zero variance for {stats['zero_var_streak']} "
                "consecutive reward batches"
            )
        if stats["reward_batch"] % 10 == 0:
            print(
                f"[reward-monitor] step={global_step} n={stats['n']} "
                f"spot={stats['spot_checked']} "
                f"spot_rejected={stats['spot_rejected']} "
                f"zero_var_groups={zero_var}/{len(complete_groups)}",
                flush=True,
            )
        if not (args.arm == "ir" and args.no_grammar):
            # the ir+no_grammar H3 ablation trains on parse noise BY
            # DESIGN; every other configuration must not
            _tripwire(texts, results, cap_truncated)
        return [r["reward"] for r in results]

    cfg = GRPOConfig(
        output_dir=output_dir,
        max_steps=args.steps,
        learning_rate=args.lr,
        beta=args.beta,                      # KL to reference
        epsilon=0.2,
        num_generations=args.num_generations,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        temperature=1.0,
        max_completion_length=completion_token_limit,
        # Dr.GRPO has two independent normalizations: do not divide group
        # advantages by their reward std, and normalize token losses by the
        # global max-completion-length constant rather than response length.
        scale_rewards=False,
        loss_type="dr_grpo",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        use_vllm=True,                       # [VERIFY]
        vllm_mode="colocate",                # [VERIFY]
        generation_kwargs=(
            {"structured_outputs": {"json": SCHEMA}}
            if args.arm == "ir" and not args.no_grammar else None
        ),
        report_to=[],
        seed=args.rng_seed,
        data_seed=args.rng_seed,
    )
    # Current TRL forwards generation_kwargs to vLLM SamplingParams and
    # converts the JSON dictionary to StructuredOutputsParams. Fail closed
    # on older TRL rather than silently running the locked IR arm unlocked.
    if args.arm == "ir" and not args.no_grammar:
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cfg)}
        if "generation_kwargs" not in field_names:
            sys.exit(
                "ERROR: installed TRL lacks GRPOConfig.generation_kwargs; "
                "install the locked training stack before running the IR arm.")
        print("grammar lock via GRPOConfig.generation_kwargs")

    # [VERIFY] Qwen3 thinking mode must be OFF in rollouts too; recent
    # TRL forwards chat_template_kwargs from the config.
    if hasattr(cfg, "chat_template_kwargs"):
        cfg.chat_template_kwargs = {"enable_thinking": False}
    else:
        sys.exit("ERROR: installed TRL lacks GRPOConfig."
                 "chat_template_kwargs; refusing to risk Qwen thinking "
                 "tokens inside executable output")

    if write_run_metadata:
        with open(metadata_path, "x") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
    elif is_main_process:
        with open(attempt_metadata_path, "x") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
    _save_guard_state()
    if world > 1:
        dist.barrier()
    archive = None
    pv = None
    trainer_ref = {}
    trainer = None
    status, error = "failed", None
    try:
        archive = open(archive_path, "x")
        pv = ParallelVerifier(
            workers=workers_per_rank, mode=args.arm,
            frontier_spot=not args.no_frontier_spot,
            spot_workers=args.spot_workers,
        )
        peft_cfg = LoraConfig(
            r=GRPO_LORA_R,
            lora_alpha=GRPO_LORA_ALPHA,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        trainer = GRPOTrainer(
            model=args.model,
            args=cfg,
            train_dataset=ds,
            reward_funcs=reward_fn,
            callbacks=[RunSafetyCallback()],
            peft_config=peft_cfg,
        )
        trainer_ref["trainer"] = trainer
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(output_dir)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pv is not None:
            pv.close()
        if archive is not None:
            archive.close()
        final_temp = f"{final_path}.tmp.{os.getpid()}"
        with open(final_temp, "w") as f:
            json.dump({
                "run_id": run_id,
                "attempt_id": args.attempt_id,
                "parent_checkpoint": args.resume_from_checkpoint,
                "status": status,
                "error": error,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": getattr(
                    getattr(trainer, "state", None), "global_step", None
                ),
                "rollouts": stats["n"],
                "parse_failures": stats["parse_fail"],
                "frontier_spot_checked": stats["spot_checked"],
                "frontier_spot_rejected": stats["spot_rejected"],
                "zero_var_streak": stats["zero_var_streak"],
                "high_kl_streak": stats["high_kl"],
            }, f, indent=2, sort_keys=True)
        os.replace(final_temp, final_path)


if __name__ == "__main__":
    main()
