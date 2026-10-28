"""S0: supervised fine-tune the base model into a teacher clone.

LoRA r=64 alpha=128 on all linear layers, lr 1e-4, 2 epochs, loss on
assistant completion tokens only. Qwen3 thinking mode OFF everywhere.

--overfit_smoke: the golden test. Train on 10 examples until memorized,
then greedy-decode each prompt: the recitations must match the expected
completions EXACTLY. If they don't, the chat template / masking / prompt
formatting is broken -- fix that before ANY real run.

Usage:
  python train/s0_sft.py --model Qwen/Qwen3-8B --overfit_smoke
  python train/s0_sft.py --model Qwen/Qwen3-8B --out runs/s0_8b
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from train.common import (sft_row, to_messages, completion_text,  # noqa: E402
                          max_sequence_tokens, max_completion_tokens,
                          verify_benchmark_manifest,
                          verify_training_stack)


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/s0")
    ap.add_argument("--overfit_smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--arm", choices=["gcode", "ir"], default="gcode",
                    help="gcode: policy writes the document (MAIN); "
                         "ir: typed-plan JSON (scaffolding ablation)")
    args = ap.parse_args()
    verify_benchmark_manifest()
    verify_training_stack()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        sys.exit(
            "ERROR: multiple GPUs are visible to single-process SFT. "
            "For smoke use CUDA_VISIBLE_DEVICES=0; for full SFT use "
            "accelerate launch.")

    records = load_rows(args.data, limit=10 if args.overfit_smoke else None)
    # RAFT keeper files are already chat rows; benchmark records are not
    ds = Dataset.from_list([
        ({**r, "chat_template_kwargs": {"enable_thinking": False}}
         if "messages" in r else sft_row(r, args.arm))
        for r in records
    ])

    tok = AutoTokenizer.from_pretrained(args.model)
    sequence_limit = max_sequence_tokens(args.arm)
    measured = []
    for r in records:
        messages = r["messages"] if "messages" in r else \
            sft_row(r, args.arm)["messages"]
        encoded = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
        )
        measured.append((len(encoded["input_ids"]), r.get("seed")))
    longest_tokens, longest_seed = max(measured, key=lambda item: item[0])
    if longest_tokens > sequence_limit:
        sys.exit(
            f"ERROR: dataset row seed={longest_seed} has "
            f"{longest_tokens} tokens, exceeding the declared "
            f"{sequence_limit}-token {args.arm} sequence limit")
    print(
        f"token preflight: max={longest_tokens} limit={sequence_limit} "
        f"seed={longest_seed}"
    )
    # Qwen3 thinking mode is disabled by each dataset row's
    # chat_template_kwargs; TRL 1.10.0 does not expose this on SFTConfig.
    peft_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0,
                          target_modules="all-linear", task_type="CAUSAL_LM")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=60 if args.overfit_smoke else args.epochs,
        learning_rate=2e-4 if args.overfit_smoke else args.lr,
        per_device_train_batch_size=2 if args.overfit_smoke else args.batch,
        gradient_accumulation_steps=1 if args.overfit_smoke else args.grad_accum,
        logging_steps=10,
        save_strategy="no" if args.overfit_smoke else "epoch",
        bf16=True,
        max_length=sequence_limit,
        # Long G-code rows require checkpointed activations; the runbook
        # lowers per-rank batch size for the measured 20k-token train tail.
        gradient_checkpointing=True,
        assistant_only_loss=True,
        report_to=[],
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg)
    trainer.train()

    if not args.overfit_smoke:
        trainer.save_model(args.out)
        print(f"saved LoRA adapter to {args.out}")
        return

    # ---- golden recitation check ----
    model = trainer.model
    # Trainer disables the KV cache while gradient checkpointing is active.
    # Re-enable normal autoregressive inference or long G-code recitation
    # becomes quadratic and appears to hang after the training bar reaches 100%.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    n_exact = 0
    for r in records:
        msgs = to_messages(r["prompt"], args.arm)
        expect = completion_text(r, args.arm)
        expected_tokens = len(
            tok(expect, add_special_tokens=False)["input_ids"]
        )
        if expected_tokens > max_completion_tokens(args.arm):
            sys.exit(
                f"ERROR: smoke completion seed={r.get('seed')} has "
                f"{expected_tokens} tokens, exceeding the declared "
                f"{max_completion_tokens(args.arm)}-token completion limit")
        encoded = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
            enable_thinking=False,  # [VERIFY] kwarg name for Qwen3
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            # The longest smoke G-code exceeds 10k tokens. Size this from the
            # golden completion so exact recitation cannot fail by truncation.
            out = model.generate(**encoded,
                                 max_new_tokens=expected_tokens + 8,
                                 do_sample=False, temperature=None, top_p=None)
        prompt_len = encoded["input_ids"].shape[1]
        text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        exact = text.strip() == expect.strip()
        n_exact += exact
        print(f"seed {r['seed']}: {'EXACT' if exact else 'MISMATCH'}")
        if not exact:
            print(f"  expect: {expect[:120]}...")
            print(f"  got:    {text[:120]}...")
    print(f"\nSMOKE {'PASS' if n_exact == len(records) else 'FAIL'} "
          f"({n_exact}/{len(records)} exact)")
    # fail-closed for && chains: a broken template must stop the night
    sys.exit(0 if n_exact == len(records) else 1)


if __name__ == "__main__":
    main()
