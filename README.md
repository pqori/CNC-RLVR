# CNC-RLVR

Code, frozen synthetic benchmark, and archived evaluation outputs for **When Does RLVR Teach Physical Strategy? Measuring Simulator-Only Post-Training for CNC Planning**.

**Shishun Dai — Harvard University**

[Paper](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Repository](https://github.com/pqori/CNC-RLVR)

Qwen3-8B writes complete multi-setup G-code from semantic descriptions of machining features. A strict interpreter and voxel process model evaluate the generated programs. The experiment compares supervised fine-tuning (SFT) with simulator-reward GRPO and separates program executability from physical success and strategy.

## Reported experiment

The paper reports one training seed and a validation-selected checkpoint at GRPO step 50 of a 100-step run. These are greedy results under the final audit-plus-spot evaluation:

| Split | SFT path-valid | GRPO path-valid | SFT success | GRPO success |
|---|---:|---:|---:|---:|
| Validation, 200 tasks | 88.5% | 91.5% | 18.0% | 18.5% |
| IID, 200 tasks | 86.0% | 93.0% | 15.0% | 15.5% |
| Multi-tool exam, 100 tasks | 89.0% | 96.0% | 0% | 0% |
| Setup exam, 100 tasks | 59.0% | 76.0% | 0% | 0% |

GRPO improves executability and in-distribution mean verifier reward, while strict success changes little and both exams remain at 0/100. This is a result for the reported training regime, not a general limit on RLVR. The archives also include SFT four-sample controls for any-of-four pass@4 and reward-selected search-4.

The benchmark contains 1,999 verified teacher pairs, 200 validation tasks, 200 IID tasks, and two 100-task exams. Tasks are synthetic AL6061 machining problems with fixed stock geometry and controlled cutting parameters. The exams test multi-tool roughing/finishing and grouping features by setup face. GRPO uses fresh seeds from the training task generator, without explicitly mixing in the exam families.

## CPU installation and table reproduction

Run commands from the repository root using Python 3.12 or newer:

```bash
git clone https://github.com/pqori/CNC-RLVR.git
cd CNC-RLVR
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
sha256sum -c data/benchmark.sha256
python artifact/reproduce_tables.py
```

On macOS, use `shasum -a 256 -c data/benchmark.sha256` for the checksum command. CPU dependencies are pinned to NumPy 2.3.3 and SciPy 1.16.2 because deterministic task regeneration is part of the benchmark contract.

`artifact/reproduce_tables.py` reads the released `results/*_summary.json` files. The corresponding `results/*_rollouts.jsonl` files contain generated G-code and per-candidate verifier metrics. Table reproduction needs neither a GPU nor a model checkpoint.

## Regrade archived outputs and check the verifier

```bash
python artifact/regrade_archive.py \
  --archive results/grpo_step50_val_k1_rollouts.jsonl \
  --split val
python scripts/audit_checks.py
python scripts/cross_check.py
```

Add `--limit 1` to the regrading command for a short smoke check. Regrading is substantially slower than reading summaries: every document is evaluated at 0.2 mm and 0.1 mm, and records marked `spot_checked` are additionally evaluated at 0.05 mm. The released headline evaluation spot-checks every audit-clean candidate; its final verdict must pass both audit and spot grids.

The adversarial battery should end with `ALL CHECKS PASS`; the voxel-versus-2.5D cross-check should report `ALL EXACT`. Archived verifier receipts are in `results/receipts/`.

`multi_mill_setup` is a setup-level tool-diversity proxy, not proof that two tools machine the same feature. `fewer_setups` is a feasibility-gated comparison against the teacher. Neither proxy alone establishes a successful physical strategy.

## Optional GPU training

Model weights and trained checkpoints are not included. The release supports offline inspection and regrading of the reported outputs. New generation requires obtaining the base model and training checkpoints; rerunning training does not guarantee the same outcomes.

Use a separate CUDA environment. `requirements-training.txt` pins TRL 1.10.0 and vLLM 0.26.0; `results/training_environment.lock` records the reported environment. Before GRPO, apply the required TRL memory patch, run its two-GPU preflight, and capture the new environment without overwriting the released lock:

```bash
python -m pip install -r requirements-training.txt
python scripts/patch_trl_temperature_division.py \
  --receipt results/trl_patch.local.json
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  --mixed_precision bf16 scripts/test_grpo_runtime_patches.py \
  --receipt results/trl_patch.local.json --stress
python -m pip freeze --all | LC_ALL=C sort \
  > results/training_environment.local.lock
```

The main entrypoints are `train/s0_sft.py`, `train/merge_lora.py`, `train/s3_grpo.py`, and `train/eval_policy.py`; each exposes `--help`. An eight-GPU SFT and GRPO launch uses:

```bash
accelerate launch --num_processes 8 train/s0_sft.py \
  --model Qwen/Qwen3-8B --out runs/s0_8b --batch 1 --grad_accum 4
python train/merge_lora.py --base Qwen/Qwen3-8B \
  --adapter runs/s0_8b --out runs/s0_8b_merged
accelerate launch --num_processes 8 train/s3_grpo.py \
  --model runs/s0_8b_merged --steps 100 \
  --num_generations 4 --per_device_batch_size 1 \
  --max_completion_tokens 8192 --prompts_per_step 8 --save_steps 25 \
  --run_id public_reproduction_seed0 --rng_seed 0 --verify_workers 16 \
  --environment_lock results/training_environment.local.lock \
  --trl_patch_receipt results/trl_patch.local.json
```

Use a fresh run ID for each new run. Preserve the environment lock and patch receipt for checkpoint recovery. GRPO verifies their hashes and refuses incompatible environments or reused archives.

After merging a selected adapter with `train/merge_lora.py`, evaluate its merged model with:

```bash
python train/eval_policy.py --model runs/selected_merged \
  --split eval_iid --k 1 --spot_all_clean
```

Use `val`, `oot_multitool`, or `oot_setups` for the other splits, and `--k 4` for four-sample evaluation. Thinking mode is disabled throughout. The reported GRPO training cap is 8,192 tokens; the evaluation configuration permits 24,576. Archived evaluation outputs do not establish training-time truncation frequency.

## Scope

`cnc/` contains the task generator, scripted plans, G-code interpreter, voxel verifier, and separate 2.5D stock checker. The stock checkers share motion expansion and path discretization. `data/` contains the frozen benchmark, `artifact/` contains reproduction tools, and `train/` contains model training and evaluation code.

This is a simulated process-planning study with semantic inputs, not a STEP or drawing parser. The verifier does not model fixtures, servo dynamics, vibration, thermal effects, or surface finish. Its success verdict is not evidence that a generated program is safe to execute on a physical CNC machine.

## Paper source

The camera-ready PDF and its source are in `paper/`. Build with a LaTeX distribution containing the standard NeurIPS dependencies:

```bash
cd paper
latexmk -pdf main.tex
```

Model weights and dependency packages are not redistributed. Qwen3-8B uses the [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE). See the upstream [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) and [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) licenses for the CPU dependencies, and the respective project licenses for the training stack. The NeurIPS style file retains its upstream notices.
