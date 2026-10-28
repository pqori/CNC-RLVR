#!/usr/bin/env python3
"""Apply memory-equivalent GRPO scoring fixes to the locked TRL.

TRL 1.10.0 computes ``logits = logits / self.temperature`` while scoring
long completions. At temperature 1 this needlessly allocates another full
FP32 logits tensor. Newer TRL uses ``logits.div_(self.temperature)``.
Its selective log-softmax also reduces every token row at once; chunking
those independent rows bounds the temporary without changing the FP32
calculation. This version- and source-guarded patch records file hashes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


EXPECTED_TRL = "1.10.0"
PATCH_SCHEMA = 2
PATCH_REVISION = "trl-1.10.0-bounded-logits-v2"
OLD = "        logits = logits / self.temperature\n"
NEW = "        logits.div_(self.temperature)\n"
OLD_LOGSUMEXP = (
    "        logsumexp_values = torch.stack("
    "[torch.logsumexp(lg, dim=-1) for lg in logits])\n"
)
NEW_LOGSUMEXP = (
    "        flat_logits = logits.reshape(-1, logits.size(-1))\n"
    "        logsumexp_values = torch.cat([\n"
    "            torch.logsumexp(chunk, dim=-1)\n"
    "            for chunk in flat_logits.split(256, dim=0)\n"
    "        ]).reshape(logits.shape[:-1])\n"
)
OLD_BF16 = (
    "        per_token_logps = []\n"
    "        for row_logits, row_labels in zip(logits, index, strict=True):  "
    "# loop to reduce peak mem consumption\n"
    "            row_logps = F.log_softmax(row_logits, dim=-1)\n"
    "            row_per_token_logps = row_logps.gather("
    "dim=-1, index=row_labels)\n"
    "            per_token_logps.append(row_per_token_logps)\n"
    "        per_token_logps = torch.stack(per_token_logps)\n"
)
NEW_BF16 = (
    "        flat_logits = logits.reshape(-1, logits.size(-1))\n"
    "        flat_index = index.reshape(-1, index.size(-1))\n"
    "        per_token_logps = []\n"
    "        for logits_chunk, index_chunk in zip(\n"
    "            flat_logits.split(256, dim=0),\n"
    "            flat_index.split(256, dim=0),\n"
    "            strict=True,\n"
    "        ):\n"
    "            chunk_logps = F.log_softmax(logits_chunk, dim=-1)\n"
    "            per_token_logps.append(chunk_logps.gather(\n"
    "                dim=-1, index=index_chunk\n"
    "            ))\n"
    "        per_token_logps = torch.cat(per_token_logps).reshape(index.shape)\n"
)
LEGACY_BF16 = (
    "        flat_logits = logits.reshape(-1, logits.size(-1))\n"
    "        flat_index = index.reshape(-1)\n"
    "        per_token_logps = []\n"
    "        for logits_chunk, index_chunk in zip(\n"
    "            flat_logits.split(256, dim=0),\n"
    "            flat_index.split(256, dim=0),\n"
    "            strict=True,\n"
    "        ):\n"
    "            chunk_logps = F.log_softmax(logits_chunk, dim=-1)\n"
    "            per_token_logps.append(chunk_logps.gather(\n"
    "                dim=-1, index=index_chunk.unsqueeze(-1)\n"
    "            ).squeeze(-1))\n"
    "        per_token_logps = torch.cat(per_token_logps).reshape(index.shape)\n"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan_standard(
        text: str, old: str, new: str, label: str) -> tuple[str, str]:
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == 0:
        return text.replace(old, new, 1), "patched"
    if old_count == 0 and new_count == 1:
        return text, "already_patched"
    raise ValueError(
        f"ambiguous {label} patch: old_count={old_count}, "
        f"new_count={new_count}"
    )


def plan_bf16(text: str) -> tuple[str, str]:
    old_count = text.count(OLD_BF16)
    new_count = text.count(NEW_BF16)
    legacy_count = text.count(LEGACY_BF16)
    if old_count == 1 and new_count == 0 and legacy_count == 0:
        return text.replace(OLD_BF16, NEW_BF16, 1), "patched"
    if old_count == 0 and new_count == 1 and legacy_count == 0:
        return text, "already_patched"
    if old_count == 0 and new_count == 0 and legacy_count == 1:
        return text.replace(LEGACY_BF16, NEW_BF16, 1), "repaired_legacy"
    raise ValueError(
        "ambiguous BF16 patch: "
        f"old_count={old_count}, new_count={new_count}, "
        f"legacy_count={legacy_count}"
    )


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_bytes(data)
    os.replace(temp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--receipt",
        default="results/trl_grpo_temperature_patch.json",
    )
    args = ap.parse_args()

    version = importlib.metadata.version("trl")
    if version != EXPECTED_TRL:
        raise SystemExit(
            f"refusing to patch trl {version}; expected {EXPECTED_TRL}"
        )

    package_root = Path(
        importlib.metadata.distribution("trl").locate_file("")
    ).resolve()
    path = package_root / "trl/trainer/grpo_trainer.py"
    utils_path = package_root / "trl/trainer/utils.py"
    if not path.is_file() or not utils_path.is_file():
        raise SystemExit(
            f"could not locate locked TRL sources under {package_root}"
        )

    before = path.read_bytes()
    text = before.decode()
    utils_before = utils_path.read_bytes()
    utils_text = utils_before.decode()

    # Preflight every replacement before mutating either installed file.
    try:
        after_text, temperature_status = plan_standard(
            text, OLD, NEW, "temperature")
        utils_after_text, logsumexp_status = plan_standard(
            utils_text, OLD_LOGSUMEXP, NEW_LOGSUMEXP, "logsumexp")
        utils_after_text, bf16_status = plan_bf16(utils_after_text)
    except ValueError as exc:
        raise SystemExit(f"refusing TRL patch: {exc}") from exc

    after = after_text.encode()
    utils_after = utils_after_text.encode()
    if after != before:
        atomic_write(path, after)
    if utils_after != utils_before:
        atomic_write(utils_path, utils_after)

    receipt = {
        "patch_schema": PATCH_SCHEMA,
        "patch_revision": PATCH_REVISION,
        "status": {
            "temperature_division": temperature_status,
            "chunked_logsumexp": logsumexp_status,
            "chunked_bf16_logsoftmax": bf16_status,
        },
        "trl_version": version,
        "temperature_division": {
            "path": str(path),
            "before_sha256": digest(before),
            "after_sha256": digest(after),
            "old": OLD.strip(),
            "new": NEW.strip(),
        },
        "selective_log_softmax": {
            "path": str(utils_path),
            "before_sha256": digest(utils_before),
            "after_sha256": digest(utils_after),
            "chunk_rows": 256,
        },
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
