#!/usr/bin/env python3
"""Preflight the installed GRPO memory patches on every launched rank."""

import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, ".")
from train.s3_grpo import (  # noqa: E402
    disable_accelerate_output_fp32,
    verify_trl_runtime_patch,
)


def check_selective_log_softmax(torch) -> None:
    import torch.nn.functional as F
    from trl.trainer.utils import selective_log_softmax

    device = torch.device("cuda", torch.cuda.current_device())
    for dtype, atol in (
            (torch.float32, 2e-5),
            (torch.bfloat16, 2e-2)):
        for top_k in (1, 3):
            source = torch.randn(
                2, 300, 257, device=device, dtype=dtype)
            index_shape = (2, 300) if top_k == 1 else (2, 300, top_k)
            index = torch.randint(
                0, 257, index_shape, device=device)

            actual_input = source.detach().clone().requires_grad_(True)
            expected_input = source.detach().clone().requires_grad_(True)
            actual = selective_log_softmax(actual_input, index)
            expected_index = (
                index.unsqueeze(-1) if top_k == 1 else index
            )
            expected = F.log_softmax(
                expected_input, dim=-1
            ).gather(dim=-1, index=expected_index)
            if top_k == 1:
                expected = expected.squeeze(-1)

            torch.testing.assert_close(
                actual.float(), expected.float(), atol=atol, rtol=atol)
            actual.float().sum().backward()
            expected.float().sum().backward()
            torch.testing.assert_close(
                actual_input.grad.float(),
                expected_input.grad.float(),
                atol=atol,
                rtol=atol,
            )


def check_accelerate_wrapper(torch) -> None:
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="bf16")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.linear(x)

    model = TinyModel().to(accelerator.device)
    prepared = accelerator.prepare_model(model)
    core = getattr(prepared, "module", prepared)
    x = torch.randn(8, 32, device=accelerator.device)
    with torch.no_grad():
        before = prepared(x)
    if before.dtype != torch.float32:
        raise AssertionError(
            f"expected Accelerate FP32 output before unwrap, got {before.dtype}"
        )
    if not disable_accelerate_output_fp32(core):
        raise AssertionError("failed to locate ConvertOutputsToFp32")
    with torch.no_grad():
        after = prepared(x)
    if after.dtype != torch.bfloat16:
        raise AssertionError(
            f"expected BF16 output after unwrap, got {after.dtype}"
        )
    loss = prepared(x).float().square().mean()
    loss.backward()
    if not all(
            p.grad is not None
            and math.isfinite(float(p.grad.float().norm().item()))
            for p in core.parameters()):
        raise AssertionError("non-finite or missing gradient after unwrap")
    accelerator.wait_for_everyone()


def check_stress(torch, tokens: int, vocab: int) -> None:
    from trl.trainer.utils import selective_log_softmax

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    logits = torch.zeros(
        1, tokens, vocab, device=device, dtype=torch.bfloat16,
        requires_grad=True)
    index = torch.zeros(1, tokens, device=device, dtype=torch.long)
    result = selective_log_softmax(logits, index)
    expected = -math.log(vocab)
    torch.testing.assert_close(
        result.float(),
        torch.full_like(result.float(), expected),
        atol=2e-2,
        rtol=2e-2,
    )
    result.float().sum().backward()
    grad_sample = None if logits.grad is None else logits.grad[
        :, ::256, ::1024
    ]
    if grad_sample is None or not torch.isfinite(grad_sample).all():
        raise AssertionError("stress test produced missing/non-finite gradient")
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    if peak_gib > 22.0:
        raise AssertionError(
            f"bounded-logit stress exceeded 22 GiB: {peak_gib:.2f} GiB"
        )
    print(f"bounded-logit stress peak: {peak_gib:.2f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--receipt",
        default="results/trl_grpo_temperature_patch.json",
    )
    ap.add_argument("--stress", action="store_true")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise SystemExit(
            f"preflight must run with exactly two ranks, got WORLD_SIZE={world}"
        )
    metadata = verify_trl_runtime_patch(Path(args.receipt))
    check_selective_log_softmax(torch)
    check_accelerate_wrapper(torch)
    if args.stress and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        check_stress(torch, tokens=24_576, vocab=151_936)
    print(
        f"GRPO runtime patch preflight passed on rank "
        f"{int(os.environ.get('RANK', '0'))}: {metadata}"
    )


if __name__ == "__main__":
    main()
