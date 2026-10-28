"""SINGLE source of truth for prompt formatting.

Every consumer -- SFT data builder, RAFT sampler, GRPO rollouts, eval --
imports these functions. Train/eval prompts must match byte-for-byte;
nothing else in the repo is allowed to build chat messages.

Two arms, one task prompt:
  arm="gcode" (MAIN): the policy writes one strict-dialect G-code
      document (cnc/gcode.py is the full grammar; the system prompt
      below is its human statement). Completion = the raw document.
  arm="ir" (ablation): the policy emits the typed-plan JSON the
      deterministic expander turns into motion (the scaffolding arm).
"""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from cnc.world import (TOOLS, CUT_PARAMS, SPINDLE_POWER_W,  # noqa: E402
                       tool_force_limit_n)
from cnc.expand import (TOOL_NUMBERS, ENTRY_FEED_FRAC, fq)  # noqa: E402


TRAINING_STACK = {"trl": "1.10.0", "vllm": "0.26.0"}

# Measured with the locked Qwen3 tokenizer over every frozen split:
# max full G-code chat = 21,193 tokens; max completion = 19,737 tokens.
# Keep one centralized budget with material headroom so SFT, RAFT, GRPO,
# and evaluation cannot silently disagree or truncate valid teacher-scale plans.
GCODE_MAX_SEQUENCE_TOKENS = 28_672
GCODE_MAX_COMPLETION_TOKENS = 24_576
IR_MAX_SEQUENCE_TOKENS = 8_192
IR_MAX_COMPLETION_TOKENS = 2_048


def max_sequence_tokens(arm: str) -> int:
    return GCODE_MAX_SEQUENCE_TOKENS if arm == "gcode" \
        else IR_MAX_SEQUENCE_TOKENS


def max_completion_tokens(arm: str) -> int:
    return GCODE_MAX_COMPLETION_TOKENS if arm == "gcode" \
        else IR_MAX_COMPLETION_TOKENS


def verify_training_stack():
    installed = {
        package: importlib.metadata.version(package)
        for package in TRAINING_STACK
    }
    mismatches = {
        package: (TRAINING_STACK[package], installed[package])
        for package in TRAINING_STACK
        if installed[package] != TRAINING_STACK[package]
    }
    if mismatches:
        raise RuntimeError(
            f"training stack differs from declared versions: {mismatches}")
    return installed


def verify_benchmark_manifest(path="data/benchmark.sha256"):
    """Fail closed if any frozen split differs from its manifest."""
    manifest = Path(path)
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relpath = line.split(maxsplit=1)
        relpath = relpath.lstrip("*")
        digest = hashlib.sha256(Path(relpath).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"frozen benchmark checksum mismatch: {relpath} "
                f"(expected {expected}, got {digest})")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _tool_table() -> str:
    rows = []
    for tid, n in sorted(TOOL_NUMBERS.items(), key=lambda kv: kv[1]):
        t = TOOLS[tid]
        fixed = CUT_PARAMS[tid]
        cc = "center-cutting" if t.center_cutting else "NOT center-cutting"
        rows.append(f"  T{n} {tid}: {t.kind} d={t.diameter:g}mm "
                    f"{t.flutes} flutes, flute length {t.flute_len:g}mm, "
                    f"stickout {t.stickout:g}mm, {cc}; S{fixed.rpm}")
    return "\n".join(rows)


def _gcode_tool_table() -> str:
    rows = [
        "T   ID    type   D     flute  stickout  center  auto_S auto_F  "
        "entry_F  force_limit"
    ]
    for tid, n in sorted(TOOL_NUMBERS.items(), key=lambda kv: kv[1]):
        t = TOOLS[tid]
        fixed = CUT_PARAMS[tid]
        center = "yes" if t.center_cutting else "no"
        if t.kind == "mill":
            entry = f"{fq(fixed.feed * ENTRY_FEED_FRAC):.1f}"
            force = f"{tool_force_limit_n(t):.1f}N"
        else:
            entry = "-"
            force = "spindle-only"
        rows.append(
            f"T{n:<2} {tid:<5} {t.kind:<6} {t.diameter:g}mm  "
            f"{t.flute_len:g}mm   {t.stickout:g}mm      {center:<3}     "
            f"{fixed.rpm:<5}  {fixed.feed:<7.1f} {entry:<7}  {force}"
        )
    return "\n".join(rows)


SYSTEM_GCODE = f"""You are a CNC machinist. Given a part description, output one G-code program and nothing else.

DIALECT
- Begin with G21 G90 and end with standalone M30.
- The only setup comment allowed is (SETUP F), where F is Z+, Z-, X+, X-, Y+, or Y-.
- A setup is a physical re-fixturing event. Position resets to X0 Y0 Z2 in the new face-local frame.
- Face-local Z=0 is the stock surface; negative Z enters material.
- Use one setup per required face and machine every feature on that face before changing setup.
- Tool change: T<n> M6. Then start the spindle with M3. RPM is injected automatically from the selected tool; S words are not allowed.
- Allowed motion is G0, G1, G2, and G3 only.
- G2/G3 use I,J center offsets. An equal or omitted XY endpoint denotes a full circle. Add Z for a helix.
- Feed is injected automatically; F words are not allowed.
- Before moving XY between features, retract with G0 Z2. Never rapid through stock.
- Do not emit prose, markdown, canned cycles, cutter compensation, R-form arcs, incremental coordinates, or unsupported comments.

TOOL CRIB
{_gcode_tool_table()}

STANDARD MACHINING PRACTICE
- Rough pockets in successive axial layers, then finish walls and floors. Do not jump directly to final depth and cut only the bottom layer.
- Begin mill entry at Z0. Enter with a helix or shallow ramp; every endmill is non-center-cutting and may not plunge vertically.
- Prefer entry angles at or below 4 degrees. The absolute maximum is 5 degrees.
- A helix requires cavity width at least 1.6 times tool diameter.
- Use the largest rigid tool that fits and reaches for bulk roughing.
- For deep pockets with tight corners, rough with a large tool, then rest-machine and finish with the appropriate small long-reach tool.
- A finish tool's radius must not exceed the requested corner radius.
- Cutting depth may use up to 100% of flute length, never more. Exceeding stickout crashes the holder.
- Use conservative radial and axial engagement. EM3L is a slender finishing/rest-machining tool, not a full-width deep rougher.
- Drills move and cut axially only.

VERIFIER SEMANTICS
- The simulator sweeps the physical tool through voxel stock. A lateral mill move removes material only over its flute span; a bottom-level path does not clear an uncut cap above the flutes.
- Rapids are checked against all remaining stock. Retracting through an uncut cap is a collision.
- All required material must be removed while preserving the part within +-0.1 mm.
- Cutting protected part material is a gouge.
- Load is measured from actual removed volume during one-second engaged-cut windows; air cuts do not reduce it.
- With MRR in mm^3/s: spindle power P=0.8*MRR watts and cutting force F=P/vc, where vc=pi*D*S/60000 m/s.
- Load utilization is u=max(F/tool force limit, P/{SPINDLE_POWER_W:g}W). Target u<=0.8; u>1 is not successful.
- Ramp, load, flute overreach, gouge severity, and completion quality are graded continuously, but unsafe programs are never counted correct.

GOAL
Produce a collision-free, load-safe program that completely machines the requested geometry. Among correct programs, prefer shorter cycle time and fewer setups."""

SYSTEM_IR = f"""You are a CNC process planner. Given a part description, output a machining plan as JSON, nothing else.

Rules:
- Plan schema: {{"setups": [{{"orientation": ..., "ops": [...]}}]}}
- orientation is one of: Z+, Z-, X+, X-, Y+, Y-. Each setup is an atomic physical re-fixturing event: the stock is removed and flipped to the named face, and coordinates reset to that face-local frame. There is no motion connecting different setups.
- Each feature states its face; its ops must be in a setup with that orientation. Use one setup per required face, group every feature on that face into it, and do not return to the same face later. Each setup change costs 3 minutes.
- op fields: op_type (pocket_rough|pocket_finish|slot|drill), feature_ref, tool_id, stepover, stepdown, entry (ramp|helix|plunge; plunge is for drill ops only), wall_stock.
- Tools and exact prescribed AL6061 parameters:
{_tool_table()}
- RPM and feed are omitted from the plan and injected canonically from the selected tool. EVERY endmill is non-center-cutting: plunge entries with mills are rejected; use helix (needs cavity room >= 1.6x tool diameter) or ramp. No cut deeper than the flute. Corner radius limits mill diameter. Drills only drill; mills only mill.
- Cutting load is graded from removed material. Let u=max(force/tool limit, power/spindle limit). Success requires u<=1; slender long-reach tools tolerate far less force. Load also has a continuous risk penalty: 0.05*phi, where z=max(0,(u-0.8)/0.2), phi=min(20,z^2) for z<=1 and phi=min(20,2z-1) above it. The deterministic expander uses 3-degree ramps/helixes, below the 4-degree zero-penalty margin; helix entry needs cavity room >= 1.6x tool diameter.
- The plan is graded by a machining simulator on: feasibility, material removed to +-0.1 mm tolerance, cycle time, and setup economy (setups beyond the distinct feature faces are penalized on top of their 3 minutes)."""


def to_messages(prompt_text: str, arm: str = "gcode") -> list:
    return [
        {"role": "system",
         "content": SYSTEM_GCODE if arm == "gcode" else SYSTEM_IR},
        {"role": "user", "content": prompt_text},
    ]


def completion_text(record: dict, arm: str = "gcode") -> str:
    """Canonical assistant completion for a benchmark record."""
    if arm == "gcode":
        return record["teacher_gcode"]
    plan = json.loads(json.dumps(record["teacher_plan"]))
    for setup in plan["setups"]:
        for op in setup["ops"]:
            op.pop("feed", None)
            op.pop("rpm", None)
    return json.dumps(plan, separators=(",", ":"))


def sft_row(record: dict, arm: str = "gcode") -> dict:
    """Benchmark record -> chat-format SFT row (TRL SFTTrainer format)."""
    return {
        "messages": to_messages(record["prompt"], arm)
        + [{"role": "assistant",
            "content": completion_text(record, arm)}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
