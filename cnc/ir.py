"""Typed intermediate representation: the policy's action space.

A Plan is an ordered list of Setups; a Setup is an orientation plus an
ordered list of Ops. This is the ONLY thing the LLM emits. The expander
(expand.py) turns it into toolpaths; the verifier (sim.py) grades it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

ORIENTATIONS = ["Z+", "Z-", "X+", "X-", "Y+", "Y-"]
OP_TYPES = ["pocket_rough", "pocket_finish", "slot", "drill"]
ENTRIES = ["ramp", "helix", "plunge"]


@dataclass
class Op:
    op_type: str
    feature_ref: str
    tool_id: str
    rpm: float
    feed: float          # mm/min
    stepover: float      # fraction of tool diameter, (0, 0.95]
    stepdown: float      # mm per depth pass
    entry: str
    wall_stock: float = 0.0  # mm left on walls (rough passes)

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type, "feature_ref": self.feature_ref,
            "tool_id": self.tool_id,
            "stepover": self.stepover, "stepdown": self.stepdown,
            "entry": self.entry, "wall_stock": self.wall_stock,
        }


@dataclass
class Setup:
    orientation: str
    ops: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"orientation": self.orientation,
                "ops": [op.to_dict() for op in self.ops]}


@dataclass
class Plan:
    setups: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"setups": [s.to_dict() for s in self.setups]}

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def n_setups(self) -> int:
        return len(self.setups)

    @property
    def all_ops(self):
        return [op for s in self.setups for op in s.ops]

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        setups = []
        for s in d["setups"]:
            ops = [Op(
                op_type=o["op_type"], feature_ref=o["feature_ref"],
                tool_id=o["tool_id"], rpm=o.get("rpm", 0.0),
                feed=o.get("feed", 0.0),
                stepover=o["stepover"], stepdown=o["stepdown"],
                entry=o["entry"], wall_stock=o["wall_stock"],
            ) for o in s["ops"]]
            setups.append(Setup(orientation=s["orientation"], ops=ops))
        return Plan(setups=setups)

    @staticmethod
    def from_json(text: str) -> "Plan":
        return Plan.from_dict(json.loads(text))


# JSON schema for grammar-locked decoding (xgrammar / vLLM guided).
OP_SCHEMA = {
    "type": "object",
    "properties": {
        "op_type": {"enum": OP_TYPES},
        "feature_ref": {"type": "string", "maxLength": 8},
        "tool_id": {"type": "string", "maxLength": 8},
        "stepover": {"type": "number", "minimum": 0.01, "maximum": 0.95},
        "stepdown": {"type": "number", "minimum": 0.05, "maximum": 20.0},
        "entry": {"enum": ENTRIES},
        "wall_stock": {"type": "number", "minimum": 0.0, "maximum": 5.0},
    },
    "required": ["op_type", "feature_ref", "tool_id",
                 "stepover", "stepdown", "entry", "wall_stock"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "setups": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "orientation": {"enum": ORIENTATIONS},
                    "ops": {"type": "array", "minItems": 1, "maxItems": 16,
                            "items": OP_SCHEMA},
                },
                "required": ["orientation", "ops"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["setups"],
    "additionalProperties": False,
}


def validate_syntax(text: str) -> tuple:
    """Parse + structural checks. Returns (plan_or_None, error_or_None).

    This is the parse gate: with grammar-locked decoding it should never
    fail; without it, failures are data (H3) and score reward -1.
    """
    try:
        d = json.loads(text)
    except Exception as e:
        return None, f"json: {e}"
    try:
        if not isinstance(d.get("setups"), list) or not d["setups"]:
            return None, "no setups"
        for s in d["setups"]:
            if s.get("orientation") not in ORIENTATIONS:
                return None, f"bad orientation {s.get('orientation')!r}"
            if not isinstance(s.get("ops"), list) or not s["ops"]:
                return None, "empty ops"
            for o in s["ops"]:
                if o.get("op_type") not in OP_TYPES:
                    return None, f"bad op_type {o.get('op_type')!r}"
                if o.get("entry") not in ENTRIES:
                    return None, f"bad entry {o.get('entry')!r}"
                for k in ("stepover", "stepdown", "wall_stock"):
                    if not isinstance(o.get(k), (int, float)):
                        return None, f"missing/non-numeric {k}"
                if not (0 < o["stepover"] <= 0.95):
                    return None, "stepover out of range"
                if not (0 < o["stepdown"] <= 20.0):
                    return None, "stepdown out of range"
        return Plan.from_dict(d), None
    except Exception as e:
        return None, f"structure: {e}"
