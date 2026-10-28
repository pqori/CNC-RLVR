"""Strict G-code interpreter: document text -> canonical motion program.

This is the ONLY deterministic layer between the policy's output and
the voxel twin. It is deliberately small and fail-loud: a fixed RS274
subset with pinned semantics, hard line-numbered errors on anything
outside it, and geometry construction shared bit-for-bit with the
expander (cnc/expand.py: mk quantization, normalize_prims, arc
sampling), so a teacher document parses back to the exact sweep that
produced it -- audited, not assumed.

Dialect contract (what the model is told, and all it gets):
- one program: `G21 G90` before any motion, `M30` at the end
- `(SETUP <face>)` opens each setup, face in {Z+, Z-, X+, X-, Y+, Y-};
  coordinates are setup-local: X=u, Y=v, Z=0 at the entered face,
  NEGATIVE Z into the stock; position starts at X0 Y0 Z2 (clearance)
- `T<n> M6` tool change, `M3` spindle start: persist across setups;
  RPM is injected canonically from the selected tool; any motion before
  both is an error
- `G0` rapid; `G1` feed move; vertical G1 = drill stroke / plunge;
  slants = ramps. Canonical per-tool cutting/entry feed is injected
  deterministically
- `G2`/`G3` CW/CCW arc, I/J center offsets from the start point
  (R-form rejected); equal/omitted XY endpoint = full circle; a Z
  word makes it a helix
- `(comments)` are ignored except SETUP directives; everything else
  -- unknown G/M words, cutter comp, canned cycles, malformed numbers,
  inconsistent arc radii, out-of-range values -- is a hard error.
"""

from __future__ import annotations

import math
import re

from .expand import (CLEAR_W, ENTRY_FEED_FRAC, Segment,
                     SetupProgram, TOOL_BY_NUMBER, fq, mk, normalize_prims, q)
from .world import CUT_PARAMS, TOOLS

MAX_LINES = 20000
MAX_LINE_LEN = 200
MAX_SETUPS = 8
ARC_RADIUS_TOL = 0.005      # mm start-vs-end radius consistency
COORD_LIMIT = 500.0         # |X|,|Y|,|Z| sanity bound, mm
_HEADER_G = {17.0, 21.0, 90.0}

_SETUP_RE = re.compile(r"^\(\s*SETUP\s+(Z\+|Z-|X\+|X-|Y\+|Y-)\s*\)$",
                       re.IGNORECASE)
_COMMENT_RE = re.compile(r"\([^()]*\)")
_WORD_RE = re.compile(r"([A-Z])\s*([+-]?(?:\d+\.?\d*|\.\d+))")


class GcodeError(ValueError):
    def __init__(self, line_no: int, msg: str):
        super().__init__(f"line {line_no}: {msg}")
        self.line_no = line_no
        self.msg = msg


def parse_gcode(text: str) -> list:
    """Parse a document into list[SetupProgram] (the same structures
    the expander assembles). Raises GcodeError on any violation."""
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise GcodeError(len(lines), f"program exceeds {MAX_LINES} lines")

    program = []
    segs = None            # current setup's segment list
    cur = None             # prim list of the open segment
    tool_id = None
    rpm = None
    pos = None             # (u, v, w), resets each setup
    header_seen = False
    ended = False

    def emit(prim):
        nonlocal cur
        if cur is None:
            seg = Segment(tool_id, float(rpm), [])
            segs.append(seg)
            cur = seg.prims
        cur.append(prim)

    def motion_feed(w1, w2):
        """Canonical feed for the selected tool and move geometry."""
        fixed = CUT_PARAMS[tool_id].feed
        if TOOLS[tool_id].kind == "mill" and w2 > w1 + 1e-9:
            return fq(fixed * ENTRY_FEED_FRAC)
        return fixed

    for ln, raw in enumerate(lines, start=1):
        if len(raw) > MAX_LINE_LEN:
            raise GcodeError(ln, f"line exceeds {MAX_LINE_LEN} chars")
        line = raw.strip()
        if not line:
            continue
        if ended:
            raise GcodeError(ln, "content after M30")

        # ---- comments and the SETUP directive ----
        if line.startswith("("):
            m = _SETUP_RE.match(line)
            if m:
                if len(program) >= MAX_SETUPS:
                    raise GcodeError(ln, f"more than {MAX_SETUPS} setups")
                segs = []
                program.append(SetupProgram(m.group(1).upper(), segs))
                pos = (0.0, 0.0, CLEAR_W)
                cur = None
                continue
            if _COMMENT_RE.sub("", line).strip():
                raise GcodeError(ln, "malformed comment")
            continue

        body = _COMMENT_RE.sub(" ", line).upper()
        rest = _WORD_RE.sub("", body).strip()
        if rest:
            raise GcodeError(ln, f"unrecognized text {rest!r}")
        words = _WORD_RE.findall(body)
        if not words:
            continue

        # header lines may carry several G words (G21 G90, G17)
        g_vals = [float(s) for L, s in words if L == "G"]
        if g_vals and len(g_vals) == len(words) \
                and all(v in _HEADER_G for v in g_vals):
            header_seen = True
            continue

        letters = [w[0] for w in words]
        vals = {}
        for L, s in words:
            if L in vals:
                raise GcodeError(ln, f"duplicate word {L}")
            vals[L] = float(s)

        # ---- machine-state lines ----
        if "M" in vals:
            mcode = int(vals["M"])
            if vals["M"] != mcode:
                raise GcodeError(ln, "non-integer M code")
            if mcode == 30:
                if set(letters) - {"M"}:
                    raise GcodeError(ln, "M30 must stand alone")
                ended = True
                continue
            if mcode == 6:
                if "T" not in vals or set(letters) - {"T", "M"}:
                    raise GcodeError(ln, "tool change must be T<n> M6")
                tn = int(vals["T"])
                if vals["T"] != tn or tn not in TOOL_BY_NUMBER:
                    raise GcodeError(ln,
                                     f"unknown tool number T{vals['T']:g}")
                tool_id = TOOL_BY_NUMBER[tn]
                rpm = None
                cur = None
                continue
            if mcode == 3:
                if set(letters) - {"M"}:
                    raise GcodeError(ln, "spindle start must be M3")
                if tool_id is None:
                    raise GcodeError(ln, "spindle start before tool change")
                rpm = float(CUT_PARAMS[tool_id].rpm)
                cur = None
                continue
            if mcode == 5:
                if set(letters) - {"M"}:
                    raise GcodeError(ln, "M5 must stand alone")
                continue
            raise GcodeError(ln, f"unsupported M{mcode}")

        if "G" not in vals:
            raise GcodeError(ln, "line has no G or M command")
        gcode = vals["G"]
        if gcode not in (0.0, 1.0, 2.0, 3.0):
            raise GcodeError(ln, f"unsupported G{gcode:g}")

        # ---- motion ----
        if not header_seen:
            raise GcodeError(ln, "motion before G21 G90 header")
        if segs is None:
            raise GcodeError(ln, "motion before any (SETUP ...) block")
        if tool_id is None or rpm is None:
            raise GcodeError(ln, "motion before T..M6 and M3")
        allowed = {"G", "X", "Y", "Z"}
        if gcode in (2.0, 3.0):
            allowed |= {"I", "J"}
        extra = set(letters) - allowed
        if extra:
            if "R" in extra:
                raise GcodeError(ln, "R-form arcs not supported (use I/J)")
            raise GcodeError(ln, f"unexpected word {sorted(extra)[0]}")
        for axis in ("X", "Y", "Z"):
            if axis in vals and abs(vals[axis]) > COORD_LIMIT:
                raise GcodeError(ln, f"{axis} out of range")

        u2 = q(vals["X"]) if "X" in vals else pos[0]
        v2 = q(vals["Y"]) if "Y" in vals else pos[1]
        w2 = q(-vals["Z"]) if "Z" in vals else pos[2]

        if gcode == 0.0:
            emit(mk("rapid", pos[0], pos[1], pos[2], u2, v2, w2, 0.0))
            pos = (u2, v2, w2)
            continue

        # Feed and RPM are controlled conditions injected from the selected
        # tool; S and F are outside the policy's dialect.
        feed = motion_feed(pos[2], w2)

        if gcode == 1.0:
            if abs(u2 - pos[0]) < 1e-9 and abs(v2 - pos[1]) < 1e-9:
                if abs(w2 - pos[2]) > 1e-9:
                    emit(mk("drill", pos[0], pos[1], pos[2],
                            pos[0], pos[1], w2, feed))
            else:
                emit(mk("line", pos[0], pos[1], pos[2], u2, v2, w2, feed))
            pos = (u2, v2, w2)
            continue

        # G2/G3 arc
        if "I" not in vals or "J" not in vals:
            raise GcodeError(ln, "arc needs both I and J")
        cu = q(pos[0] + vals["I"])
        cv = q(pos[1] + vals["J"])
        r_start = math.hypot(pos[0] - cu, pos[1] - cv)
        r_end = math.hypot(u2 - cu, v2 - cv)
        if r_start < 1e-9:
            raise GcodeError(ln, "arc with zero radius (I0 J0)")
        if abs(r_start - r_end) > ARC_RADIUS_TOL:
            raise GcodeError(ln, f"arc radius mismatch "
                                 f"({r_start:.3f} vs {r_end:.3f})")
        emit(mk("arc", pos[0], pos[1], pos[2], u2, v2, w2, feed,
                cu=cu, cv=cv, ccw=(gcode == 3.0)))
        pos = (u2, v2, w2)

    if not ended:
        raise GcodeError(len(lines), "missing M30")
    if not program:
        raise GcodeError(len(lines), "no (SETUP ...) block")
    for sp in program:
        sp.segments = [Segment(s.tool_id, s.rpm, normalize_prims(s.prims))
                       for s in sp.segments]
        sp.segments = [s for s in sp.segments if s.prims]
        if not any(pr.kind != "rapid" for s in sp.segments
                   for pr in s.prims):
            raise GcodeError(len(lines),
                             f"setup {sp.orientation} cuts nothing")
    return program
