"""Independent per-face 2.5D heightfield reference engine (validation).

Second implementation of the material-removal sweep, the classic z-map
way: one remaining-height (in CELLS) per column. Exact for plans that
machine only from ONE face -- any face. Used solely to cross-check the
voxel engine: on the shared subfamily, per-column material state must
match EXACTLY. For side faces this also independently checks the
transpose/flip frame mapping in sim._face_view, the riskiest code path
side milling exercises.

Shared conventions (deliberately): grid pitch, cell centers, and the
center-inclusion rule (a cell is affected iff its CENTER is inside the
swept region). The geometry math itself (footprints, bands) is written
independently of sim.py.
"""

from __future__ import annotations

import math
import numpy as np

from .ir import Plan
from .world import STOCK, RES, TOOLS, Task
from .expand import expand_op_moves

# face -> indices into STOCK for the (u, v, s) extents of that face's
# local frame; must correspond to sim._face_view's axis order
_FACE_DIMS = {"Z": (0, 1, 2), "X": (1, 2, 0), "Y": (0, 2, 1)}


def heightfield_remaining(task: Task, plan: Plan, res: float = RES,
                          face: str = "Z+"):
    """Run the plan on a single-face heightfield. Returns per-column
    remaining material in CELLS, shape (nu, nv), in the same (u, v)
    frame as sim._face_view(stock, face). Raises on setups from any
    other face."""
    ud, vd, sd = _FACE_DIMS[face[0]]
    nu = int(round(STOCK[ud] / res))
    nv = int(round(STOCK[vd] / res))
    ns = int(round(STOCK[sd] / res))
    # cells cut per column, counted from the entered face; cell si has
    # center depth (si + 0.5) * res
    cut_cells = np.zeros((nu, nv), dtype=np.int32)

    us = (np.arange(nu) + 0.5) * res
    vs = (np.arange(nv) + 0.5) * res

    for setup in plan.setups:
        if setup.orientation != face:
            raise ValueError(f"reference engine run is {face}-only")
        for op in setup.ops:
            feat = task.feature(op.feature_ref)
            tool = TOOLS[op.tool_id]
            rt = tool.diameter / 2.0
            for m in expand_op_moves(op, feat):
                if m.kind == "rapid":
                    continue        # traverses remove nothing
                w = m.w
                # cells with center depth <= w  (center rule)
                n_cut = int(math.floor(w / res - 0.5)) + 1
                n_cut = max(0, min(n_cut, ns))
                if n_cut == 0:
                    continue
                u0 = max(0, int(math.floor((min(m.u1, m.u2) - rt) / res)))
                u1 = min(nu, int(math.ceil((max(m.u1, m.u2) + rt) / res)))
                v0 = max(0, int(math.floor((min(m.v1, m.v2) - rt) / res)))
                v1 = min(nv, int(math.ceil((max(m.v1, m.v2) + rt) / res)))
                if u1 <= u0 or v1 <= v0:
                    continue
                U, V = np.meshgrid(us[u0:u1], vs[v0:v1], indexing="ij")
                # independent point-to-segment distance implementation
                ax, ay = m.u1, m.v1
                bx, by = m.u2, m.v2
                abx, aby = bx - ax, by - ay
                denom = abx * abx + aby * aby
                if denom < 1e-12:
                    d2 = (U - ax) ** 2 + (V - ay) ** 2
                else:
                    t = ((U - ax) * abx + (V - ay) * aby) / denom
                    t = np.minimum(1.0, np.maximum(0.0, t))
                    d2 = (U - (ax + t * abx)) ** 2 + (V - (ay + t * aby)) ** 2
                inside = d2 <= rt * rt
                sub = cut_cells[u0:u1, v0:v1]
                sub[inside] = np.maximum(sub[inside], n_cut)

    return ns - cut_cells        # remaining cells per column
