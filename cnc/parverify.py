"""Parallel plan verification across CPU cores.

Used by the RL inner loops (GRPO reward function, RAFT sampling), where
thousands of plans must be verified per step. Completions are grouped
by task seed so each worker builds a task's geometry once and verifies
all of that task's candidates against it; worker-local caches persist
across calls, so repeated seeds are warm.

Uses the 'spawn' start method: parent processes hold CUDA contexts
(vLLM / torch), and forking a CUDA process is undefined behavior.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time

_CACHE = {}          # per-worker: seed -> task/coarse/t_ref entry
_CACHE_MAX = 8
_SPOT_LOCK_PATHS = ()


def _init_worker(spot_lock_paths):
    global _SPOT_LOCK_PATHS
    _SPOT_LOCK_PATHS = tuple(spot_lock_paths)


def _acquire_spot_slot():
    """Acquire a node-wide, owner-death-safe fine-verifier slot."""
    import fcntl
    while True:
        for path in _SPOT_LOCK_PATHS:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                os.close(fd)
        time.sleep(0.05)


def _release_spot_slot(fd):
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _verify_group(args):
    """Worker: (seed, split, mode, frontier_spot, [text, ...]) -> results.

    mode "gcode": texts are raw documents (the student's arm);
    mode "ir": texts are typed-plan JSON (the ablation arm)."""
    seed, split, mode, frontier_spot, texts = args
    import sys
    sys.path.insert(0, ".")
    from cnc.ir import validate_syntax
    from cnc.world import SPOT_RES, make_task
    from cnc.sim import (Verifier, frontier_spot_required, is_ok,
                         reward_v9)
    from cnc.planner import template_plan, multitool_plan, grouped_plan

    if seed not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        task = make_task(seed, split)
        ver = Verifier(task)
        # t_ref ladder, identical to the freeze: first feasible rung
        t_ref = None
        for fn in (template_plan, multitool_plan, grouped_plan):
            rres = ver.verify(fn(task))
            if rres["feasible"]:
                t_ref = rres["time_s"]
                break
        if t_ref is None:       # cannot happen on frozen splits
            t_ref = 600.0
        _CACHE[seed] = {"task": task, "coarse": ver, "t_ref": t_ref}
    cached = _CACHE[seed]
    task, ver, t_ref = cached["task"], cached["coarse"], cached["t_ref"]

    out, spot_ver, spot_lock_fd = [], None, None
    try:
        for text in texts:
            if mode == "gcode":
                res = ver.verify_gcode(text)
                plan = None
            else:
                plan, err = validate_syntax(text)
                if plan is None:
                    out.append(dict(reward=-1.0, ok=False, feasible=False,
                                    fail=f"parse:{err}", clearance=0.0,
                                    linf=99.0, time_s=0.0, n_setups=0,
                                    path_valid=False, load_safe=True,
                                    ramp_safe=True, ramp_util=0.0,
                                    ramp_angle_deg=0.0,
                                    reach_safe=True, reach_util=0.0,
                                    gouge_safe=True, gouge_ratio=0.0,
                                    gouge_depth_mm=0.0,
                                    gouge_volume_mm3=0.0,
                                    force_util=0.0, power_util=0.0,
                                    spot_checked=False))
                    continue
                res = ver.verify(plan)
            coarse_reward = reward_v9(res, t_ref)
            final_res, final_reward = res, coarse_reward
            spot_checked = bool(
                frontier_spot and frontier_spot_required(res, t_ref)
            )
            spot_reward = None
            if spot_checked:
                if spot_ver is None:
                    # A 0.05 mm verifier is intentionally not cached: its
                    # dense arrays are large. Bound concurrent construction
                    # across coarse workers and reuse only within this task
                    # group, then release it immediately.
                    spot_lock_fd = _acquire_spot_slot()
                    spot_ver = Verifier(task, SPOT_RES)
                spot_res = spot_ver.verify_gcode(text) if mode == "gcode" \
                    else spot_ver.verify(plan)
                spot_reward = reward_v9(spot_res, t_ref)
                final_res, final_reward = spot_res, spot_reward
            out.append(dict(
                reward=final_reward,
                reward_coarse=coarse_reward,
                reward_spot=spot_reward,
                spot_checked=spot_checked,
                spot_ok=is_ok(final_res) if spot_checked else None,
                spot_linf=final_res["linf"] if spot_checked else None,
                ok=is_ok(final_res),
                feasible=final_res["feasible"],
                path_valid=final_res.get("path_valid", False),
                load_safe=final_res.get("load_safe", True),
                ramp_safe=final_res.get("ramp_safe", True),
                reach_safe=final_res.get("reach_safe", True),
                gouge_safe=final_res.get("gouge_safe", True),
                fail=final_res["fail"],
                clearance=final_res["clearance"],
                linf=final_res["linf"],
                time_s=final_res["time_s"],
                n_setups=final_res["n_setups"],
                force_util=final_res.get("force_util", 0.0),
                power_util=final_res.get("power_util", 0.0),
                ramp_util=final_res.get("ramp_util", 0.0),
                ramp_angle_deg=final_res.get("ramp_angle_deg", 0.0),
                reach_util=final_res.get("reach_util", 0.0),
                gouge_ratio=final_res.get("gouge_ratio", 0.0),
                gouge_depth_mm=final_res.get("gouge_depth_mm", 0.0),
                gouge_volume_mm3=final_res.get("gouge_volume_mm3", 0.0),
            ))
    finally:
        spot_ver = None
        if spot_lock_fd is not None:
            _release_spot_slot(spot_lock_fd)
    return seed, t_ref, out


class ParallelVerifier:
    """Persistent worker pool. verify_batch takes parallel lists of
    seeds and completion texts (arbitrary order, repeated seeds fine)
    and returns per-item result dicts in the same order. mode "gcode"
    grades raw documents; mode "ir" grades typed-plan JSON."""

    def __init__(self, workers: int = None, split: str = "train",
                 mode: str = "gcode", frontier_spot: bool = False,
                 spot_workers: int = 1, spot_lock_prefix: str = None):
        ctx = mp.get_context("spawn")
        self.split = split
        self.mode = mode
        self.frontier_spot = frontier_spot
        prefix = spot_lock_prefix or os.path.join(
            tempfile.gettempdir(), "cnc_rlvr_spot")
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        spot_lock_paths = [
            f"{prefix}.{i}.lock" for i in range(max(1, spot_workers))
        ]
        # Large GPU boxes often expose 96-192 host CPUs. One dense geometry
        # cache per process makes cpu_count()-2 an unsafe default there;
        # cap the pool unless the caller has measured host RAM explicitly.
        n_workers = workers or min(16, max(2, mp.cpu_count() - 2))
        self.pool = ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(spot_lock_paths,),
        )

    def verify_batch(self, seeds: list, texts: list) -> list:
        if len(seeds) != len(texts):
            raise ValueError("seeds and texts must have the same length")
        groups = {}
        for i, (s, t) in enumerate(zip(seeds, texts)):
            groups.setdefault(s, []).append((i, t))
        args = [(s, self.split, self.mode, self.frontier_spot,
                 [t for _, t in items])
                for s, items in groups.items()]
        results = [None] * len(seeds)
        for seed, t_ref, outs in self.pool.imap_unordered(
                _verify_group, args, chunksize=1):
            for (i, _), r in zip(groups[seed], outs):
                r["t_ref_s"] = t_ref
                results[i] = r
        return results

    def close(self):
        self.pool.terminate()
        self.pool.join()
