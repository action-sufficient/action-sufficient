#!/usr/bin/env python3
"""Random phi-family sweep for the two-cube grid toy (exact finite sums).

This script imports the base toy implementation and evaluates a diverse family of
*discrete* representations Z=phi(S,G), computing information-theoretic statistics:

  - H(A|S,G)
  - H(A|S,Z)
  - Delta_A = H(A|S,Z) - H(A|S,G)
  - H(V|S,Z)
  - Delta_V = H(V|S,Z) (since H(V|S,G) = 0)
  - H(A|S,V)
  - H(A|S,V,Z)
  - I(A;Z|S,V) = H(A|S,V) - H(A|S,V,Z)
  - I(A;V|S,Z) = H(A|S,Z) - H(A|S,V,Z)

Outputs:
  (1) CSV with all stats per phi
  (2) PNG scatter plot of (Delta_V vs Delta_A)

Coverage of the 4 regions:
  The generator intentionally includes templates that tend to yield:
   - value-sufficient / action-insufficient (e.g., Z=V, Z=(V,target) ...)
   - value-sufficient / action-sufficient (richer actor-like reps)
   - value-insufficient / action-sufficient (sign-based directional reps)
   - both insufficient (partial sign/dropped id + hashed compressions)

Place this file next to `grid_two_cube_bayes_entropy_pipeline.py`.

Usage:
  python sweep_random_phi_family.py --n 3 --num_random 300 --seed 0
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ----------------------
# Utilities
# ----------------------

def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def sgn(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def clip_abs(x: int, c: int) -> int:
    return min(abs(x), c)


def bucket_leq(x: int, thr: int) -> int:
    return x if x < thr else thr


def make_universal_hash(coeffs: np.ndarray, mod: int, bias: int = 0) -> Callable[[Tuple[int, ...]], int]:
    """Simple universal hash for small integer tuples."""
    coeffs = np.asarray(coeffs, dtype=np.int64)

    def _h(vals: Tuple[int, ...]) -> int:
        v = np.asarray(vals, dtype=np.int64)
        if v.size < coeffs.size:
            v = np.pad(v, (0, coeffs.size - v.size), constant_values=0)
        elif v.size > coeffs.size:
            v = v[:coeffs.size]
        return int((int(bias) + int(np.dot(coeffs, v))) % int(mod))

    return _h


def grouped_entropy_over_actions(
    action_mat: np.ndarray,
    group_map: Dict[Any, List[int]],
) -> float:
    ent_list: List[float] = []
    w_list: List[int] = []
    for idxs in group_map.values():
        avg = action_mat[idxs].mean(axis=0)
        ent_list.append(entropy_from_probs(avg))
        w_list.append(len(idxs))
    return float(np.average(np.asarray(ent_list), weights=np.asarray(w_list)))


def grouped_entropy_over_values(
    values: np.ndarray,
    group_map: Dict[Any, List[int]],
) -> float:
    ent_list: List[float] = []
    w_list: List[int] = []
    for idxs in group_map.values():
        vs = values[idxs]
        _, counts = np.unique(vs, return_counts=True)
        p = counts.astype(float) / counts.sum()
        ent_list.append(entropy_from_probs(p))
        w_list.append(len(idxs))
    return float(np.average(np.asarray(ent_list), weights=np.asarray(w_list)))


# ----------------------
# Phi generator
# ----------------------

@dataclass
class PhiSpec:
    phi_id: str
    desc: str
    fn: Callable[[int], Tuple[int, ...]]


def build_phi_family(
    rng: np.random.Generator,
    feats: Dict[str, np.ndarray],
    num_random: int,
) -> List[PhiSpec]:
    """Construct a diverse phi family with discrete outputs.

    feats contains integer arrays of length num_pairs for:
      si, holding, target, dx1, dy1, dx2, dy2, d_at, d_bg, V
    """

    holding = feats["holding"]
    target = feats["target"]
    dx1, dy1 = feats["dx1"], feats["dy1"]
    dx2, dy2 = feats["dx2"], feats["dy2"]
    d_at, d_bg = feats["d_at"], feats["d_bg"]
    V = feats["V"]

    phis: List[PhiSpec] = []

    # --- Baselines that anchor the 4 regions ---
    phis.append(PhiSpec(
        phi_id="value_only",
        desc="Z=(V)",
        fn=lambda i: (int(V[i]),),
    ))
    phis.append(PhiSpec(
        phi_id="distances",
        desc="Z=(holding, d_bg, d_at)",
        fn=lambda i: (int(holding[i]), int(d_bg[i]), int(d_at[i])),
    ))
    phis.append(PhiSpec(
        phi_id="actor_like",
        desc="Z=(holding,target,dx1,dy1,dx2,dy2)",
        fn=lambda i: (int(holding[i]), int(target[i]), int(dx1[i]), int(dy1[i]), int(dx2[i]), int(dy2[i])),
    ))
    phis.append(PhiSpec(
        phi_id="signs",
        desc="Z=(holding,target,sgn(dx1),sgn(dy1),sgn(dx2),sgn(dy2))",
        fn=lambda i: (int(holding[i]), int(target[i]), sgn(int(dx1[i])), sgn(int(dy1[i])), sgn(int(dx2[i])), sgn(int(dy2[i]))),
    ))
    phis.append(PhiSpec(
        phi_id="dx_sign",
        desc="Z=(holding,target,sgn(dx2))",
        fn=lambda i: (int(holding[i]), int(target[i]), sgn(int(dx2[i]))),
    ))

    # --- Transforms ---
    def t_raw(x: int) -> int:
        return int(x)

    def t_sign(x: int) -> int:
        return sgn(int(x))

    def t_abs(x: int) -> int:
        return abs(int(x))

    def t_clip1(x: int) -> int:
        return clip_abs(int(x), 1)

    def t_clip2(x: int) -> int:
        return clip_abs(int(x), 2)

    def t_clip3(x: int) -> int:
        return clip_abs(int(x), 3)

    def t_parity(x: int) -> int:
        return abs(int(x)) % 2

    def t_sgn_bucket2(x: int) -> int:
        xi = int(x)
        return sgn(xi) * bucket_leq(abs(xi), 2)

    def t_sgn_bucket3(x: int) -> int:
        xi = int(x)
        return sgn(xi) * bucket_leq(abs(xi), 3)

    def t_d_raw(x: int) -> int:
        return int(x)

    def t_d_bucket2(x: int) -> int:
        return bucket_leq(int(x), 2)

    def t_d_bucket3(x: int) -> int:
        return bucket_leq(int(x), 3)

    def t_d_bucket4(x: int) -> int:
        return bucket_leq(int(x), 4)

    def t_v_raw(x: int) -> int:
        return int(x)

    def t_v_bucket3(x: int) -> int:
        cost = -int(x)
        b = bucket_leq(cost, 3)
        return -b

    dir_transforms = [t_raw, t_sign, t_abs, t_clip1, t_clip2, t_clip3, t_parity, t_sgn_bucket2, t_sgn_bucket3]
    dist_transforms = [t_d_raw, t_d_bucket2, t_d_bucket3, t_d_bucket4, t_parity]
    value_transforms = [t_v_raw, t_v_bucket3]

    ACTOR_HASH_MODS = [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    DIST_HASH_MODS = [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64]

    dir_feats = ["dx1", "dy1", "dx2", "dy2"]

    def get_feat(name: str, i: int) -> int:
        return int(feats[name][i])

    # --- Template-driven random phis ---
    for k in range(num_random):
        templates = [
            "value_plus",
            "dist_coarse",
            "dir_subset",
            "dir_coarse",
            "mixed_dir_dist",
            "phase_split",
            "proj_mod",
            "two_hash",
            "hashed_actor",
            "hashed_dist",
            "drop_id",
        ]
        weights = np.array([0.14, 0.16, 0.18, 0.16, 0.12, 0.08, 0.08, 0.04, 0.08, 0.08, 0.08], dtype=float)
        weights = weights / weights.sum()
        template = rng.choice(templates, p=weights)

        if template == "value_plus":
            use_target = bool(rng.integers(0, 2))
            v_tf = rng.choice(value_transforms)
            parts = [("holding", None)]
            if use_target:
                parts.append(("target", None))
            parts.append(("V", v_tf))

            def make_fn(parts_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    out: List[int] = []
                    for name, tf in parts_local:
                        x = get_feat(name, i)
                        out.append(int(tf(x)) if tf is not None else int(x))
                    return tuple(out)
                return _phi

            phi_id = f"rand_value_plus_{k}"
            desc = "Z=" + "+".join([p[0] + (":" + (p[1].__name__ if p[1] else "raw")) for p in parts])
            phis.append(PhiSpec(phi_id, desc, make_fn(list(parts))))

        elif template == "dist_coarse":
            tf1 = rng.choice(dist_transforms)
            tf2 = rng.choice(dist_transforms)
            include_target = bool(rng.integers(0, 2))

            parts = [("holding", None)]
            if include_target:
                parts.append(("target", None))
            parts.append(("d_bg", tf1))
            parts.append(("d_at", tf2))

            def make_fn(parts_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    out: List[int] = []
                    for name, tf in parts_local:
                        x = get_feat(name, i)
                        out.append(int(tf(x)) if tf is not None else int(x))
                    return tuple(out)
                return _phi

            phi_id = f"rand_dist_{k}"
            desc = "Z=" + "+".join([p[0] + (":" + (p[1].__name__ if p[1] else "raw")) for p in parts])
            phis.append(PhiSpec(phi_id, desc, make_fn(list(parts))))

        elif template == "dir_subset":
            include_target = bool(rng.integers(0, 2))
            num = int(rng.integers(1, 5))
            chosen = rng.choice(dir_feats, size=num, replace=False)

            parts = [("holding", None)]
            if include_target:
                parts.append(("target", None))
            for name in chosen:
                parts.append((name, t_sign))

            def make_fn(parts_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    out: List[int] = []
                    for name, tf in parts_local:
                        x = get_feat(name, i)
                        out.append(int(tf(x)) if tf is not None else int(x))
                    return tuple(out)
                return _phi

            phi_id = f"rand_dir_subset_{k}"
            desc = "Z=" + "+".join([p[0] + (":" + (p[1].__name__ if p[1] else "raw")) for p in parts])
            phis.append(PhiSpec(phi_id, desc, make_fn(list(parts))))

        elif template == "dir_coarse":
            include_target = True
            tf_dx1 = rng.choice(dir_transforms)
            tf_dy1 = rng.choice(dir_transforms)
            tf_dx2 = rng.choice(dir_transforms)
            tf_dy2 = rng.choice(dir_transforms)

            def make_phi(include_target_local, tf_dx1_local, tf_dy1_local, tf_dx2_local, tf_dy2_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    return (
                        int(holding[i]),
                        int(target[i]) if include_target_local else 0,
                        int(tf_dx1_local(int(dx1[i]))),
                        int(tf_dy1_local(int(dy1[i]))),
                        int(tf_dx2_local(int(dx2[i]))),
                        int(tf_dy2_local(int(dy2[i]))),
                    )
                return _phi

            phi_id = f"rand_dir_coarse_{k}"
            desc = f"Z=(holding,target,dx1:{tf_dx1.__name__},dy1:{tf_dy1.__name__},dx2:{tf_dx2.__name__},dy2:{tf_dy2.__name__})"
            phis.append(PhiSpec(phi_id, desc, make_phi(include_target, tf_dx1, tf_dy1, tf_dx2, tf_dy2)))

        elif template == "mixed_dir_dist":
            include_target = bool(rng.integers(0, 2))
            num_dir = int(rng.integers(1, 5))
            chosen_dir = rng.choice(dir_feats, size=num_dir, replace=False)
            dir_tfs = [rng.choice([t_sign, t_parity, t_clip1, t_sgn_bucket2, t_sgn_bucket3]) for _ in range(num_dir)]
            tf_bg = rng.choice(dist_transforms)
            tf_at = rng.choice(dist_transforms)

            parts = [("holding", None)]
            if include_target:
                parts.append(("target", None))
            for name, tf in zip(chosen_dir, dir_tfs):
                parts.append((name, tf))
            parts.append(("d_bg", tf_bg))
            parts.append(("d_at", tf_at))

            def make_fn(parts_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    out: List[int] = []
                    for name, tf in parts_local:
                        x = get_feat(name, i)
                        out.append(int(tf(x)) if tf is not None else int(x))
                    return tuple(out)
                return _phi

            phi_id = f"rand_mixed_dir_dist_{k}"
            desc = "Z=" + "+".join([p[0] + (":" + (p[1].__name__ if p[1] else "raw")) for p in parts])
            phis.append(PhiSpec(phi_id, desc, make_fn(list(parts))))

        elif template == "phase_split":
            sent = -9
            tf_a = rng.choice([t_sign, t_sgn_bucket2, t_parity])
            tf_b = rng.choice([t_sign, t_sgn_bucket2, t_parity])
            tf_da = rng.choice([t_d_bucket2, t_d_bucket3, t_d_bucket4])
            tf_db = rng.choice([t_d_bucket2, t_d_bucket3, t_d_bucket4])

            def make_phi(sent_local, tf_a_local, tf_b_local, tf_da_local, tf_db_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    h = int(holding[i])
                    t = int(target[i])
                    if h == 0:
                        return (h, t,
                                int(tf_a_local(int(dx1[i]))), int(tf_a_local(int(dy1[i]))),
                                sent_local, sent_local,
                                int(tf_da_local(int(d_at[i]))))
                    else:
                        return (h, t,
                                sent_local, sent_local,
                                int(tf_b_local(int(dx2[i]))), int(tf_b_local(int(dy2[i]))),
                                int(tf_db_local(int(d_bg[i]))))
                return _phi

            phi_id = f"rand_phase_split_{k}"
            desc = f"Z=phase_split(tf_a={tf_a.__name__},tf_b={tf_b.__name__},d_at={tf_da.__name__},d_bg={tf_db.__name__})"
            phis.append(PhiSpec(phi_id, desc, make_phi(sent, tf_a, tf_b, tf_da, tf_db)))

        elif template == "proj_mod":
            pool = ["holding", "target", "dx1", "dy1", "dx2", "dy2", "d_at", "d_bg", "V"]
            m = int(rng.integers(3, len(pool) + 1))
            chosen = rng.choice(pool, size=m, replace=False).tolist()
            mod = int(rng.choice(ACTOR_HASH_MODS))
            coeffs = rng.integers(1, 2**31 - 1, size=m, dtype=np.int64)
            bias = int(rng.integers(0, 10**6))
            keep_prefix = rng.choice(["none", "holding", "holding_target"], p=[0.30, 0.35, 0.35])

            def make_phi(chosen_local, coeffs_local, bias_local, mod_local, keep_prefix_local):
                chosen_local = tuple(chosen_local)
                coeffs_local = np.array(coeffs_local, dtype=np.int64)

                def _phi(i: int) -> Tuple[int, ...]:
                    vec = np.array([get_feat(nm, i) for nm in chosen_local], dtype=np.int64)
                    hv = int((bias_local + int(np.dot(coeffs_local, vec))) % mod_local)
                    if keep_prefix_local == "none":
                        return (hv,)
                    if keep_prefix_local == "holding":
                        return (int(holding[i]), hv)
                    return (int(holding[i]), int(target[i]), hv)
                return _phi

            phi_id = f"rand_proj_mod_{k}"
            desc = f"Z=proj_mod(chosen={'+'.join(chosen)},mod={mod},prefix={keep_prefix})"
            phis.append(PhiSpec(phi_id, desc, make_phi(chosen, coeffs, bias, mod, keep_prefix)))

        elif template == "two_hash":
            mod1 = int(rng.choice(DIST_HASH_MODS))
            mod2 = int(rng.choice(DIST_HASH_MODS))
            c1 = rng.integers(1, 2**31 - 1, size=3, dtype=np.int64)
            c2 = rng.integers(1, 2**31 - 1, size=3, dtype=np.int64)
            b1 = int(rng.integers(0, 10**6))
            b2 = int(rng.integers(0, 10**6))
            h1 = make_universal_hash(c1, mod1, bias=b1)
            h2 = make_universal_hash(c2, mod2, bias=b2)
            keep_target = bool(rng.integers(0, 2))

            def make_phi(h1_local, h2_local, keep_target_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    code1 = int(h1_local((int(dx1[i]), int(dy1[i]), int(d_at[i]))))
                    code2 = int(h2_local((int(dx2[i]), int(dy2[i]), int(d_bg[i]))))
                    if keep_target_local:
                        return (int(holding[i]), int(target[i]), code1, code2)
                    return (int(holding[i]), code1, code2)
                return _phi

            phi_id = f"rand_two_hash_{k}"
            desc = f"Z=two_hash(mod1={mod1},mod2={mod2},keep_target={keep_target})"
            phis.append(PhiSpec(phi_id, desc, make_phi(h1, h2, keep_target)))

        elif template == "hashed_actor":
            mod = int(rng.choice(ACTOR_HASH_MODS))
            coeffs = rng.integers(1, 2**31 - 1, size=6, dtype=np.int64)
            bias = int(rng.integers(0, 10**6))
            h = make_universal_hash(coeffs, mod, bias=bias)
            keep_prefix = rng.choice(["none", "holding", "holding_target"], p=[0.25, 0.35, 0.40])

            def make_phi(h_local, keep_prefix_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    base_tup = (int(dx1[i]), int(dy1[i]), int(dx2[i]), int(dy2[i]), int(d_bg[i]), int(d_at[i]))
                    hv = int(h_local(base_tup))
                    if keep_prefix_local == "none":
                        return (hv,)
                    if keep_prefix_local == "holding":
                        return (int(holding[i]), hv)
                    return (int(holding[i]), int(target[i]), hv)
                return _phi

            phi_id = f"rand_hashed_actor_{k}"
            desc = f"Z=hash_actor(mod={mod},prefix={keep_prefix})"
            phis.append(PhiSpec(phi_id, desc, make_phi(h, keep_prefix)))

        elif template == "hashed_dist":
            mod = int(rng.choice(DIST_HASH_MODS))
            coeffs = rng.integers(1, 2**31 - 1, size=3, dtype=np.int64)
            bias = int(rng.integers(0, 10**6))
            h = make_universal_hash(coeffs, mod, bias=bias)
            keep_target = bool(rng.integers(0, 2))

            def make_phi(h_local, keep_target_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    base_tup = (int(d_bg[i]), int(d_at[i]), int(V[i]))
                    hv = int(h_local(base_tup))
                    if keep_target_local:
                        return (int(holding[i]), int(target[i]), hv)
                    return (int(holding[i]), hv)
                return _phi

            phi_id = f"rand_hashed_dist_{k}"
            desc = f"Z=hash_dist(mod={mod},keep_target={keep_target})"
            phis.append(PhiSpec(phi_id, desc, make_phi(h, keep_target)))

        elif template == "drop_id":
            tf_dx1 = rng.choice([t_sign, t_parity, t_clip1])
            tf_dy1 = rng.choice([t_sign, t_parity, t_clip1])
            tf_dx2 = rng.choice([t_sign, t_parity, t_clip1])
            tf_dy2 = rng.choice([t_sign, t_parity, t_clip1])

            def make_phi(tf_dx1_local, tf_dy1_local, tf_dx2_local, tf_dy2_local):
                def _phi(i: int) -> Tuple[int, ...]:
                    return (
                        int(holding[i]),
                        int(tf_dx1_local(int(dx1[i]))),
                        int(tf_dy1_local(int(dy1[i]))),
                        int(tf_dx2_local(int(dx2[i]))),
                        int(tf_dy2_local(int(dy2[i]))),
                    )
                return _phi

            phi_id = f"rand_drop_id_{k}"
            desc = f"Z=(holding,drop_target,dx1:{tf_dx1.__name__},dy1:{tf_dy1.__name__},dx2:{tf_dx2.__name__},dy2:{tf_dy2.__name__})"
            phis.append(PhiSpec(phi_id, desc, make_phi(tf_dx1, tf_dy1, tf_dx2, tf_dy2)))

    return phis


# ----------------------
# Main sweep logic
# ----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_module", type=str, default="grid_two_cube_bayes_entropy_pipeline",
                    help="Python module name for the base file")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--num_random", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="phi_sweep.csv")
    ap.add_argument("--out_png", type=str, default="deltaA_deltaV.png")
    ap.add_argument("--tol", type=float, default=1e-9, help="tolerance to call a gap ~0")

    args = ap.parse_args()

    # Import base module
    base = importlib.import_module(args.base_module)

    # Build optimal action matrix (v4 interface: only n required)
    data = base.build_optimal_action_matrix(n=args.n)

    meta = data["meta"]
    rows = data["rows"]
    states = data["states"]
    goals = data["goals"]
    action_mat = data["action"]
    V = data["V"].astype(int)

    num_pairs = int(rows.shape[0])

    # --- Precompute per-pair integer features ---
    si = rows[:, 0].astype(int)
    gi = rows[:, 1].astype(int)

    ax = states[si, 0]
    ay = states[si, 1]
    rx = states[si, 2]
    ry = states[si, 3]
    bx = states[si, 4]
    by = states[si, 5]
    holding = states[si, 6]

    target = goals[gi, 0]
    gx = goals[gi, 1]
    gy = goals[gi, 2]

    # Target block position on floor (when not holding target)
    is_red = (target == base.TARGET_RED)
    tx_floor = np.where(is_red, rx, bx)
    ty_floor = np.where(is_red, ry, by)

    # Holding target? Then effective target position is agent pos
    holding_target = ((holding == base.HOLD_RED) & is_red) | ((holding == base.HOLD_BLUE) & (~is_red))
    tx_eff = np.where(holding_target, ax, tx_floor)
    ty_eff = np.where(holding_target, ay, ty_floor)

    # Phase-aware geometry
    dx1 = (tx_eff - ax).astype(int)
    dy1 = (ty_eff - ay).astype(int)
    dx2 = (gx - tx_eff).astype(int)
    dy2 = (gy - ty_eff).astype(int)

    d_at = (np.abs(dx1) + np.abs(dy1)).astype(int)
    d_bg = (np.abs(dx2) + np.abs(dy2)).astype(int)

    feats: Dict[str, np.ndarray] = {
        "si": si,
        "holding": holding.astype(int),
        "target": target.astype(int),
        "dx1": dx1,
        "dy1": dy1,
        "dx2": dx2,
        "dy2": dy2,
        "d_at": d_at,
        "d_bg": d_bg,
        "V": V,
    }

    # --- Base constants ---
    H_A_SG = float(np.mean([entropy_from_probs(action_mat[i]) for i in range(num_pairs)]))
    H_V_SG = 0.0

    # Precompute H(A|S,V) and groups_sv
    groups_sv: Dict[Tuple[int, int], List[int]] = {}
    for i in range(num_pairs):
        key = (int(si[i]), int(V[i]))
        groups_sv.setdefault(key, []).append(i)
    H_A_SV = grouped_entropy_over_actions(action_mat, groups_sv)

    # Build phi family
    rng = np.random.default_rng(args.seed)
    phi_specs = build_phi_family(rng, feats, num_random=args.num_random)

    results: List[Dict[str, Any]] = []

    # Evaluate each phi
    pbar = tqdm(phi_specs, desc="Sweeping phi", unit="phi")
    for ps in pbar:
        # Build Z keys
        Z = [ps.fn(i) for i in range(num_pairs)]

        # Group by (S,Z)
        groups_sz: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        for i in range(num_pairs):
            key = (int(si[i]), Z[i])
            groups_sz.setdefault(key, []).append(i)

        H_A_SZ = grouped_entropy_over_actions(action_mat, groups_sz)
        H_V_SZ = grouped_entropy_over_values(V, groups_sz)

        # Group by (S,V,Z)
        groups_svz: Dict[Tuple[int, int, Tuple[int, ...]], List[int]] = {}
        for i in range(num_pairs):
            key = (int(si[i]), int(V[i]), Z[i])
            groups_svz.setdefault(key, []).append(i)

        H_A_SVZ = grouped_entropy_over_actions(action_mat, groups_svz)

        I_A_Z_SV = H_A_SV - H_A_SVZ
        I_A_V_SZ = H_A_SZ - H_A_SVZ

        delta_A = H_A_SZ - H_A_SG
        delta_V = H_V_SZ - H_V_SG

        residual = H_A_SZ - (H_A_SV - I_A_Z_SV + I_A_V_SZ)

        value_suff = abs(delta_V) <= args.tol
        action_suff = abs(delta_A) <= args.tol

        if value_suff and action_suff:
            quad = "value_suff & action_suff"
        elif value_suff and (not action_suff):
            quad = "value_suff & action_insuff"
        elif (not value_suff) and action_suff:
            quad = "value_insuff & action_suff"
        else:
            quad = "value_insuff & action_insuff"

        results.append({
            "phi_id": ps.phi_id,
            "phi_desc": ps.desc,
            "num_pairs": num_pairs,
            "num_groups_sz": len(groups_sz),
            "num_groups_sv": len(groups_sv),
            "num_groups_svz": len(groups_svz),
            "H_A_SG": H_A_SG,
            "H_A_SZ": H_A_SZ,
            "Delta_A": delta_A,
            "H_V_SG": H_V_SG,
            "H_V_SZ": H_V_SZ,
            "Delta_V": delta_V,
            "H_A_SV": H_A_SV,
            "H_A_SVZ": H_A_SVZ,
            "I_A_Z_given_SV": I_A_Z_SV,
            "I_A_V_given_SZ": I_A_V_SZ,
            "identity_residual": residual,
            "value_sufficient": value_suff,
            "action_sufficient": action_suff,
            "quadrant": quad,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    # Print quadrant coverage
    print("\n" + "=" * 50)
    print("Quadrant counts:")
    quad_counts = df["quadrant"].value_counts().to_dict()
    for k, v in quad_counts.items():
        print(f"  {k}: {v}")

    # Correlation diagnostics
    if len(df) >= 2:
        dv = df["Delta_V"].to_numpy(dtype=float)
        da = df["Delta_A"].to_numpy(dtype=float)
        pearson = float(np.corrcoef(dv, da)[0, 1])
        dv_r = pd.Series(dv).rank(method="average").to_numpy()
        da_r = pd.Series(da).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(dv_r, da_r)[0, 1])
        print(f"Pearson corr(Delta_V, Delta_A): {pearson:.4f}")
        print(f"Spearman corr(Delta_V, Delta_A): {spearman:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Delta_V"].values, df["Delta_A"].values, s=15, alpha=0.6)
    plt.xlabel("Delta_V = H(V|S,Z)")
    plt.ylabel("Delta_A = H(A|S,Z) - H(A|S,G)")
    plt.title(f"Two-cube toy (n={args.n}): value gap vs action gap")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # Annotate baselines
    for base_name in ["value_only", "distances", "actor_like", "signs", "dx_sign"]:
        sub = df[df["phi_id"] == base_name]
        if len(sub) == 1:
            x = float(sub["Delta_V"].iloc[0])
            y = float(sub["Delta_A"].iloc[0])
            plt.scatter([x], [y], s=80, marker="x", linewidths=2)
            plt.annotate(f" {base_name}", (x, y), fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)

    print(f"\nSaved CSV to: {args.out_csv}")
    print(f"Saved PNG to: {args.out_png}")


if __name__ == "__main__":
    main()