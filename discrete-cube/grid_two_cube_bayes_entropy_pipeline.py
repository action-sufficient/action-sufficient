#!/usr/bin/env python3
"""Two-cube discrete Grid Manipulation toy for exact Bayes-risk style entropy calculations.

State filtering rules:
- holding == NONE: red, blue, goal must all be distinct positions. Agent can be anywhere.
- holding != NONE: agent, remaining floor cube, goal must all be distinct positions.

No V!=0 filtering needed (already implied by goal not overlapping cubes).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# ---------------------------
# Basic definitions
# ---------------------------

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "PLACE"]
A2I = {a: i for i, a in enumerate(ACTIONS)}

HOLD_NONE = 0
HOLD_RED = 1
HOLD_BLUE = 2

TARGET_RED = 1
TARGET_BLUE = 2

OFF_FLOOR = -1


def _sgn(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


@dataclass(frozen=True)
class State:
    ax: int
    ay: int
    rx: int
    ry: int
    bx: int
    by: int
    holding: int  # 0 none, 1 red, 2 blue

    def agent_pos(self) -> Tuple[int, int]:
        return (self.ax, self.ay)

    def red_pos(self) -> Tuple[int, int]:
        return (self.rx, self.ry)

    def blue_pos(self) -> Tuple[int, int]:
        return (self.bx, self.by)

    def red_on_floor(self) -> bool:
        return self.holding != HOLD_RED

    def blue_on_floor(self) -> bool:
        return self.holding != HOLD_BLUE

    def floor_occupied(self, p: Tuple[int, int]) -> bool:
        if self.red_on_floor() and (self.rx, self.ry) == p:
            return True
        if self.blue_on_floor() and (self.bx, self.by) == p:
            return True
        return False

    def red_pos_effective(self) -> Tuple[int, int]:
        if self.holding == HOLD_RED:
            return self.agent_pos()
        return self.red_pos()

    def blue_pos_effective(self) -> Tuple[int, int]:
        if self.holding == HOLD_BLUE:
            return self.agent_pos()
        return self.blue_pos()

    def target_on_floor_at_goal(self, g: "Goal") -> bool:
        gp = g.goal_pos()
        if g.target == TARGET_RED:
            return self.red_on_floor() and (self.rx, self.ry) == gp
        else:
            return self.blue_on_floor() and (self.bx, self.by) == gp


@dataclass(frozen=True)
class Goal:
    target: int  # 1 red, 2 blue
    gx: int
    gy: int

    def goal_pos(self) -> Tuple[int, int]:
        return (self.gx, self.gy)


def target_pos_effective(s: State, g: Goal) -> Tuple[int, int]:
    if g.target == TARGET_RED:
        return s.red_pos_effective()
    else:
        return s.blue_pos_effective()


# ---------------------------
# Dynamics
# ---------------------------

def is_success(s: State, g: Goal) -> bool:
    if g.target == TARGET_RED:
        return (s.holding != HOLD_RED) and s.target_on_floor_at_goal(g)
    else:
        return (s.holding != HOLD_BLUE) and s.target_on_floor_at_goal(g)


def _move_clipped(ax: int, ay: int, act: str, n: int) -> Tuple[int, int]:
    if act == "LEFT":
        return (max(0, ax - 1), ay)
    if act == "RIGHT":
        return (min(n - 1, ax + 1), ay)
    if act == "DOWN":
        return (ax, max(0, ay - 1))
    if act == "UP":
        return (ax, min(n - 1, ay + 1))
    return (ax, ay)


def step_state(s: State, act: str, n: int) -> State:
    if act in ("UP", "DOWN", "LEFT", "RIGHT"):
        nax, nay = _move_clipped(s.ax, s.ay, act, n)
        return State(nax, nay, s.rx, s.ry, s.bx, s.by, s.holding)

    if act == "PICK":
        if s.holding != HOLD_NONE:
            return s
        a = s.agent_pos()
        if s.red_on_floor() and a == s.red_pos():
            return State(s.ax, s.ay, OFF_FLOOR, OFF_FLOOR, s.bx, s.by, HOLD_RED)
        if s.blue_on_floor() and a == s.blue_pos():
            return State(s.ax, s.ay, s.rx, s.ry, OFF_FLOOR, OFF_FLOOR, HOLD_BLUE)
        return s

    if act == "PLACE":
        if s.holding == HOLD_NONE:
            return s
        a = s.agent_pos()
        if s.floor_occupied(a):
            return s
        if s.holding == HOLD_RED:
            return State(s.ax, s.ay, s.ax, s.ay, s.bx, s.by, HOLD_NONE)
        else:
            return State(s.ax, s.ay, s.rx, s.ry, s.ax, s.ay, HOLD_NONE)

    raise ValueError(f"unknown action: {act}")


_GRAPH_CACHE: Dict[int, Dict[str, Any]] = {}
_GOAL_DIST_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}


def _get_graph(n: int) -> Dict[str, Any]:
    if n in _GRAPH_CACHE:
        return _GRAPH_CACHE[n]

    states = enumerate_states(n)
    idx_of = {s: i for i, s in enumerate(states)}

    num_states = len(states)
    num_actions = len(ACTIONS)
    succ = np.empty((num_states, num_actions), dtype=np.int32)
    pred: List[List[int]] = [[] for _ in range(num_states)]

    for i, s in enumerate(states):
        for ai, act in enumerate(ACTIONS):
            sp = step_state(s, act, n)
            j = idx_of[sp]
            succ[i, ai] = j
            pred[j].append(i)

    g = {"states": states, "idx_of": idx_of, "succ": succ, "pred": pred}
    _GRAPH_CACHE[n] = g
    return g


def _goal_key(n: int, g: Goal) -> Tuple[int, int, int, int]:
    return (n, int(g.target), int(g.gx), int(g.gy))


def _compute_dist_to_success(n: int, g: Goal) -> np.ndarray:
    graph = _get_graph(n)
    states: List[State] = graph["states"]
    pred: List[List[int]] = graph["pred"]

    dist = np.full(len(states), -1, dtype=np.int32)
    q = deque()

    for i, s in enumerate(states):
        if is_success(s, g):
            dist[i] = 0
            q.append(i)

    while q:
        u = q.popleft()
        du = dist[u]
        for p in pred[u]:
            if dist[p] == -1:
                dist[p] = du + 1
                q.append(p)

    return dist


def get_goal_dist(n: int, g: Goal) -> np.ndarray:
    key = _goal_key(n, g)
    if key not in _GOAL_DIST_CACHE:
        _GOAL_DIST_CACHE[key] = _compute_dist_to_success(n, g)
    return _GOAL_DIST_CACHE[key]


def value_V(s: State, g: Goal) -> int:
    n = max(s.ax, s.ay, s.rx, s.ry, s.bx, s.by, g.gx, g.gy) + 1
    graph = _get_graph(n)
    dist = get_goal_dist(n, g)
    si = graph["idx_of"][s]
    d = int(dist[si])
    if d < 0:
        return -10**9
    return -d


def optimal_action_dist(s: State, g: Goal, n: int) -> np.ndarray:
    graph = _get_graph(n)
    dist = get_goal_dist(n, g)
    si = graph["idx_of"][s]
    d = int(dist[si])

    probs = np.zeros(len(ACTIONS), dtype=float)

    if d <= 0:
        probs[A2I["UP"]] = 1.0
        return probs

    good = []
    for ai in range(len(ACTIONS)):
        sj = int(graph["succ"][si, ai])
        dj = int(dist[sj])
        if dj == d - 1:
            good.append(ai)

    if not good:
        probs[A2I["UP"]] = 1.0
        return probs

    p = 1.0 / len(good)
    for ai in good:
        probs[ai] = p
    return probs


# ---------------------------
# Enumeration
# ---------------------------

def enumerate_states(n: int) -> List[State]:
    """Enumerate all states (holding in {NONE, RED, BLUE})."""
    states: List[State] = []
    cells = [(x, y) for x in range(n) for y in range(n)]

    for ax, ay in cells:
        # holding == NONE: both cubes on floor, distinct positions
        for rx, ry in cells:
            for bx, by in cells:
                if (rx, ry) == (bx, by):
                    continue
                states.append(State(ax, ay, rx, ry, bx, by, HOLD_NONE))

        # holding == RED: blue on floor
        for bx, by in cells:
            states.append(State(ax, ay, OFF_FLOOR, OFF_FLOOR, bx, by, HOLD_RED))

        # holding == BLUE: red on floor
        for rx, ry in cells:
            states.append(State(ax, ay, rx, ry, OFF_FLOOR, OFF_FLOOR, HOLD_BLUE))

    return states


def enumerate_goals(n: int) -> List[Goal]:
    goals: List[Goal] = []
    cells = [(x, y) for x in range(n) for y in range(n)]
    for target in (TARGET_RED, TARGET_BLUE):
        for gx, gy in cells:
            goals.append(Goal(target, gx, gy))
    return goals


def is_valid_pair(s: State, g: Goal) -> bool:
    """Check if (s, g) pair satisfies position constraints.

    - holding == NONE: red, blue, goal all distinct
    - holding != NONE: agent, floor_cube, goal all distinct
    """
    gp = g.goal_pos()

    if s.holding == HOLD_NONE:
        # red, blue, goal must all be distinct
        red = s.red_pos()
        blue = s.blue_pos()
        return len({red, blue, gp}) == 3

    else:
        # agent, floor_cube, goal must all be distinct
        agent = s.agent_pos()
        if s.holding == HOLD_RED:
            floor_cube = s.blue_pos()
        else:
            floor_cube = s.red_pos()
        return len({agent, floor_cube, gp}) == 3


# ---------------------------
# Representations
# ---------------------------

def phi_value_only(s: State, g: Goal) -> Tuple[float]:
    return (float(value_V(s, g)),)


def phi_distances(s: State, g: Goal) -> Tuple[int, int, int]:
    a = s.agent_pos()
    gp = g.goal_pos()
    tgt = target_pos_effective(s, g)
    return (s.holding, manhattan(tgt, gp), manhattan(a, tgt))


def phi_actor_like(s: State, g: Goal) -> Tuple[int, int, int, int, int, int]:
    a = s.agent_pos()
    gp = g.goal_pos()
    tgt = target_pos_effective(s, g)
    dx1, dy1 = (tgt[0] - a[0], tgt[1] - a[1])
    dx2, dy2 = (gp[0] - tgt[0], gp[1] - tgt[1])
    return (s.holding, g.target, dx1, dy1, dx2, dy2)


def phi_signs(s: State, g: Goal) -> Tuple[int, int, int, int, int, int]:
    holding, target, dx1, dy1, dx2, dy2 = phi_actor_like(s, g)
    return (holding, target, _sgn(dx1), _sgn(dy1), _sgn(dx2), _sgn(dy2))


def phi_dx_sign(s: State, g: Goal) -> Tuple[int, int, int]:
    _, target, _, _, dx2, _ = phi_actor_like(s, g)
    return (s.holding, target, _sgn(dx2))


PHI_REGISTRY = {
    "value_only": phi_value_only,
    "distances": phi_distances,
    "actor_like": phi_actor_like,
    "signs": phi_signs,
    "dx_sign": phi_dx_sign,
}


def quantize_z(z: Tuple[Any, ...], eps: Optional[float], round_decimals: Optional[int]) -> Tuple[Any, ...]:
    out = []
    for x in z:
        if isinstance(x, (float, np.floating)):
            xf = float(x)
            if eps is not None and eps > 0:
                xf = eps * round(xf / eps)
            if round_decimals is not None:
                xf = round(xf, round_decimals)
            out.append(xf)
        else:
            out.append(int(x) if isinstance(x, (np.integer,)) else x)
    return tuple(out)


# ---------------------------
# Entropy utilities
# ---------------------------

def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


# ---------------------------
# Core pipeline
# ---------------------------

def build_optimal_action_matrix(n: int) -> Dict[str, Any]:
    """Enumerate valid (s,g) pairs and compute optimal action dist and V."""
    states = enumerate_states(n)
    goals = enumerate_goals(n)

    rows = []
    action_mat = []
    values = []

    for si, s in enumerate(states):
        for gi, g in enumerate(goals):
            if not is_valid_pair(s, g):
                continue

            a_dist = optimal_action_dist(s, g, n)
            v = value_V(s, g)

            rows.append((si, gi))
            action_mat.append(a_dist)
            values.append(v)

    action_mat = np.stack(action_mat, axis=0)
    values = np.asarray(values, dtype=int)

    meta = {
        "n": n,
        "num_states": len(states),
        "num_goals_total": len(goals),
        "num_pairs": action_mat.shape[0],
    }

    states_arr = np.array([[s.ax, s.ay, s.rx, s.ry, s.bx, s.by, s.holding] for s in states], dtype=int)
    goals_arr = np.array([[g.target, g.gx, g.gy] for g in goals], dtype=int)
    rows_arr = np.array(rows, dtype=int)

    return {
        "meta": meta,
        "states": states_arr,
        "goals": goals_arr,
        "rows": rows_arr,
        "action": action_mat,
        "V": values,
    }


def compute_H_A_given_SG(action_mat: np.ndarray) -> float:
    ent = np.array([entropy_from_probs(action_mat[i]) for i in range(action_mat.shape[0])], dtype=float)
    return float(ent.mean())


def compute_mixed_action_and_H_A_given_SZ(
    data: Dict[str, Any],
    phi_name: str,
    eps: Optional[float],
    round_decimals: Optional[int],
) -> Tuple[float, np.ndarray, Dict[Tuple[int, Tuple[Any, ...]], List[int]]]:
    states_arr = data["states"]
    goals_arr = data["goals"]
    rows = data["rows"]
    action_mat = data["action"]

    phi = PHI_REGISTRY[phi_name]
    groups: Dict[Tuple[int, Tuple[Any, ...]], List[int]] = {}

    for idx in range(action_mat.shape[0]):
        si, gi = int(rows[idx, 0]), int(rows[idx, 1])
        s = State(*map(int, states_arr[si].tolist()))
        g = Goal(*map(int, goals_arr[gi].tolist()))
        z = phi(s, g)
        z_key = quantize_z(z, eps=eps, round_decimals=round_decimals)
        key = (si, z_key)
        groups.setdefault(key, []).append(idx)

    mixed_action = np.zeros_like(action_mat)
    ent_list = []
    weight_list = []

    for key, idxs in groups.items():
        avg = action_mat[idxs].mean(axis=0)
        mixed_action[idxs] = avg
        ent_list.append(entropy_from_probs(avg))
        weight_list.append(len(idxs))

    H = float(np.average(np.array(ent_list), weights=np.array(weight_list)))
    return H, mixed_action, groups


def compute_H_A_given_SV(data: Dict[str, Any]) -> Tuple[float, Dict[Tuple[int, int], List[int]]]:
    rows = data["rows"]
    action_mat = data["action"]
    V = data["V"]

    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(action_mat.shape[0]):
        si = int(rows[idx, 0])
        v = int(V[idx])
        groups.setdefault((si, v), []).append(idx)

    ent_list: List[float] = []
    weight_list: List[int] = []
    for idxs in groups.values():
        avg = action_mat[idxs].mean(axis=0)
        ent_list.append(entropy_from_probs(avg))
        weight_list.append(len(idxs))

    H = float(np.average(np.array(ent_list), weights=np.array(weight_list)))
    return H, groups


def compute_H_A_given_SVZ(
    data: Dict[str, Any],
    phi_name: str,
    eps: Optional[float],
    round_decimals: Optional[int],
) -> Tuple[float, Dict[Tuple[int, int, Tuple[Any, ...]], List[int]]]:
    states_arr = data["states"]
    goals_arr = data["goals"]
    rows = data["rows"]
    action_mat = data["action"]
    V = data["V"]

    phi = PHI_REGISTRY[phi_name]
    groups: Dict[Tuple[int, int, Tuple[Any, ...]], List[int]] = {}

    for idx in range(action_mat.shape[0]):
        si, gi = int(rows[idx, 0]), int(rows[idx, 1])
        s = State(*map(int, states_arr[si].tolist()))
        g = Goal(*map(int, goals_arr[gi].tolist()))
        z = phi(s, g)
        z_key = quantize_z(z, eps=eps, round_decimals=round_decimals)
        v = int(V[idx])
        groups.setdefault((si, v, z_key), []).append(idx)

    ent_list: List[float] = []
    weight_list: List[int] = []
    for idxs in groups.values():
        avg = action_mat[idxs].mean(axis=0)
        ent_list.append(entropy_from_probs(avg))
        weight_list.append(len(idxs))

    H = float(np.average(np.array(ent_list), weights=np.array(weight_list)))
    return H, groups


def compute_H_V_given_SZ(
    data: Dict[str, Any],
    groups: Dict[Tuple[int, Tuple[Any, ...]], List[int]],
) -> float:
    V = data["V"]
    ent_list = []
    weight_list = []

    for key, idxs in groups.items():
        vs = V[idxs]
        vals, counts = np.unique(vs, return_counts=True)
        p = counts.astype(float) / counts.sum()
        ent_list.append(entropy_from_probs(p))
        weight_list.append(len(idxs))

    return float(np.average(np.array(ent_list), weights=np.array(weight_list)))


# ---------------------------
# Save / Load
# ---------------------------

def save_npz(path: str, data: Dict[str, Any]) -> None:
    meta_json = json.dumps(data["meta"], sort_keys=True)
    np.savez_compressed(
        path,
        meta=np.array([meta_json]),
        states=data["states"],
        goals=data["goals"],
        rows=data["rows"],
        action=data["action"],
        V=data["V"],
    )


def load_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta_json = str(z["meta"][0])
    meta = json.loads(meta_json)
    return {
        "meta": meta,
        "states": z["states"],
        "goals": z["goals"],
        "rows": z["rows"],
        "action": z["action"],
        "V": z["V"],
    }


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--phi", choices=list(PHI_REGISTRY.keys()), default="value_only")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--round_decimals", type=int, default=None)
    ap.add_argument("--save_optimal", type=str, default=None)
    ap.add_argument("--load_optimal", type=str, default=None)

    args = ap.parse_args()

    if args.load_optimal is None:
        data = build_optimal_action_matrix(n=args.n)
        if args.save_optimal is not None:
            save_npz(args.save_optimal, data)
            print(f"Saved optimal matrix to: {args.save_optimal}")
    else:
        data = load_npz(args.load_optimal)

    action_mat = data["action"]

    H_A_SG = compute_H_A_given_SG(action_mat)
    H_A_SZ, mixed_action, groups = compute_mixed_action_and_H_A_given_SZ(
        data, phi_name=args.phi, eps=args.eps, round_decimals=args.round_decimals,
    )

    H_A_SV, groups_sv = compute_H_A_given_SV(data)
    H_A_SVZ, groups_svz = compute_H_A_given_SVZ(
        data, phi_name=args.phi, eps=args.eps, round_decimals=args.round_decimals,
    )

    I_A_Z_given_SV = H_A_SV - H_A_SVZ
    I_A_V_given_SZ = H_A_SZ - H_A_SVZ
    identity_residual = H_A_SZ - (H_A_SV - I_A_Z_given_SV + I_A_V_given_SZ)

    H_V_SZ = compute_H_V_given_SZ(data, groups)

    print("=== Two-cube Grid Toy: exact entropies ===")
    print("meta:", json.dumps(data["meta"], indent=2, sort_keys=True))
    print(f"phi: {args.phi} (eps={args.eps}, round_decimals={args.round_decimals})")
    print(f"H(A|S,G): {H_A_SG:.6f}")
    print(f"H(A|S,Z): {H_A_SZ:.6f}")
    print(f"Delta_A (action): H(A|S,Z)-H(A|S,G) = {H_A_SZ - H_A_SG:.6f}")
    print(f"H(V|S,Z): {H_V_SZ:.6f}")
    print(f"Delta_V (value): H(V|S,Z) = {H_V_SZ:.6f}")
    print(f"H(A|S,V): {H_A_SV:.6f}")
    print(f"H(A|S,V,Z): {H_A_SVZ:.6f}")
    print(f"I(A;Z|S,V): {I_A_Z_given_SV:.6f}")
    print(f"I(A;V|S,Z): {I_A_V_given_SZ:.6f}")
    print(f"identity residual (should be ~0): {identity_residual:.12f}")
    print(f"#(s,z) groups: {len(groups)}")


if __name__ == "__main__":
    main()