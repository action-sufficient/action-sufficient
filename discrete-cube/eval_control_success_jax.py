#!/usr/bin/env python3
"""
Evaluate control success using Bayes-optimal mixed policy P(A|S,Z).

JAX-ACCELERATED VERSION:
- Fully vectorized rollouts using JAX vmap
- GPU/TPU support (auto-detected)
- ~100x speedup on GPU

Usage:
  python eval_control_success_jax.py --phi_csv phi_sweep.csv --num_eval_random 2000
"""

from __future__ import annotations
import argparse, importlib, os, re, sys
from typing import Dict, List
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw): return x

BASELINES = ["value_only", "distances", "actor_like", "signs", "dx_sign"]

print(f"[jax] Devices: {jax.devices()}")
print(f"[jax] Backend: {jax.default_backend()}")


# =============================================================================
# JAX-compiled rollout
# =============================================================================

@jit
def step_jax(state, action, n):
    """State transition."""
    ax, ay, rx, ry, bx, by, h = state[0], state[1], state[2], state[3], state[4], state[5], state[6]

    # Movement
    new_ax = jnp.where(action == 2, jnp.maximum(ax - 1, 0), ax)  # LEFT
    new_ax = jnp.where(action == 3, jnp.minimum(ax + 1, n - 1), new_ax)  # RIGHT
    new_ay = jnp.where(action == 0, jnp.minimum(ay + 1, n - 1), ay)  # UP
    new_ay = jnp.where(action == 1, jnp.maximum(ay - 1, 0), new_ay)  # DOWN

    # PICK
    can_pick_red = (h == 0) & (rx != -1) & (ax == rx) & (ay == ry)
    can_pick_blue = (h == 0) & (bx != -1) & (ax == bx) & (ay == by)

    new_rx = jnp.where((action == 4) & can_pick_red, -1, rx)
    new_ry = jnp.where((action == 4) & can_pick_red, -1, ry)
    new_bx = jnp.where((action == 4) & can_pick_blue, -1, bx)
    new_by = jnp.where((action == 4) & can_pick_blue, -1, by)
    new_h = jnp.where((action == 4) & can_pick_red, 1, h)
    new_h = jnp.where((action == 4) & can_pick_blue, 2, new_h)

    # PLACE
    can_place_red = (h == 1) & ~((bx != -1) & (bx == new_ax) & (by == new_ay))
    can_place_blue = (h == 2) & ~((rx != -1) & (rx == new_ax) & (ry == new_ay))

    new_rx = jnp.where((action == 5) & can_place_red, new_ax, new_rx)
    new_ry = jnp.where((action == 5) & can_place_red, new_ay, new_ry)
    new_bx = jnp.where((action == 5) & can_place_blue, new_ax, new_bx)
    new_by = jnp.where((action == 5) & can_place_blue, new_ay, new_by)
    new_h = jnp.where((action == 5) & can_place_red, 0, new_h)
    new_h = jnp.where((action == 5) & can_place_blue, 0, new_h)

    return jnp.array([new_ax, new_ay, new_rx, new_ry, new_bx, new_by, new_h], dtype=jnp.int32)


def make_rollout_fn(n, max_horizon):
    """Create JIT-compiled batch rollout function."""

    @partial(jit, static_argnums=(2, 3))
    def rollout_batch(states0, goals, n_static, max_h_static, fullA, rand_vals, state_lookup, goal_indices):
        """
        Vectorized rollouts.
        states0: (num_tasks, 7)
        goals: (num_tasks, 3)
        fullA: (nS, nG, 6)
        rand_vals: (num_tasks, num_rollouts, max_horizon)
        state_lookup: (n, n, n+1, n+1, n+1, n+1, 3)
        goal_indices: (num_tasks,) - actual goal indices for policy lookup
        """
        nG = fullA.shape[1]

        def single_rollout(state0, goal, gi_actual, rands):
            """Single rollout."""
            target, gx, gy = goal[0], goal[1], goal[2]

            def check_succ(st):
                rx, ry, bx, by, h = st[2], st[3], st[4], st[5], st[6]
                red_ok = (target == 1) & (rx == gx) & (ry == gy) & (h != 1)
                blue_ok = (target == 2) & (bx == gx) & (by == gy) & (h != 2)
                return red_ok | blue_ok

            def step_fn(carry, inputs):
                state, done = carry
                t, rand_val = inputs

                succ = check_succ(state)
                done = done | succ

                # Get state index
                ax, ay = state[0], state[1]
                rx, ry = state[2], state[3]
                bx, by = state[4], state[5]
                h = state[6]

                si = state_lookup[ax, ay, rx + 1, ry + 1, bx + 1, by + 1, h]

                # Use the actual goal index passed in
                probs = fullA[si, gi_actual]
                cumsum = jnp.cumsum(probs)
                action = jnp.sum(cumsum <= rand_val).astype(jnp.int32)
                action = jnp.clip(action, 0, 5)

                new_state = jnp.where(done, state, step_jax(state, action, n_static))

                return (new_state, done), succ

            init = (state0, jnp.array(False))
            inputs = (jnp.arange(max_h_static), rands)
            (final_state, _), succs = lax.scan(step_fn, init, inputs)

            return (check_succ(final_state) | jnp.any(succs)).astype(jnp.float32)

        # vmap over rollouts then tasks
        over_rollouts = vmap(single_rollout, in_axes=(None, None, None, 0))
        over_tasks = vmap(over_rollouts, in_axes=(0, 0, 0, 0))

        results = over_tasks(states0, goals, goal_indices, rand_vals)
        return results.mean()

    return rollout_batch


# =============================================================================
# Utility functions
# =============================================================================

def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return 0.0 if not p.size else float(-(p * np.log(p)).sum())

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    raise KeyError(cands)

def parse_int(s):
    m = re.search(r"_(\d+)$", str(s))
    return int(m.group(1)) if m else None

def fps(xs, ys, ids, k, seeds):
    if k <= 0: return list(seeds)
    X = np.stack([xs, ys], 1).astype(float)
    mu, sig = X.mean(0), X.std(0); sig[sig==0] = 1; Xn = (X-mu)/sig
    id2i = {p: i for i, p in enumerate(ids)}
    ch = list(dict.fromkeys([id2i[s] for s in seeds if s in id2i]))
    if not ch: ch = [int(np.argmax((Xn**2).sum(1)))]
    rem = np.ones(len(ids), bool)
    for c in ch: rem[c] = False
    md = np.full(len(ids), np.inf)
    for c in ch: md = np.minimum(md, np.linalg.norm(Xn - Xn[c], 1))
    while len(ch) < min(len(ids), len(seeds) + k):
        cand = np.where(rem)[0]
        if not cand.size: break
        b = cand[np.argmax(md[cand])]; ch.append(int(b)); rem[b] = False
        md = np.minimum(md, np.linalg.norm(Xn - Xn[b], 1))
    out = [s for s in seeds if s in [ids[i] for i in ch]]
    for i in ch:
        if ids[i] not in out: out.append(ids[i])
        if len(out) >= len(seeds) + k: break
    return out

def compute_z_vec(pid, feats):
    h, t = feats["holding"], feats["target"]
    dx1, dy1, dx2, dy2 = feats["dx1"], feats["dy1"], feats["dx2"], feats["dy2"]
    d_at, d_bg, V = feats["d_at"], feats["d_bg"], feats["V"]

    if pid == "value_only": return V.astype(np.int64)
    if pid == "distances": return (h*10000 + d_bg*100 + d_at).astype(np.int64)
    if pid == "actor_like":
        return (h.astype(np.int64)*10**10 + t.astype(np.int64)*10**8 +
                (dx1+10).astype(np.int64)*10**6 + (dy1+10).astype(np.int64)*10**4 +
                (dx2+10).astype(np.int64)*10**2 + (dy2+10).astype(np.int64))
    if pid == "signs":
        return (h*1000 + t*100 + (np.sign(dx1)+1)*27 + (np.sign(dy1)+1)*9 +
                (np.sign(dx2)+1)*3 + (np.sign(dy2)+1)).astype(np.int64)
    if pid == "dx_sign": return (h*100 + t*10 + np.sign(dx2)+1).astype(np.int64)
    return None

def build_state_lookup(states, n):
    base = n + 1
    lookup = np.full((n, n, base, base, base, base, 3), 0, dtype=np.int32)
    for si, s in enumerate(states):
        ax, ay, rx, ry, bx, by, h = s
        lookup[ax, ay, rx+1, ry+1, bx+1, by+1, h] = si
    return lookup


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_csv", required=True)
    ap.add_argument("--base_module", default="grid_two_cube_bayes_entropy_pipeline")
    ap.add_argument("--sweep_module", default="sweep_random_phi_family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_eval_random", type=int, default=40)
    ap.add_argument("--num_tasks", type=int, default=600)
    ap.add_argument("--task_seed", type=int, default=0)
    ap.add_argument("--num_rollouts", type=int, default=50)
    ap.add_argument("--rollout_seed", type=int, default=0)
    ap.add_argument("--horizon_margin", type=int, default=5)
    ap.add_argument("--horizon_cap", type=int, default=30)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out_csv", default="control_eval_results.csv")
    ap.add_argument("--out_png", default="control_eval_scatter.png")
    args = ap.parse_args()

    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path: sys.path.insert(0, d)
    base = importlib.import_module(args.base_module)
    sweep = importlib.import_module(args.sweep_module)

    # Load CSV
    df = pd.read_csv(args.phi_csv)
    id_col = pick_col(df, ["phi_id","name","phi"])
    dA_col = pick_col(df, ["Delta_A","delta_A"])
    dV_col = pick_col(df, ["Delta_V","delta_V"])

    df_r = df[~df[id_col].astype(str).isin(BASELINES)].drop_duplicates(id_col)
    chosen = fps(df_r[dV_col].values, df_r[dA_col].values,
                 df_r[id_col].astype(str).tolist(), args.num_eval_random, BASELINES)
    for b in BASELINES:
        if b not in chosen: chosen.insert(0, b)
    chosen = list(dict.fromkeys(chosen))
    print(f"[sel] {len(chosen)} phis")

    maxk = max((parse_int(p) or 0) for p in df_r[id_col].astype(str))
    num_rand = maxk + 1 if maxk else len(df_r)

    # Build env
    print("[env] Building...")
    slist = base.enumerate_states(args.n)
    glist = base.enumerate_goals(args.n)
    states = np.array([[s.ax,s.ay,s.rx,s.ry,s.bx,s.by,s.holding] for s in slist], np.int32)
    goals = np.array([[g.target,g.gx,g.gy] for g in glist], np.int32)
    nS, nG = len(states), len(goals)

    fullA = np.zeros((nS, nG, 6), np.float32)
    fullV = np.zeros((nS, nG), np.int32)
    for si, s in enumerate(slist):
        for gi, g in enumerate(glist):
            fullA[si,gi] = base.optimal_action_dist(s, g, args.n)
            fullV[si,gi] = base.value_V(s, g)
    print(f"[env] S={nS}, G={nG}")

    state_lookup = build_state_lookup(states, args.n)
    state_lookup_jax = jnp.array(state_lookup)
    fullA_jax = jnp.array(fullA)

    # Features for ALL pairs (for policy lookup during rollout)
    nPairs = nS * nG
    flat_si = np.repeat(np.arange(nS, dtype=np.int32), nG)
    flat_gi = np.tile(np.arange(nG, dtype=np.int32), nS)
    ax_a, ay_a = states[flat_si,0], states[flat_si,1]
    rx_a, ry_a = states[flat_si,2], states[flat_si,3]
    bx_a, by_a = states[flat_si,4], states[flat_si,5]
    h_a = states[flat_si,6]
    t_a, gx_a, gy_a = goals[flat_gi,0], goals[flat_gi,1], goals[flat_gi,2]

    is_red = t_a == 1
    txf, tyf = np.where(is_red, rx_a, bx_a), np.where(is_red, ry_a, by_a)
    ht = ((h_a==1)&is_red)|((h_a==2)&~is_red)
    tx, ty = np.where(ht, ax_a, txf), np.where(ht, ay_a, tyf)

    feats_all = {
        "si": flat_si, "holding": h_a.astype(np.int32), "target": t_a.astype(np.int32),
        "dx1": (tx-ax_a).astype(np.int32), "dy1": (ty-ay_a).astype(np.int32),
        "dx2": (gx_a-tx).astype(np.int32), "dy2": (gy_a-ty).astype(np.int32),
        "d_at": (np.abs(tx-ax_a)+np.abs(ty-ay_a)).astype(np.int32),
        "d_bg": (np.abs(gx_a-tx)+np.abs(gy_a-ty)).astype(np.int32),
        "V": fullV.flatten().astype(np.int32),
    }

    # Filtered data (for entropy computation)
    data_f = base.build_optimal_action_matrix(args.n)
    rows_f, actF, VF = data_f["rows"], data_f["action"].astype(np.float32), data_f["V"].astype(np.int32)
    nPF = len(rows_f)
    siF, giF = rows_f[:,0].astype(np.int32), rows_f[:,1].astype(np.int32)
    flatF = siF * nG + giF
    H_A_SG = np.mean([entropy(actF[i]) for i in range(nPF)])

    # Build feats for FILTERED pairs (MUST match sweep's feats!)
    ax_f, ay_f = states[siF,0], states[siF,1]
    rx_f, ry_f = states[siF,2], states[siF,3]
    bx_f, by_f = states[siF,4], states[siF,5]
    h_f = states[siF,6]
    t_f, gx_f, gy_f = goals[giF,0], goals[giF,1], goals[giF,2]

    is_red_f = t_f == 1
    txf_f, tyf_f = np.where(is_red_f, rx_f, bx_f), np.where(is_red_f, ry_f, by_f)
    ht_f = ((h_f==1)&is_red_f)|((h_f==2)&~is_red_f)
    tx_f, ty_f = np.where(ht_f, ax_f, txf_f), np.where(ht_f, ay_f, tyf_f)

    feats_filtered = {
        "si": siF, "holding": h_f.astype(np.int32), "target": t_f.astype(np.int32),
        "dx1": (tx_f-ax_f).astype(np.int32), "dy1": (ty_f-ay_f).astype(np.int32),
        "dx2": (gx_f-tx_f).astype(np.int32), "dy2": (gy_f-ty_f).astype(np.int32),
        "d_at": (np.abs(tx_f-ax_f)+np.abs(ty_f-ay_f)).astype(np.int32),
        "d_bg": (np.abs(gx_f-tx_f)+np.abs(gy_f-ty_f)).astype(np.int32),
        "V": VF.astype(np.int32),
    }

    # Phi family - use FILTERED feats to match sweep!
    rng = np.random.default_rng(args.seed)
    phis = sweep.build_phi_family(rng, feats_filtered, num_rand)
    phi_map = {p.phi_id: p for p in phis}
    miss = [p for p in chosen if p not in phi_map]
    if miss: print(f"[warn] Dropping {len(miss)}"); chosen = [p for p in chosen if p in phi_map]

    # Tasks
    rng_t = np.random.default_rng(args.task_seed)
    hnone = np.where(states[:,6]==0)[0]
    tasks = []
    att = 0
    while len(tasks) < args.num_tasks and att < args.num_tasks*100:
        att += 1
        si0 = int(rng_t.choice(hnone))
        gi0 = int(rng_t.integers(0,nG))
        s0, g0 = tuple(states[si0]), tuple(goals[gi0])
        t,gx,gy = g0; ax,ay,rx,ry,bx,by,h = s0
        if (t==1 and rx==gx and ry==gy and h!=1) or (t==2 and bx==gx and by==gy and h!=2): continue
        v0 = int(fullV[si0,gi0])
        if v0==0 or v0<=-10000: continue
        tasks.append((si0,gi0,min(-v0+args.horizon_margin,args.horizon_cap)))
    print(f"[tasks] {len(tasks)}")

    tasks_states = np.array([states[t[0]] for t in tasks], np.int32)
    tasks_goals = np.array([goals[t[1]] for t in tasks], np.int32)
    tasks_gi = np.array([t[1] for t in tasks], np.int32)  # actual goal indices
    max_horizon = max(t[2] for t in tasks)

    tasks_states_jax = jnp.array(tasks_states)
    tasks_goals_jax = jnp.array(tasks_goals)
    tasks_gi_jax = jnp.array(tasks_gi)

    # Create and compile rollout function
    print("[jax] Compiling...")
    rollout_fn = make_rollout_fn(args.n, max_horizon)

    # Warmup - include goal_indices
    dummy_rand = jnp.array(np.random.random((2, args.num_rollouts, max_horizon)).astype(np.float32))
    dummy_gi = jnp.array([0, 1], dtype=jnp.int32)
    _ = rollout_fn(tasks_states_jax[:2], tasks_goals_jax[:2], args.n, max_horizon,
                   fullA_jax, dummy_rand, state_lookup_jax, dummy_gi)
    print("[jax] Ready")

    # Evaluate
    results = []
    for pid in tqdm(chosen, desc="Evaluating"):
        ps = phi_map[pid]

        # Z (use feats_filtered, indices into filtered pairs)
        Z_flat = compute_z_vec(pid, feats_filtered)
        if Z_flat is None:
            Z_list = [ps.fn(i) for i in range(nPF)]  # nPF, not nPairs!
            Z_flat = np.array([hash(z) if isinstance(z,tuple) else int(z) for z in Z_list], np.int64)

        # Entropy (Z_flat is now indexed by filtered pair index i, not flatF[i])
        grp = {}
        for i in range(nPF):
            z = int(Z_flat[i])  # Z_flat is length nPF, indexed directly
            k = (int(siF[i]), z)
            if k not in grp: grp[k] = {"sum": actF[i].astype(np.float64).copy(), "cnt":1, "vs":[int(VF[i])]}
            else: grp[k]["sum"] += actF[i]; grp[k]["cnt"] += 1; grp[k]["vs"].append(int(VF[i]))

        H_A, H_V = 0.0, 0.0
        for v in grp.values():
            avg = v["sum"]/v["cnt"]
            p = avg[avg>0]
            H_A += v["cnt"]*(-(p*np.log(p)).sum() if p.size else 0)
            vs = np.array(v["vs"])
            _, cts = np.unique(vs, return_counts=True)
            pv = cts/v["cnt"]; pv = pv[pv>0]
            H_V += v["cnt"]*(-(pv*np.log(pv)).sum() if pv.size else 0)

        dA, dV = H_A/nPF - H_A_SG, H_V/nPF

        # Rollout - need to use mixed policy pi_sz
        # Build (si, gi) -> Z mapping from filtered pairs
        # For non-filtered pairs, we'll fallback to optimal policy anyway

        # Create Z lookup from filtered pairs: flatF[i] -> Z_flat[i]
        Z_lookup = {}  # (si, gi) -> z
        for i in range(nPF):
            si_i, gi_i = int(siF[i]), int(giF[i])
            Z_lookup[(si_i, gi_i)] = int(Z_flat[i])

        # Create policy array
        policy_lookup = np.zeros_like(fullA)
        for si in range(nS):
            for gi in range(nG):
                key = (si, gi)
                if key in Z_lookup:
                    z = Z_lookup[key]
                    k = (si, z)
                    if k in grp:
                        policy_lookup[si, gi] = grp[k]["sum"] / grp[k]["cnt"]
                    else:
                        policy_lookup[si, gi] = fullA[si, gi]
                else:
                    # Non-filtered pair: use optimal policy
                    policy_lookup[si, gi] = fullA[si, gi]

        policy_lookup_jax = jnp.array(policy_lookup.astype(np.float32))

        pseed = (args.rollout_seed + abs(hash(pid))) % (2**31)
        rand_vals = jnp.array(np.random.default_rng(pseed).random(
            (len(tasks), args.num_rollouts, max_horizon)).astype(np.float32))

        sr = float(rollout_fn(tasks_states_jax, tasks_goals_jax, args.n, max_horizon,
                              policy_lookup_jax, rand_vals, state_lookup_jax, tasks_gi_jax))

        results.append({"phi_id": pid, "Delta_A": dA, "Delta_V": dV, "success_rate": sr})

    # Save
    for r in results:
        rc = df[df[id_col].astype(str)==r["phi_id"]]
        r["Delta_A_csv"] = float(rc[dA_col].iloc[0]) if len(rc) else np.nan
        r["Delta_V_csv"] = float(rc[dV_col].iloc[0]) if len(rc) else np.nan

    res = pd.DataFrame(results)
    res.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}")

    def corr(a,b):
        a,b = np.asarray(a,float), np.asarray(b,float)
        return np.nan if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
    print(f"[corr] sr~dA: {corr(res.success_rate, res.Delta_A):.4f}")
    print(f"[corr] sr~dV: {corr(res.success_rate, res.Delta_V):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9,7))
    x,y,c = res.Delta_V.values, res.Delta_A.values, res.success_rate.values
    lo,hi = np.percentile(c,[2,98])
    if lo==hi: lo,hi = c.min()-1e-9, c.max()+1e-9
    sc = ax.scatter(x,y,c=c,s=90,vmin=lo,vmax=hi,alpha=.85,cmap="viridis")
    plt.colorbar(sc, ax=ax, label="success_rate")
    ax.set_xlabel("Delta_V"); ax.set_ylabel("Delta_A")
    ax.set_title(f"Control success (n={args.n}, JAX)")
    ax.axhline(0,c='gray',ls='--',lw=.5); ax.axvline(0,c='gray',ls='--',lw=.5)
    for b in BASELINES:
        r = res[res.phi_id==b]
        if len(r):
            ax.scatter(float(r.Delta_V.iloc[0]), float(r.Delta_A.iloc[0]), marker="x", s=160, c="red")
            ax.annotate(f" {b}", (float(r.Delta_V.iloc[0]), float(r.Delta_A.iloc[0])), fontsize=9)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=200)
    print(f"[saved] {args.out_png}")


if __name__ == "__main__":
    main()