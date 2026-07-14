# Discrete Cube: a two-cube GCRL toy

Companion code for the Discrete Cube experiment.

A two-cube grid manipulation MDP is solved **exactly** — breadth-first search over
the full state graph, no learning — and a large family of goal representations
`Z = phi(S, G)` is then swept to measure two information gaps:

- `Delta_A = I(A;G | S,Z) = H(A|S,Z) - H(A|S,G)` — action insufficiency
- `Delta_V = I(V;G | S,Z) = H(V|S,Z)` — value insufficiency (`H(V|S,G) = 0`)

Each representation's Bayes-optimal mixed policy `pi(A|S,Z)` is then rolled out to
get a control success rate, which is what the two gaps are tested against.

## Setup

Use the shared environment from the [main environment setup](../README.md#environment-setup). From the repository root:

```bash
source .venv/bin/activate
cd discrete-cube
```

## Running

```bash
./run.sh          # everything: sweep -> eval -> figure
```

That is the full research configuration (`n=4`, 2000 random phis, 600 tasks) and
it is the one the paper reports. It writes to `results/n4_2000/`:

| file | produced by | contents |
|---|---|---|
| `phi_sweep.csv` | sweep | one row per phi: `Delta_A`, `Delta_V`, `H_A_SZ`, `I_A_Z_given_SV`, quadrant, ... |
| `phi_sweep.png` | sweep | quick `Delta_V` vs `Delta_A` scatter |
| `control_eval.csv` | eval | the evaluated subset, plus `success_rate` |
| `control_eval.png` | eval | quick success-rate scatter |
| `toy_final.png` / `.pdf` | plot | the publication figure |

To change the configuration, edit the variables at the top of `run.sh` (`N`,
`NUM_RANDOM`, `NUM_TASKS`, `NUM_ROLLOUTS`, `HORIZON_MARGIN`, `HORIZON_CAP`,
`SEED`). The output directory follows `N` and `NUM_RANDOM` automatically, so
different configurations do not overwrite each other.
