<div align="center">

# Action-Sufficient Goal Representations

Jinu Hyeon<sup>1,\*</sup> · Woobin Park<sup>1,\*</sup> · Hongjoon Ahn<sup>1,2,\*</sup> · Taesup Moon<sup>1</sup>

<sup>1</sup>Seoul National University &nbsp;&nbsp; <sup>2</sup>Trillion Labs<br>
<sup>\*</sup>Equal contribution

**ICML 2026**

[**Paper**](https://arxiv.org/abs/2601.22496) &nbsp;·&nbsp; [**Project Page**](https://action-sufficient.github.io/)

</div>

Official JAX implementation and experiment code for **Action-Sufficient Goal Representations**.

## Overview

Hierarchical offline goal-conditioned reinforcement learning uses a compressed goal representation as the interface between high-level subgoal planning and low-level control. Representations learned for value prediction can collapse goals that have the same value but require different actions. This work formalizes **action sufficiency** and learns goal representations directly through the actor objective.

This repository compares actor- and value-derived representations with both standard and flow-matching policies, alongside GCIQL and GCIVL baselines.

## Environment setup

The original experiments used Python 3.10 and JAX with CUDA 12. The following commands create a local virtual environment using [`uv`](https://docs.astral.sh/uv/):

```bash
# Install the OpenGL system dependencies on Ubuntu or Debian.
sudo apt-get install libgl1-mesa-glx libegl1 libopengl0

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt

python -c "import jax; print(jax.devices())"
```

Activate the environment before running an experiment; alternatively, pass its interpreter to the launcher explicitly:

```bash
PYTHON=.venv/bin/python scripts/train.sh ota-actor cube-double-play
```

## Experiments

The public launcher covers six methods:

| Selector | Description |
| --- | --- |
| `ota-value` | OTA with a value-derived goal representation |
| `ota-actor` | OTA with an actor-derived goal representation |
| `ota-flow-value` | Flow-matching OTA with a value-derived representation |
| `ota-flow-actor` | Flow-matching OTA with an actor-derived representation |
| `gciql` | Goal-conditioned implicit Q-learning baseline |
| `gcivl` | Goal-conditioned implicit V-learning baseline |

The following environments are supported:

| Category | Selectors | Dataset size |
| --- | --- | --- |
| Cube play | `cube-double-play`, `cube-triple-play`, `cube-quadruple-play` | 100M |
| Noisy cube | `cube-double-noisy`, `cube-triple-noisy`, `cube-quadruple-noisy` | 100M |
| Scene | `scene-play` | 100M |
| Visual | `visual-cube`, `visual-scene` | 1M |

The visual experiments use an `impala_small` encoder, image augmentation, and a batch size of 256, matching the paper experiments.

### Discrete Cube sub-experiment

The [`discrete-cube`](discrete-cube/) directory contains the companion code for the two-cube discrete GCRL experiment. It solves the finite MDP exactly and evaluates how action- and value-sufficiency gaps relate to control success across a large family of goal representations. See the [Discrete Cube experiment guide](discrete-cube/EXPERIMENT.md) for setup, execution, and output details.

## Data

By default, datasets are read from the `data/` directory in the repository root.

For datasets that are available for download, follow the [Horizon Reduction large-dataset instructions](https://github.com/seohongpark/horizon-reduction#using-large-datasets) and the [OGBench dataset list](https://github.com/seohongpark/ogbench#additional-features). If a required dataset is not available, create it by following the [OGBench dataset reproduction instructions](https://github.com/seohongpark/ogbench#reproducing-datasets). Place downloaded or generated files in the corresponding directories below.

```text
action-sufficient-rep/
└── data/
    ├── cube-double-play-100m-v0/
    │   ├── cube-double-play-v0-000.npz
    │   ├── cube-double-play-v0-000-val.npz
    │   ├── ...
    │   ├── cube-double-play-v0-099.npz
    │   └── cube-double-play-v0-099-val.npz
    ├── cube-triple-play-100m-v0/
    ├── cube-quadruple-play-100m-v0/
    ├── cube-double-noisy-100m-v0/
    ├── cube-triple-noisy-100m-v0/
    ├── cube-quadruple-noisy-100m-v0/
    ├── scene-play-100m-v0/
    ├── visual-cube-double-play-1m-v0/
    └── visual-scene-play-1m-v0/
```

Each dataset directory should contain one or more training `.npz` files and their corresponding `-val.npz` files. When multiple training shards are present, the launcher rotates through them during training.

Set `DATA_ROOT` to use another location:

```bash
DATA_ROOT=/path/to/data scripts/train.sh gciql cube-triple-play 0
```

## Running experiments

Run one method-environment combination with `scripts/train.sh`. The seed is optional and defaults to `0`:

```text
scripts/train.sh METHOD ENVIRONMENT [SEED] [-- EXTRA_MAIN_FLAGS...]
```

```bash
scripts/train.sh ota-actor cube-quadruple-play
scripts/train.sh ota-flow-actor cube-triple-noisy 3
scripts/train.sh gcivl visual-scene 0
```

Select a GPU in the usual way:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train.sh ota-value scene-play
```

Additional `main.py` flags can be appended after `--`:

```bash
scripts/train.sh gciql cube-double-play -- \
    --wandb_mode=online \
    --offline_steps=100000
```

By default, W&B logging is disabled. Runs are saved beneath `exp/action-sufficient-rep/<run-group>/`, with the run group defaulting to `paper`. Set `WANDB_MODE` and `RUN_GROUP` to change these defaults.

Run the complete 54-experiment matrix sequentially for seed `0` with:

```bash
scripts/run_all.sh
```

To inspect commands without starting training:

```bash
DRY_RUN=1 scripts/run_all.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hyeon2026action,
  title     = {Action-Sufficient Goal Representations},
  author    = {Hyeon, Jinu and Park, Woobin and Ahn, Hongjoon and Moon, Taesup},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
