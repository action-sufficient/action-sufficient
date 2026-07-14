#!/usr/bin/env bash
# Launch one paper experiment. See the repository README for usage and examples.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/train.sh METHOD ENVIRONMENT [SEED] [-- EXTRA_MAIN_FLAGS...]

Methods:
  ota-value  ota-actor  ota-flow-value  ota-flow-actor  gciql  gcivl

Environments:
  cube-double-play   cube-triple-play   cube-quadruple-play
  cube-double-noisy  cube-triple-noisy  cube-quadruple-noisy
  scene-play  visual-cube  visual-scene

Environment variables:
  DATA_ROOT     Dataset root (default: <repository>/data)
  PYTHON        Python executable (default: python)
  WANDB_MODE    online, offline, or disabled (default: disabled)
  RUN_GROUP     W&B run group (default: paper)
  DRY_RUN       Set to 1 to print the command without running it

Examples:
  scripts/train.sh ota-value cube-double-play 0
  CUDA_VISIBLE_DEVICES=1 scripts/train.sh gcivl scene-play 3
  DRY_RUN=1 scripts/train.sh ota-flow-actor cube-quadruple-play 0
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if (( $# < 2 )); then
    usage >&2
    exit 2
fi

method=$1
environment=$2
shift 2
seed=0
if [[ ${1:-} =~ ^[0-9]+$ ]]; then
    seed=$1
    shift
fi
if [[ ${1:-} == "--" ]]; then
    shift
fi
extra_args=("$@")
visual=false
visual_args=()

case "$method" in
    ota-value|ota-actor|ota-flow-value|ota-flow-actor|gciql|gcivl) ;;
    *)
        echo "Unknown method: $method" >&2
        usage >&2
        exit 2
        ;;
esac

case "$environment" in
    cube-double-play)
        env_name=cube-double-play-v0
        dataset_name=cube-double-play-100m-v0
        task_name=cube-double-play-100m
        abstraction_factor=1
        ;;
    cube-triple-play)
        env_name=cube-triple-play-v0
        dataset_name=cube-triple-play-100m-v0
        task_name=cube-triple-play-100m
        abstraction_factor=1
        ;;
    cube-quadruple-play)
        env_name=cube-quadruple-play-v0
        dataset_name=cube-quadruple-play-100m-v0
        task_name=cube-quadruple-play-100m
        abstraction_factor=5
        ;;
    cube-double-noisy)
        env_name=cube-double-noisy-v0
        dataset_name=cube-double-noisy-100m-v0
        task_name=cube-double-noisy-100m
        abstraction_factor=1
        ;;
    cube-triple-noisy)
        env_name=cube-triple-noisy-v0
        dataset_name=cube-triple-noisy-100m-v0
        task_name=cube-triple-noisy-100m
        abstraction_factor=1
        ;;
    cube-quadruple-noisy)
        env_name=cube-quadruple-noisy-v0
        dataset_name=cube-quadruple-noisy-100m-v0
        task_name=cube-quadruple-noisy-100m
        abstraction_factor=5
        ;;
    scene-play)
        env_name=scene-play-v0
        dataset_name=scene-play-100m-v0
        task_name=scene-play-100m
        abstraction_factor=1
        ;;
    visual-cube)
        env_name=visual-cube-double-play-v0
        dataset_name=visual-cube-double-play-1m-v0
        task_name=visual-cube-double-play-1m
        abstraction_factor=1
        visual=true
        visual_args=(--agent.encoder=impala_small --agent.p_aug=0.5 --agent.batch_size=256)
        ;;
    visual-scene)
        env_name=visual-scene-play-v0
        dataset_name=visual-scene-play-1m-v0
        task_name=visual-scene-play-1m
        abstraction_factor=1
        visual=true
        visual_args=(--agent.encoder=impala_small --agent.p_aug=0.5 --agent.batch_size=256)
        ;;
    *)
        echo "Unknown environment: $environment" >&2
        usage >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
data_root=${DATA_ROOT:-"$repo_root/data"}
python=${PYTHON:-python}
wandb_mode=${WANDB_MODE:-disabled}
run_group=${RUN_GROUP:-paper}

common_args=(
    --seed="$seed"
    --save_interval=500000
    --wandb_mode="$wandb_mode"
    --run_group="$run_group"
    --evaluate_low_level=False
    --env_name="$env_name"
    --dataset_dir="$data_root/$dataset_name"
    --dataset_replace_interval=1000
)

case "$method" in
    ota-value|ota-actor|ota-flow-value|ota-flow-actor)
        if [[ $method == ota-flow-* ]]; then
            agent_config=agents/ota_flow.py
            method_tag=ota_flow
        else
            agent_config=agents/ota.py
            method_tag=ota
        fi

        if [[ $method == *-value ]]; then
            representation=value_rep
            representation_args=(
                --agent.use_value_rep=True
                --agent.use_actor_rep=False
                --agent.general_goal=False
            )
        else
            representation=actor_rep
            representation_args=(
                --agent.use_value_rep=False
                --agent.use_actor_rep=True
                --agent.general_goal=True
            )
        fi

        exp_name="${task_name}-${method_tag}_3.0_3.0-expectile_0.7-discount_0.995-n_${abstraction_factor}-k_25-${representation}"
        method_args=(
            --agent="$agent_config"
            --agent.discount=0.995
            --agent.expectile=0.7
            --agent.high_alpha=3.0
            --agent.low_alpha=3.0
            --agent.abstraction_factor="$abstraction_factor"
            --agent.value_geom_sample=True
            --agent.subgoal_steps=25
            --agent.actor_p_trajgoal=1.0
            --agent.actor_p_randomgoal=0.0
            --agent.actor_geom_sample=False
            "${representation_args[@]}"
        )
        ;;
    gciql)
        exp_name="${task_name}-gciql_1.0-expectile_0.9-discount_0.995-n_${abstraction_factor}-no_rep"
        method_args=(
            --agent=agents/gciql.py
            --agent.discount=0.995
            --agent.expectile=0.9
            --agent.actor_loss=ddpgbc
            --agent.alpha=1.0
            --agent.value_subgoal_steps="$abstraction_factor"
            --agent.value_geom_sample=True
            --agent.actor_p_trajgoal=1.0
            --agent.actor_p_randomgoal=0.0
            --agent.actor_geom_sample=False
            --agent.goal_rep=False
        )
        ;;
    gcivl)
        exp_name="${task_name}-gcivl_10.0-expectile_0.9-discount_0.995"
        method_args=(
            --agent=agents/gcivl.py
            --agent.discount=0.995
            --agent.expectile=0.9
            --agent.alpha=10.0
            --agent.value_geom_sample=True
            --agent.actor_p_trajgoal=1.0
            --agent.actor_p_randomgoal=0.0
            --agent.actor_geom_sample=False
        )
        ;;
esac

if [[ $visual == true ]]; then
    method_args+=("${visual_args[@]}")
elif [[ $method == gcivl ]]; then
    method_args+=(--agent.encoder=None)
fi

command=(
    "$python"
    "$repo_root/main.py"
    "${common_args[@]}"
    --exp_name="$exp_name"
    "${method_args[@]}"
    "${extra_args[@]}"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    cd "$repo_root"
    exec "${command[@]}"
fi
