#!/usr/bin/env bash
# Run the complete 6-method x 9-environment paper matrix sequentially.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

methods=(ota-value ota-actor ota-flow-value ota-flow-actor gciql gcivl)
environments=(
    cube-double-play
    cube-triple-play
    cube-quadruple-play
    cube-double-noisy
    cube-triple-noisy
    cube-quadruple-noisy
    scene-play
    visual-cube
    visual-scene
)

# Use a whitespace-separated list, for example: SEEDS="0 1 2 3 4 5 6 7".
read -r -a seeds <<< "${SEEDS:-0}"

for seed in "${seeds[@]}"; do
    for environment in "${environments[@]}"; do
        for method in "${methods[@]}"; do
            echo "==> method=$method environment=$environment seed=$seed"
            "$script_dir/train.sh" "$method" "$environment" "$seed" "$@"
        done
    done
done
