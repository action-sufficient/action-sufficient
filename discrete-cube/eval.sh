#!/bin/bash
# Re-run eval + plot only, reusing the sweep already in results/n${N}_${NUM_RANDOM}/

# Settings (must match run.sh)
N=4
NUM_RANDOM=2000
NUM_TASKS=600
NUM_ROLLOUTS=50
HORIZON_MARGIN=6
HORIZON_CAP=30
SEED=0

OUT_DIR="results/n${N}_${NUM_RANDOM}"

PHI_CSV="${OUT_DIR}/phi_sweep.csv"
EVAL_CSV="${OUT_DIR}/control_eval.csv"
EVAL_PNG="${OUT_DIR}/control_eval.png"
FIG="${OUT_DIR}/toy_final"

if [ ! -f "${PHI_CSV}" ]; then
    echo "error: ${PHI_CSV} not found. Run ./run.sh first (it does the sweep)."
    exit 1
fi

echo "=== Running eval + plot ==="
echo "n=${N}, phis=${NUM_RANDOM}, tasks=${NUM_TASKS}, rollouts=${NUM_ROLLOUTS}"
echo "Input:  ${PHI_CSV}"
echo "Output: ${OUT_DIR}/"
echo ""

python eval_control_success_jax.py \
    --phi_csv ${PHI_CSV} \
    --n ${N} \
    --num_eval_random ${NUM_RANDOM} \
    --num_tasks ${NUM_TASKS} \
    --num_rollouts ${NUM_ROLLOUTS} \
    --horizon_margin ${HORIZON_MARGIN} \
    --horizon_cap ${HORIZON_CAP} \
    --seed ${SEED} \
    --out_csv ${EVAL_CSV} \
    --out_png ${EVAL_PNG}

if [ $? -ne 0 ]; then
    echo "Eval failed!"
    exit 1
fi

python plot.py \
    --eval_csv ${EVAL_CSV} \
    --sweep_csv ${PHI_CSV} \
    --out ${FIG}

echo ""
echo "=== Done -> ${OUT_DIR}/ ==="
