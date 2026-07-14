#!/bin/bash
# Full pipeline: sweep -> eval -> plot
# All outputs go to results/n${N}_${NUM_RANDOM}/

# Settings
N=4
NUM_RANDOM=2000
NUM_TASKS=600
NUM_ROLLOUTS=50
HORIZON_MARGIN=6
HORIZON_CAP=30
SEED=0

# Output directory
OUT_DIR="results/n${N}_${NUM_RANDOM}"
mkdir -p "${OUT_DIR}"

PHI_CSV="${OUT_DIR}/phi_sweep.csv"
PHI_PNG="${OUT_DIR}/phi_sweep.png"
EVAL_CSV="${OUT_DIR}/control_eval.csv"
EVAL_PNG="${OUT_DIR}/control_eval.png"
FIG="${OUT_DIR}/toy_final"

echo "========================================"
echo "  GCRL-Toy Pipeline"
echo "  n=${N}, phis=${NUM_RANDOM}, tasks=${NUM_TASKS}"
echo "  output: ${OUT_DIR}/"
echo "========================================"
echo ""

# Step 1: Sweep
echo "[1/3] Running sweep..."
python sweep_random_phi_family.py \
    --n ${N} \
    --num_random ${NUM_RANDOM} \
    --seed ${SEED} \
    --out_csv ${PHI_CSV} \
    --out_png ${PHI_PNG}

if [ $? -ne 0 ]; then
    echo "Sweep failed!"
    exit 1
fi
echo "Saved: ${PHI_CSV}"
echo ""

# Step 2: Eval
echo "[2/3] Running eval..."
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
echo "Saved: ${EVAL_CSV}"
echo ""

# Step 3: Figure
echo "[3/3] Making figure..."
python plot.py \
    --eval_csv ${EVAL_CSV} \
    --sweep_csv ${PHI_CSV} \
    --out ${FIG}

if [ $? -ne 0 ]; then
    echo "Plot failed!"
    exit 1
fi
echo ""

echo "========================================"
echo "  Done!"
echo "  Results in ${OUT_DIR}/"
ls -1 "${OUT_DIR}"
echo "========================================"
