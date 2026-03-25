#!/bin/bash
# Test: GradientNormFractionScheduler + compile_mode: compile
#
# Two tests:
#   1. 2-head (Cu+Al) — basic scheduler + compile
#   2. 5-head (mixed NaN data) — scheduler + compile + partial gradients
#
# Usage: sbatch run_scheduler_compile_test.sh

#SBATCH --job-name=test-scheduler-compile
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_scheduler_compile_%j.out
#SBATCH --error=test_scheduler_compile_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0

run_test() {
    local TEST_NAME="$1"
    local CONFIG_NAME="$2"
    local WORK_DIR
    WORK_DIR=$(mktemp -d /tmp/test_${CONFIG_NAME}_XXXXXX)

    echo ""
    echo "============================================================"
    echo "TEST: $TEST_NAME"
    echo "Config: $CONFIG_NAME"
    echo "Work dir: $WORK_DIR"
    echo "============================================================"

    if nequip-train \
        --config-dir="$SCRIPT_DIR" \
        --config-name="$CONFIG_NAME" \
        hydra.run.dir="$WORK_DIR" \
        2>&1 | tee "$WORK_DIR/train.log"; then

        if [ -f "$WORK_DIR/last.ckpt" ]; then
            echo "  -> Checkpoint exists: OK"
            # Verify checkpoint is loadable (just torch.load, no model rebuild)
            if python -c "
import torch
ckpt = torch.load('$WORK_DIR/last.ckpt', map_location='cpu', weights_only=False)
print(f'  -> Checkpoint keys: {list(ckpt.keys())[:5]}')
print(f'  -> Epoch: {ckpt.get(\"epoch\", \"?\")}')
print(f'  -> Global step: {ckpt.get(\"global_step\", \"?\")}')
" 2>&1; then
                echo "  -> PASS"
                PASS=$((PASS + 1))
            else
                echo "  -> FAIL: Corrupt checkpoint"
                FAIL=$((FAIL + 1))
            fi
        else
            echo "  -> FAIL: No last.ckpt"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  -> FAIL: Training crashed"
        echo "  Last 30 lines:"
        tail -30 "$WORK_DIR/train.log"
        FAIL=$((FAIL + 1))
    fi

    rm -rf "$WORK_DIR"
}

echo "=== GradientNormFractionScheduler + compile_mode: compile ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "NequIP: $(python -c 'import nequip; print(nequip.__version__)')"
echo "nequip-multihead: $(python -c 'import nequip_multihead; print(nequip_multihead.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

run_test "2-head (Cu+Al) with scheduler + compile" \
    "test_scheduler_compile"

run_test "5-head (mixed NaN) with scheduler + compile" \
    "test_5head_scheduler_compile"

echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "============================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
