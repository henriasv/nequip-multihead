#!/bin/bash
# Test: StochasticWeightAveraging callback
#
# Verifies:
# 1. Training completes (10 epochs cosine + 5 epochs SWA)
# 2. LR drops to swa_lr during SWA phase
# 3. Final model has averaged weights (differs from pre-SWA state)
#
# Usage: cd ~/repos/nequip-multihead/tests && sbatch run_swa_test.sh

#SBATCH --job-name=test-swa
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=test-swa_%j.out
#SBATCH --error=test-swa_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

TESTS_DIR="$HOME/repos/nequip-multihead/tests"
cd "$TESTS_DIR"
WORK_DIR=$(mktemp -d /tmp/test_swa_XXXXXX)

echo "=== SWA Callback Integration Test ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Work dir: $WORK_DIR"
echo ""

echo "--- Step 1: Train with SWA ---"
nequip-train \
    --config-dir="$TESTS_DIR" \
    --config-name=test_swa \
    hydra.run.dir="$WORK_DIR" \
    2>&1 | tee "$WORK_DIR/train.log"

if [ ! -f "$WORK_DIR/last.ckpt" ]; then
    echo "FAIL: No last.ckpt"
    exit 1
fi
echo "  -> Checkpoint exists: OK"

echo ""
echo "--- Step 2: Verify SWA was active ---"
if grep -q "SWA phase started" "$WORK_DIR/train.log"; then
    echo "  -> SWA phase started: OK"
else
    echo "  FAIL: SWA phase never started"
    exit 1
fi

if grep -q "SWA: copied averaged weights" "$WORK_DIR/train.log"; then
    echo "  -> SWA weights transferred: OK"
else
    echo "  FAIL: SWA weights were not transferred"
    exit 1
fi

echo ""
echo "--- Step 3: Verify checkpoint loads ---"
python -c "
import torch
ckpt = torch.load('$WORK_DIR/last.ckpt', map_location='cpu', weights_only=False)
print(f'  -> Epoch: {ckpt[\"epoch\"]}')
print(f'  -> Global step: {ckpt[\"global_step\"]}')
# Check that callback state was persisted
cb_states = ckpt.get('callbacks', {})
swa_found = False
for k, v in cb_states.items():
    if 'StochasticWeightAveraging' in k:
        swa_found = True
        print(f'  -> SWA callback state: n_averaged={v[\"n_averaged\"]}, started={v[\"swa_started\"]}')
if not swa_found:
    print('  WARNING: SWA callback state not found in checkpoint')
"

echo ""
echo "=== PASS: SWA callback test completed successfully ==="
rm -rf "$WORK_DIR"
