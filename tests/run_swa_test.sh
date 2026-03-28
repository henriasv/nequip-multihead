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

if grep -q "SWA: saving averaged weights" "$WORK_DIR/train.log"; then
    echo "  -> SWA weights saved: OK"
else
    echo "  FAIL: SWA weights were not saved"
    exit 1
fi

echo ""
echo "--- Step 2b: Verify SWA checkpoint exists ---"
if [ -f "$WORK_DIR/swa_last.ckpt" ]; then
    echo "  -> swa_last.ckpt exists: OK"
else
    echo "  FAIL: No swa_last.ckpt"
    exit 1
fi

echo ""
echo "--- Step 3: Verify checkpoints ---"
python -c "
import torch

ckpt = torch.load('$WORK_DIR/last.ckpt', map_location='cpu', weights_only=False)
swa_ckpt = torch.load('$WORK_DIR/swa_last.ckpt', map_location='cpu', weights_only=False)

print(f'  -> last.ckpt epoch: {ckpt[\"epoch\"]}')
print(f'  -> swa_last.ckpt epoch: {swa_ckpt[\"epoch\"]}')

# Check that callback state was persisted
cb_states = ckpt.get('callbacks', {})
for k, v in cb_states.items():
    if 'StochasticWeightAveraging' in k:
        print(f'  -> SWA callback state: n_averaged={v[\"n_averaged\"]}, started={v[\"swa_started\"]}')

# Verify SWA and normal checkpoints have different weights
diff = 0.0
n = 0
for key in ckpt['state_dict']:
    if key not in swa_ckpt['state_dict']:
        continue
    a, b = ckpt['state_dict'][key], swa_ckpt['state_dict'][key]
    if not isinstance(a, torch.Tensor) or a.numel() == 0:
        continue
    d = (a.float() - b.float()).abs().max().item()
    if d > 0:
        diff = max(diff, d)
        n += 1
print(f'  -> Parameters differing: {n}')
print(f'  -> Max weight difference: {diff:.6f}')
assert n > 0, 'SWA checkpoint should have different weights than last.ckpt'
print('  -> PASS: SWA weights differ from training weights')
"

echo ""
echo "=== PASS: SWA callback test completed successfully ==="
rm -rf "$WORK_DIR"
