#!/bin/bash
# Test: SWA averaging correctness
#
# Captures per-epoch snapshots during SWA, manually averages them,
# and verifies the result matches the SWA callback's online average.
#
# Usage: cd ~/repos/nequip-multihead/tests && sbatch run_swa_averaging_test.sh

#SBATCH --job-name=test-swa-avg
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=test-swa-avg_%j.out
#SBATCH --error=test-swa-avg_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

TESTS_DIR="$HOME/repos/nequip-multihead/tests"
cd "$TESTS_DIR"

echo "=== SWA Averaging Correctness Test ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

python "$TESTS_DIR/test_swa_averaging_correctness.py"
