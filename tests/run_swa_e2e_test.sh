#!/bin/bash
# End-to-end test: SWA + EMA + compile + head extraction
#
# Usage: cd ~/repos/nequip-multihead/tests && sbatch run_swa_e2e_test.sh

#SBATCH --job-name=test-swa-e2e
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test-swa-e2e_%j.out
#SBATCH --error=test-swa-e2e_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

TESTS_DIR="$HOME/repos/nequip-multihead/tests"
cd "$TESTS_DIR"

echo "=== SWA End-to-End Integration Test ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

python "$TESTS_DIR/test_swa_e2e.py"

echo ""
echo "=== Done ==="
