#!/bin/bash
# Integration test: full train → compile → inference pipeline.
#
# Tests that extract_head and extract_summed_heads modifiers produce
# correct compiled models with distinct per-head outputs.
#
# Usage: cd ~/repos/nequip-multihead/tests && sbatch run_full_pipeline_test.sh

#SBATCH --job-name=test-pipeline
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test-pipeline_%j.out
#SBATCH --error=test-pipeline_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

# Use absolute path to the tests directory
TESTS_DIR="$HOME/repos/nequip-multihead/tests"
cd "$TESTS_DIR"

echo "=== Full Pipeline Integration Test ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "NequIP: $(python -c 'import nequip; print(nequip.__version__)')"
echo "nequip-multihead: $(python -c 'import nequip_multihead; print(nequip_multihead.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

python "$TESTS_DIR/test_full_pipeline.py"

echo ""
echo "=== Done ==="
