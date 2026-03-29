#!/bin/bash
# Test matrix: num_layers x l_max x per_head_l_max combinations
#
# Usage: cd ~/repos/nequip-multihead/tests && sbatch run_arch_matrix_test.sh

#SBATCH --job-name=test-arch-matrix
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=test-arch-matrix_%j.out
#SBATCH --error=test-arch-matrix_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_multihead_ext

TESTS_DIR="$HOME/repos/nequip-multihead/tests"
cd "$TESTS_DIR"

echo "=== Architecture Matrix Test ==="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

python "$TESTS_DIR/test_arch_matrix.py"
