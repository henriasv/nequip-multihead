#!/bin/bash
# Run the Allegro multi-head integration test.
# Requires: GPU, allegro + nequip-multihead installed in active env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/test_allegro_multihead.py" "$@"
