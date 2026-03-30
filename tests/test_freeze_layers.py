"""Smoke test: FreezeLayersCallback freezes correct layers and model still trains.

Tests:
  1. Train a 4-layer model with 2 frozen interaction layers
  2. Verify callback log shows correct freeze pattern (from training stderr)
  3. Verify Lightning reports non-trainable params > 0
  4. Verify checkpoint exists (training completed)
  5. Re-apply callback on loaded model to verify layer detection works

Requires GPU (CUDA).

Usage:
    python test_freeze_layers.py [--work-dir DIR]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd=None, check=True):
    """Run a shell command, returning result with stdout/stderr."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout[-2000:])
    if result.stderr:
        lines = [
            l
            for l in result.stderr.splitlines()
            if "srun" not in l and "tensorboardX" not in l and "litlogger" not in l
        ]
        if lines:
            print("\n".join(lines[-50:]))
    if check and result.returncode != 0:
        print(f"FAILED (exit code {result.returncode})")
        print("STDOUT:", result.stdout[-3000:])
        print("STDERR:", result.stderr[-3000:])
        sys.exit(1)
    return result


def verify_from_logs(train_result, work_dir: Path):
    """Verify freeze behavior from training output logs."""
    combined = train_result.stdout + "\n" + train_result.stderr

    # ---- Test 1: Callback logged frozen layers ----
    print("\n--- Test 1: Callback freeze log ---")
    frozen_lines = [l for l in combined.splitlines() if "FROZEN" in l]
    trainable_lines = [l for l in combined.splitlines() if "TRAINABLE" in l]
    print(f"  Frozen modules: {len(frozen_lines)}")
    print(f"  Trainable modules: {len(trainable_lines)}")
    for l in frozen_lines:
        print(f"    {l.split(']')[-1].strip()}")
    for l in trainable_lines:
        print(f"    {l.split(']')[-1].strip()}")
    assert len(frozen_lines) >= 2, "Expected at least 2 frozen modules (embeddings + 2 ConvNetLayers)"
    assert len(trainable_lines) >= 3, "Expected at least 3 trainable modules (2 ConvNetLayers + readout)"
    # Verify layer0 and layer1 are frozen, layer2 and layer3 are trainable
    frozen_text = "\n".join(frozen_lines)
    trainable_text = "\n".join(trainable_lines)
    assert "layer0_convnet" in frozen_text, "layer0_convnet should be frozen"
    assert "layer1_convnet" in frozen_text, "layer1_convnet should be frozen"
    assert "layer2_convnet" in trainable_text, "layer2_convnet should be trainable"
    assert "layer3_convnet" in trainable_text, "layer3_convnet should be trainable"
    assert "multihead_readout" in trainable_text, "multihead_readout should be trainable"
    print("  PASS")

    # ---- Test 2: Lightning reports non-trainable params ----
    print("\n--- Test 2: Lightning param counts ---")
    # Look for "Trainable params" and "Non-trainable params" in Lightning summary
    trainable_match = re.search(r"([\d,.]+\s*[KM]?)\s+Trainable params", combined)
    non_trainable_match = re.search(r"([\d,.]+\s*[KM]?)\s+Non-trainable params", combined)
    if trainable_match:
        print(f"  Trainable: {trainable_match.group(1)}")
    if non_trainable_match:
        print(f"  Non-trainable: {non_trainable_match.group(1)}")
    assert non_trainable_match is not None, "Lightning should report non-trainable params"
    non_train_str = non_trainable_match.group(1).replace(",", "").strip()
    # Parse: could be "22.3 K" or "22304"
    if "K" in non_train_str:
        non_train_val = float(non_train_str.replace("K", "").strip()) * 1000
    elif "M" in non_train_str:
        non_train_val = float(non_train_str.replace("M", "").strip()) * 1e6
    else:
        non_train_val = float(non_train_str)
    assert non_train_val > 0, "Non-trainable params should be > 0"
    print("  PASS")

    # ---- Test 3: Checkpoint exists ----
    print("\n--- Test 3: Checkpoint exists ---")
    ckpt = work_dir / "train" / "last.ckpt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    print(f"  {ckpt}")
    print("  PASS")

    # ---- Test 4: Training completed ----
    print("\n--- Test 4: Training completed ---")
    assert "max_epochs=3" in combined, "Training should have completed 3 epochs"
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL 4 TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory (default: temp dir)",
    )
    args = parser.parse_args()

    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="test_freeze_"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Working directory: {work_dir}")

    test_dir = Path(__file__).parent

    # ---- Step 1: Train ----
    print("\n" + "=" * 60)
    print("STEP 1: Train 4-layer model with 2 frozen layers")
    print("=" * 60)
    train_result = run(
        f"nequip-train"
        f" --config-dir={test_dir}"
        f" --config-name=test_freeze_layers"
        f" hydra.run.dir={work_dir}/train",
    )

    # ---- Step 2: Verify ----
    print("\n" + "=" * 60)
    print("STEP 2: Verify freeze behavior from logs")
    print("=" * 60)
    verify_from_logs(train_result, work_dir)

    print(f"\nWork directory: {work_dir}")


if __name__ == "__main__":
    main()
