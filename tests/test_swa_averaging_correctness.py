"""Verify SWA averaging correctness: save per-epoch snapshots, manually average,
compare with swa_last.ckpt.

Uses save_snapshots=true to save individual weight snapshots at each SWA
epoch. Then loads all snapshots, computes the equal-weight average, and
checks it matches the SWA model in swa_last.ckpt.

Requires GPU (CUDA).

Usage:
    python test_swa_averaging_correctness.py
"""

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def run(cmd, check=True):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 3} lines) ...")
            print("\n".join(lines[-3:]))
        else:
            print(result.stdout.strip())
    if check and result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        sys.exit(1)
    return result


def main():
    work_dir = Path(tempfile.mkdtemp(prefix="test_swa_avg_"))
    test_dir = Path(__file__).parent
    train_dir = work_dir / "train"
    print(f"Work dir: {work_dir}")

    # ---- Step 1: Train with save_snapshots=true ----
    print("\n=== Step 1: Train with SWA + save_snapshots ===")
    run(
        f"nequip-train"
        f" --config-dir={test_dir}"
        f" --config-name=test_swa_e2e"
        f" hydra.run.dir={train_dir}"
        f" '+trainer.callbacks.1.save_snapshots=true'",
    )

    swa_ckpt = train_dir / "swa_last.ckpt"
    snap_dir = train_dir / "swa_snapshots"
    assert swa_ckpt.exists(), f"Missing: {swa_ckpt}"
    assert snap_dir.exists(), f"Missing snapshot dir: {snap_dir}"

    snapshots = sorted(snap_dir.glob("snapshot_*.pt"))
    print(f"  Snapshots saved: {len(snapshots)}")
    assert len(snapshots) == 5, f"Expected 5 snapshots, got {len(snapshots)}"
    for s in snapshots:
        print(f"    {s.name}")

    # ---- Step 2: Load all snapshots and compute manual average ----
    print("\n=== Step 2: Compute manual average of snapshots ===")
    all_params = []
    for snap_path in snapshots:
        params = torch.load(str(snap_path), map_location="cpu", weights_only=True)
        all_params.append(params)

    n = len(all_params)
    manual_avg = [torch.zeros_like(p) for p in all_params[0]]
    for params in all_params:
        for avg_p, snap_p in zip(manual_avg, params):
            avg_p.add_(snap_p)
    for avg_p in manual_avg:
        avg_p.div_(n)
    print(f"  Averaged {n} snapshots ({len(manual_avg)} parameters each)")

    # ---- Step 3: Also compute running average (same formula as callback) ----
    print("\n=== Step 3: Compute running average (callback formula) ===")
    running_avg = [p.clone() for p in all_params[0]]
    n_running = 1
    for i in range(1, len(all_params)):
        for avg_p, snap_p in zip(running_avg, all_params[i]):
            avg_p.add_((snap_p - avg_p) / (n_running + 1))
        n_running += 1

    # Verify running average == direct mean
    formula_diff = 0.0
    for manual_p, running_p in zip(manual_avg, running_avg):
        d = (manual_p.float() - running_p.float()).abs().max().item()
        formula_diff = max(formula_diff, d)
    print(f"  Max diff (direct mean vs running avg): {formula_diff:.2e}")
    assert formula_diff < 1e-5, f"Running average formula error: {formula_diff}"
    print("  PASS: Running average formula matches direct mean")

    # ---- Step 4: Load swa_last.ckpt and compare ----
    print("\n=== Step 4: Compare manual average with swa_last.ckpt ===")
    swa_data = torch.load(str(swa_ckpt), map_location="cpu", weights_only=False)
    swa_sd = swa_data["state_dict"]

    # Extract model parameter keys (skip EMA buffers and non-param entries)
    param_keys = []
    for k, v in swa_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "ema.ema_weight_" in k:
            continue
        if v.numel() == 0:
            continue
        param_keys.append(k)

    # The snapshot saves model.parameters() in order. The checkpoint
    # state_dict has keys like "model.sole_model.model.XXX".
    # We need to match ordering — iterate both in the same order.
    #
    # Approach: the snapshot has params in model.parameters() order,
    # and the state_dict has named keys. We filter state_dict to
    # parameter-like keys and sort to match model.parameters() order.
    # But this ordering isn't guaranteed to match...
    #
    # More robust: load the checkpoint as a model and iterate
    # model.parameters() in the same order as the snapshot.

    from nequip.model.saved_models.load_utils import load_saved_model
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)

    model = load_saved_model(str(swa_ckpt), compile_mode="eager")
    model_params = list(model.parameters())

    assert len(model_params) == len(manual_avg), (
        f"Parameter count mismatch: model has {len(model_params)}, "
        f"manual avg has {len(manual_avg)}"
    )

    max_diff = 0.0
    n_compared = 0
    for manual_p, model_p in zip(manual_avg, model_params):
        if manual_p.numel() == 0:
            continue
        d = (manual_p.float() - model_p.cpu().float()).abs().max().item()
        max_diff = max(max_diff, d)
        n_compared += 1

    print(f"  Parameters compared: {n_compared}")
    print(f"  Max difference (manual avg vs swa_last.ckpt): {max_diff:.2e}")
    assert max_diff < 1e-4, (
        f"Manual average doesn't match swa_last.ckpt: max diff = {max_diff}"
    )
    print("  PASS: Manual average matches SWA checkpoint")

    # ---- Step 5: Verify snapshots are actually different from each other ----
    print("\n=== Step 5: Verify snapshots differ (training progressed) ===")
    snap_diffs = []
    for i in range(1, len(all_params)):
        d = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(all_params[0], all_params[i])
            if a.numel() > 0
        )
        snap_diffs.append(d)
    print(f"  Max diffs from snapshot 1: {[f'{d:.4f}' for d in snap_diffs]}")
    assert all(d > 1e-6 for d in snap_diffs), "Snapshots should differ"
    print("  PASS: Snapshots are all different (training progressed)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
