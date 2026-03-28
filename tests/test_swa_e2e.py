"""End-to-end test: SWA callback with EMA + compile + head extraction.

Tests the full lifecycle:
  1. Train with EMA + SWA + compile_mode: compile
  2. Verify swa_last.ckpt and last.ckpt both exist
  3. Compile Cu_head from last.ckpt (EMA model)
  4. Compile Cu_head from swa_last.ckpt (SWA model)
  5. Verify both compiled models produce valid outputs
  6. Verify SWA and EMA models give different energies (they should)
  7. Verify SWA model is deterministic (load twice → same result)

Requires GPU (CUDA).

Usage:
    python test_swa_e2e.py [--work-dir DIR]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        # Only show last bit of stdout
        lines = result.stdout.strip().splitlines()
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 3} lines) ...")
            print("\n".join(lines[-3:]))
        else:
            print(result.stdout.strip())
    if check and result.returncode != 0:
        print(f"FAILED (exit code {result.returncode})")
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        sys.exit(1)
    return result


def verify(work_dir: Path):
    """Load checkpoints, apply modifiers, run inference, compare."""
    import torch
    from ase.build import bulk
    from nequip.data import AtomicDataDict, from_ase
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )
    from nequip.model.saved_models.load_utils import load_saved_model
    from nequip.model.modify_utils import modify
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)
    device = torch.device("cuda")
    type_names = ["Cu", "Al"]

    # Build test structure
    rng = np.random.default_rng(42)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)

    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map={"Cu": "Cu"},
        ),
        NeighborListTransform(r_max=4.0),
    ]
    data = from_ase(atoms)
    for t in transforms:
        data = t(data)
    data = AtomicDataDict.to_(data, device)

    def load_and_run(ckpt_path, modifier_configs):
        model = load_saved_model(str(ckpt_path), compile_mode="eager")
        if modifier_configs:
            model = modify(model, modifier_configs)
        model = model.to(device)
        model.eval()

        data_copy = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v_new = v.clone().detach()
                if k == AtomicDataDict.POSITIONS_KEY:
                    v_new.requires_grad_(True)
                data_copy[k] = v_new
            else:
                data_copy[k] = v

        with torch.enable_grad():
            out = model(data_copy)
        return out[AtomicDataDict.TOTAL_ENERGY_KEY].item()

    ema_ckpt = work_dir / "train" / "last.ckpt"
    swa_ckpt = work_dir / "train" / "swa_last.ckpt"
    extract_cu = [{"modifier": "extract_head", "head_name": "Cu_head"}]

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # --- Test 1: Both checkpoints exist ---
    print("\n--- Test 1: Checkpoints exist ---")
    assert ema_ckpt.exists(), f"Missing: {ema_ckpt}"
    assert swa_ckpt.exists(), f"Missing: {swa_ckpt}"
    ema_size = ema_ckpt.stat().st_size / 1e6
    swa_size = swa_ckpt.stat().st_size / 1e6
    print(f"  last.ckpt:     {ema_size:.1f} MB")
    print(f"  swa_last.ckpt: {swa_size:.1f} MB")
    print("  PASS")

    # --- Test 2: SWA callback state in last.ckpt ---
    print("\n--- Test 2: SWA callback state persisted ---")
    import torch as _torch

    ckpt = _torch.load(str(ema_ckpt), map_location="cpu", weights_only=False)
    cb_states = ckpt.get("callbacks", {})
    swa_state = None
    for k, v in cb_states.items():
        if "StochasticWeightAveraging" in k:
            swa_state = v
    assert swa_state is not None, "SWA callback state not in checkpoint"
    assert swa_state["swa_started"], "SWA should have started"
    assert swa_state["n_averaged"] == 5, (
        f"Expected 5 snapshots, got {swa_state['n_averaged']}"
    )
    print(f"  n_averaged: {swa_state['n_averaged']}")
    print(f"  swa_started: {swa_state['swa_started']}")
    print("  PASS")

    # --- Test 3: Weight diff between last.ckpt and swa_last.ckpt ---
    print("\n--- Test 3: SWA weights differ from EMA weights ---")
    swa_sd = _torch.load(str(swa_ckpt), map_location="cpu", weights_only=False)
    n_diff = 0
    max_diff = 0.0
    for key in ckpt["state_dict"]:
        if key not in swa_sd["state_dict"]:
            continue
        a, b = ckpt["state_dict"][key], swa_sd["state_dict"][key]
        if not isinstance(a, _torch.Tensor) or a.numel() == 0:
            continue
        d = (a.float() - b.float()).abs().max().item()
        if d > 0:
            n_diff += 1
            max_diff = max(max_diff, d)
    print(f"  Parameters differing: {n_diff}")
    print(f"  Max weight difference: {max_diff:.6f}")
    assert n_diff > 0, "SWA and EMA checkpoints should differ"
    print("  PASS")

    # --- Test 4: Load and run Cu_head from EMA checkpoint ---
    print("\n--- Test 4: EMA model inference ---")
    e_ema = load_and_run(ema_ckpt, extract_cu)
    print(f"  EMA Cu_head energy: {e_ema:.6f}")
    assert np.isfinite(e_ema), "EMA energy is not finite"
    print("  PASS")

    # --- Test 5: Load and run Cu_head from SWA checkpoint ---
    print("\n--- Test 5: SWA model inference ---")
    e_swa = load_and_run(swa_ckpt, extract_cu)
    print(f"  SWA Cu_head energy: {e_swa:.6f}")
    assert np.isfinite(e_swa), "SWA energy is not finite"
    print("  PASS")

    # --- Test 6: SWA and EMA give different energies ---
    print("\n--- Test 6: SWA != EMA ---")
    diff = abs(e_swa - e_ema)
    print(f"  EMA energy: {e_ema:.6f}")
    print(f"  SWA energy: {e_swa:.6f}")
    print(f"  Difference: {diff:.6f}")
    assert diff > 1e-6, f"SWA and EMA should give different energies (diff={diff})"
    print("  PASS")

    # --- Test 7: SWA is deterministic ---
    print("\n--- Test 7: SWA deterministic ---")
    e_swa2 = load_and_run(swa_ckpt, extract_cu)
    assert abs(e_swa - e_swa2) < 1e-5, (
        f"Same SWA checkpoint should give same result: {e_swa} vs {e_swa2}"
    )
    print(f"  Run 1: {e_swa:.8f}")
    print(f"  Run 2: {e_swa2:.8f}")
    print("  PASS")

    # --- Test 8: Compile from swa_last.ckpt works ---
    print("\n--- Test 8: Compile Cu_head from swa_last.ckpt ---")
    compiled_path = work_dir / "cu_swa.nequip.pt2"
    run(
        f"nequip-compile {swa_ckpt} {compiled_path}"
        f" --mode aotinductor --device cuda --target ase"
        f" --modifiers extract_head head_name=Cu_head",
    )
    assert compiled_path.exists(), f"Compiled model not found: {compiled_path}"
    size_mb = compiled_path.stat().st_size / 1e6
    print(f"  cu_swa.nequip.pt2: {size_mb:.1f} MB")
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL 8 TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="test_swa_e2e_"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Working directory: {work_dir}")
    test_dir = Path(__file__).parent

    # Step 1: Train
    print("\n" + "=" * 60)
    print("STEP 1: Train with EMA + SWA + compile")
    print("=" * 60)
    run(
        f"nequip-train"
        f" --config-dir={test_dir}"
        f" --config-name=test_swa_e2e"
        f" hydra.run.dir={work_dir}/train",
    )

    # Step 2: Verify
    verify(work_dir)

    print(f"\nWork directory: {work_dir}")


if __name__ == "__main__":
    main()
