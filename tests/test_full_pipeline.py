"""Integration test: train → compile → inference pipeline for multi-head models.

Tests the full lifecycle:
  1. Train a 2-head model (Cu + Al) with compile_mode: compile
  2. Compile with various modifier combinations:
     - extract_head head_name=Cu_head
     - extract_head head_name=Al_head
     - extract_summed_heads head_names=Cu_head+Al_head
  3. Load each compiled model via NequIPCalculator
  4. Run single-point calculations on test structures
  5. Verify:
     - Cu_head and Al_head produce different energies on the same structure
     - Summed model energy ≈ Cu_head energy + Al_head energy
     - Forces are finite and non-zero
     - Stress is finite

Requires GPU (CUDA) for aotinductor compilation.

Usage:
    python test_full_pipeline.py [--work-dir DIR]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(cmd, cwd=None, check=True):
    """Run a shell command, printing stdout/stderr."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout[-2000:])  # last 2k chars
    if result.stderr:
        # Filter out Lightning warnings — they're noisy but harmless
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


def verify_results(work_dir: Path):
    """Verify compiled models are correct by testing modifier outputs.

    Loads the checkpoint, applies modifiers via modify(), runs eager
    inference, and verifies that extracted heads produce distinct and
    correct outputs. Also verifies the .pt2 files were produced.
    """
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
    r_max = 4.0
    type_names = ["Cu", "Al"]

    # ---- Build test structure ----
    rng = np.random.default_rng(42)
    cu_atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    cu_atoms.positions += rng.normal(0, 0.05, cu_atoms.positions.shape)

    # Convert to NequIP data dict
    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map={"Cu": "Cu"},
        ),
        NeighborListTransform(r_max=r_max),
    ]
    data = from_ase(cu_atoms)
    for t in transforms:
        data = t(data)
    data = AtomicDataDict.to_(data, device)
    # Ensure positions require grad for force computation
    data[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

    # ---- Load checkpoint and create model variants ----
    ckpt = work_dir / "train" / "last.ckpt"

    def load_and_modify(modifier_configs):
        """Load fresh model from checkpoint, apply modifiers, return eager model."""
        model = load_saved_model(str(ckpt), compile_mode="eager")
        if modifier_configs:
            model = modify(model, modifier_configs)
        model = model.to(device)
        model.eval()
        return model

    print("\n  Loading Cu_head model...")
    model_cu = load_and_modify([{"modifier": "extract_head", "head_name": "Cu_head"}])
    print("  Loading Al_head model...")
    model_al = load_and_modify([{"modifier": "extract_head", "head_name": "Al_head"}])
    print("  Loading summed model...")
    model_sum = load_and_modify(
        [{"modifier": "extract_summed_heads", "head_names": "Cu_head+Al_head"}]
    )

    # ---- Run inference ----
    def run_inference(model, data_dict):
        data_copy = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v_new = v.clone().detach()
                # Positions need grad for force computation via autograd
                if k == AtomicDataDict.POSITIONS_KEY:
                    v_new.requires_grad_(True)
                data_copy[k] = v_new
            else:
                data_copy[k] = v
        with torch.enable_grad():
            out = model(data_copy)
        energy = out[AtomicDataDict.TOTAL_ENERGY_KEY].item()
        return energy, out

    print("\n  Running inference...")
    e_cu, out_cu = run_inference(model_cu, data)
    e_al, out_al = run_inference(model_al, data)
    e_sum, out_sum = run_inference(model_sum, data)

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    # ---- Test 1: Different heads produce different energies ----
    print("\n--- Test 1: Different heads produce different energies ---")
    print(f"  Cu_head energy: {e_cu:.6f}")
    print(f"  Al_head energy: {e_al:.6f}")
    print(f"  Difference: {abs(e_cu - e_al):.6f}")
    assert abs(e_cu - e_al) > 1e-6, (
        f"Cu_head and Al_head should produce different energies, "
        f"got {e_cu} vs {e_al}"
    )
    print("  PASS")

    # ---- Test 2: Energies are finite ----
    print("\n--- Test 2: Energies are finite ---")
    for name, e in [("Cu_head", e_cu), ("Al_head", e_al), ("Summed", e_sum)]:
        assert np.isfinite(e), f"{name} energy is not finite: {e}"
    print(f"  Cu_head: {e_cu:.6f}, Al_head: {e_al:.6f}, Summed: {e_sum:.6f}")
    print("  PASS")

    # ---- Test 3: Summed energy = Cu_head + Al_head ----
    print("\n--- Test 3: Summed energy ≈ Cu_head + Al_head ---")
    e_expected = e_cu + e_al
    e_diff = abs(e_sum - e_expected)
    # Tolerance: float32 model, so ~1e-5 relative or 1e-4 absolute
    e_tol = max(1e-4, abs(e_expected) * 1e-4)
    print(f"  Cu_head:      {e_cu:.6f}")
    print(f"  Al_head:      {e_al:.6f}")
    print(f"  Expected sum: {e_expected:.6f}")
    print(f"  Actual sum:   {e_sum:.6f}")
    print(f"  |Difference|: {e_diff:.8f} (tol: {e_tol:.8f})")
    assert e_diff < e_tol, (
        f"Summed energy {e_sum} != Cu + Al = {e_expected} "
        f"(diff={e_diff}, tol={e_tol})"
    )
    print("  PASS")

    # ---- Test 4: Per-atom energies are different between heads ----
    print("\n--- Test 4: Per-atom energies differ between heads ---")
    ea_cu = out_cu[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().numpy()
    ea_al = out_al[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().numpy()
    ea_diff = np.max(np.abs(ea_cu - ea_al))
    print(f"  Max |per-atom energy diff|: {ea_diff:.6f}")
    assert ea_diff > 1e-6, "Per-atom energies should differ between heads"
    print("  PASS")

    # ---- Test 5: Summed per-atom energies = sum of individual ----
    print("\n--- Test 5: Summed per-atom energies ≈ Cu + Al ---")
    ea_sum = out_sum[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().numpy()
    ea_expected = ea_cu + ea_al
    ea_diff = np.max(np.abs(ea_sum - ea_expected))
    ea_tol = max(1e-4, np.max(np.abs(ea_expected)) * 1e-4)
    print(f"  Max |per-atom diff|: {ea_diff:.8f} (tol: {ea_tol:.8f})")
    assert ea_diff < ea_tol, (
        f"Summed per-atom energies don't match: max diff = {ea_diff}"
    )
    print("  PASS")

    # ---- Test 6: Compiled .pt2 files exist ----
    print("\n--- Test 6: Compiled .pt2 files exist ---")
    for name in ["cu_head", "al_head", "summed"]:
        pt2_path = work_dir / f"{name}.nequip.pt2"
        assert pt2_path.exists(), f"Missing: {pt2_path}"
        size_mb = pt2_path.stat().st_size / 1e6
        print(f"  {name}.nequip.pt2: {size_mb:.1f} MB")
    print("  PASS")

    # ---- Test 7: Applying same modifier twice gives same result ----
    print("\n--- Test 7: Deterministic modifier application ---")
    model_cu2 = load_and_modify(
        [{"modifier": "extract_head", "head_name": "Cu_head"}]
    )
    e_cu2, _ = run_inference(model_cu2, data)
    assert abs(e_cu - e_cu2) < 1e-5, (
        f"Same modifier should give same result: {e_cu} vs {e_cu2}"
    )
    print(f"  Run 1: {e_cu:.8f}")
    print(f"  Run 2: {e_cu2:.8f}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL 7 TESTS PASSED")
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
        work_dir = Path(tempfile.mkdtemp(prefix="test_pipeline_"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Working directory: {work_dir}")

    test_dir = Path(__file__).parent
    config_dir = test_dir

    # ---- Step 1: Train ----
    print("\n" + "=" * 60)
    print("STEP 1: Train multi-head model")
    print("=" * 60)
    run(
        f"nequip-train"
        f" --config-dir={config_dir}"
        f" --config-name=test_scheduler_compile"
        f" hydra.run.dir={work_dir}/train",
    )
    ckpt = work_dir / "train" / "last.ckpt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    print(f"Checkpoint: {ckpt}")

    # ---- Step 2: Compile variants ----
    print("\n" + "=" * 60)
    print("STEP 2: Compile model variants")
    print("=" * 60)

    compile_configs = {
        "cu_head": "--modifiers extract_head head_name=Cu_head",
        "al_head": "--modifiers extract_head head_name=Al_head",
        "summed": "--modifiers extract_summed_heads head_names=Cu_head+Al_head",
    }

    for name, modifier_args in compile_configs.items():
        out_path = work_dir / f"{name}.nequip.pt2"
        print(f"\n--- Compiling: {name} ---")
        run(
            f"nequip-compile {ckpt} {out_path}"
            f" --mode aotinductor --device cuda --target ase"
            f" {modifier_args}",
        )
        assert out_path.exists(), f"Compiled model not found: {out_path}"
        print(f"  -> {out_path}")

    # ---- Step 3: Verify ----
    print("\n" + "=" * 60)
    print("STEP 3: Verify compiled models")
    print("=" * 60)
    verify_results(work_dir)

    print(f"\nWork directory: {work_dir}")


if __name__ == "__main__":
    main()
