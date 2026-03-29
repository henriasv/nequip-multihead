"""Test matrix of architecture combinations: num_layers x l_max x per_head_l_max.

Builds and trains each combination for 2 epochs to verify no crashes.
All use EMT data, tiny features (8), parity=true, compile_mode=eager.

Includes both NequIP and Allegro multi-head models.

Usage:
    python test_arch_matrix.py [--allegro] [--nequip] [--all]
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from itertools import product


NEQUIP_CONFIGS = []

# (num_layers, l_max, per_head_l_max dict or None)
for num_layers in [2, 3, 4]:
    for l_max in [1, 2]:
        # Without per_head_l_max
        NEQUIP_CONFIGS.append((num_layers, l_max, None))
        # With per_head_l_max: head A = full, head B = 0
        NEQUIP_CONFIGS.append((num_layers, l_max, {"Cu_head": l_max, "Al_head": 0}))
        # With per_head_l_max: both restricted to 1 (only if l_max >= 1)
        if l_max >= 2:
            NEQUIP_CONFIGS.append(
                (num_layers, l_max, {"Cu_head": l_max, "Al_head": 1})
            )

# Allegro configs: (num_layers, l_max) — no per_head_l_max for Allegro
ALLEGRO_CONFIGS = []
for num_layers in [1, 2]:
    for l_max in [1, 2]:
        ALLEGRO_CONFIGS.append((num_layers, l_max))


def make_yaml(num_layers, l_max, per_head_l_max):
    """Generate a minimal YAML config string."""
    phm_block = ""
    if per_head_l_max is not None:
        phm_lines = "\n".join(
            f"      {k}: {v}" for k, v in per_head_l_max.items()
        )
        phm_block = f"""
    per_head_l_max:
{phm_lines}"""

    return f"""run: [train]

seed: 42
cutoff_radius: 4.0
model_type_names: [Cu, Al]
chemical_species: ${{model_type_names}}

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 5
  shuffle: true

data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: ${{seed}}
  train_dataset:
    _target_: torch.utils.data.ConcatDataset
    datasets:
      - _target_: nequip.data.dataset.EMTTestDataset
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${{model_type_names}}
            chemical_species_to_atom_type_map:
              Cu: Cu
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${{cutoff_radius}}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 0
        supercell: [2, 2, 2]
        seed: ${{seed}}
        num_frames: 10
        element: Cu
      - _target_: nequip.data.dataset.EMTTestDataset
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${{model_type_names}}
            chemical_species_to_atom_type_map:
              Al: Al
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${{cutoff_radius}}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 1
        supercell: [2, 2, 2]
        seed: 43
        num_frames: 10
        element: Al
  val_dataset:
    - _target_: nequip.data.dataset.EMTTestDataset
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${{model_type_names}}
          chemical_species_to_atom_type_map:
            Cu: Cu
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${{cutoff_radius}}
        - _target_: nequip_multihead.transforms.HeadStamper
          head_index: 0
      supercell: [2, 2, 2]
      seed: 999
      num_frames: 5
      element: Cu
  train_dataloader: ${{dataloader}}
  val_dataloader: ${{dataloader}}
  stats_manager:
    _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
    type_names: ${{model_type_names}}

trainer:
  _target_: lightning.Trainer
  accelerator: gpu
  devices: 1
  max_epochs: 2
  check_val_every_n_epoch: 2
  log_every_n_steps: 5
  enable_progress_bar: false
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: ${{hydra:runtime.output_dir}}
      save_last: true

training_module:
  _target_: nequip.train.EMALightningModule
  loss:
    _target_: nequip.train.EnergyForceStressLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 10.0
      stress: 1.0
  val_metrics:
    _target_: nequip.train.EnergyForceStressMetrics
  train_metrics: null
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.01
  model:
    _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
    seed: ${{seed}}
    model_dtype: float32
    compile_mode: eager
    type_names: ${{model_type_names}}
    r_max: ${{cutoff_radius}}
    l_max: {l_max}
    parity: true
    num_layers: {num_layers}
    num_features: 8
    radial_mlp_depth: 1
    radial_mlp_width: 8
    avg_num_neighbors: ${{training_data_stats:num_neighbors_mean}}
    head_names: [Cu_head, Al_head]
    per_type_energy_scales:
      Cu_head: ${{training_data_stats:forces_rms}}
      Al_head: ${{training_data_stats:forces_rms}}
    per_type_energy_shifts:
      Cu_head:
        Cu: 0.0
        Al: 0.0
      Al_head:
        Cu: 0.0
        Al: 0.0{phm_block}
"""


def make_allegro_yaml(num_layers, l_max):
    """Generate a minimal Allegro multi-head YAML config string."""
    return f"""run: [train]

seed: 42
cutoff_radius: 4.0
model_type_names: [Cu, Al]
chemical_species: ${{model_type_names}}

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 5
  shuffle: true

data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: ${{seed}}
  train_dataset:
    _target_: torch.utils.data.ConcatDataset
    datasets:
      - _target_: nequip.data.dataset.EMTTestDataset
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${{model_type_names}}
            chemical_species_to_atom_type_map:
              Cu: Cu
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${{cutoff_radius}}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 0
        supercell: [2, 2, 2]
        seed: ${{seed}}
        num_frames: 10
        element: Cu
      - _target_: nequip.data.dataset.EMTTestDataset
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${{model_type_names}}
            chemical_species_to_atom_type_map:
              Al: Al
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${{cutoff_radius}}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 1
        supercell: [2, 2, 2]
        seed: 43
        num_frames: 10
        element: Al
  val_dataset:
    - _target_: nequip.data.dataset.EMTTestDataset
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${{model_type_names}}
          chemical_species_to_atom_type_map:
            Cu: Cu
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${{cutoff_radius}}
        - _target_: nequip_multihead.transforms.HeadStamper
          head_index: 0
      supercell: [2, 2, 2]
      seed: 999
      num_frames: 5
      element: Cu
  train_dataloader: ${{dataloader}}
  val_dataloader: ${{dataloader}}
  stats_manager:
    _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
    type_names: ${{model_type_names}}

trainer:
  _target_: lightning.Trainer
  accelerator: gpu
  devices: 1
  max_epochs: 2
  check_val_every_n_epoch: 2
  log_every_n_steps: 5
  enable_progress_bar: false
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: ${{hydra:runtime.output_dir}}
      save_last: true

training_module:
  _target_: nequip.train.EMALightningModule
  loss:
    _target_: nequip.train.EnergyForceStressLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 10.0
      stress: 1.0
  val_metrics:
    _target_: nequip.train.EnergyForceStressMetrics
  train_metrics: null
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.01
  model:
    _target_: nequip_multihead.model.MultiHeadAllegroModel
    seed: ${{seed}}
    model_dtype: float32
    compile_mode: eager
    type_names: ${{model_type_names}}
    r_max: ${{cutoff_radius}}
    l_max: {l_max}
    parity: true
    head_names: [Cu_head, Al_head]
    radial_chemical_embed:
      _target_: allegro.nn.TwoBodyBesselScalarEmbed
    num_layers: {num_layers}
    num_scalar_features: 16
    num_tensor_features: 8
    avg_num_neighbors: ${{training_data_stats:num_neighbors_mean}}
    per_type_energy_scales:
      Cu_head: ${{training_data_stats:forces_rms}}
      Al_head: ${{training_data_stats:forces_rms}}
    per_type_energy_shifts:
      Cu_head:
        Cu: 0.0
        Al: 0.0
      Al_head:
        Cu: 0.0
        Al: 0.0
"""


def desc(num_layers, l_max, per_head_l_max):
    phm = "none" if per_head_l_max is None else str(per_head_l_max)
    return f"L={num_layers} l_max={l_max} phm={phm}"


def allegro_desc(num_layers, l_max):
    return f"Allegro L={num_layers} l_max={l_max}"


def run_config(label, yaml_content, work_root, idx):
    """Write config and run training, return (label, status)."""
    config_dir = work_root / f"config_{idx}"
    config_dir.mkdir()
    train_dir = work_root / f"train_{idx}"

    config_file = config_dir / "cfg.yaml"
    config_file.write_text(yaml_content)

    print(f"[{idx + 1}] {label}")
    result = subprocess.run(
        f"nequip-train --config-dir={config_dir} --config-name=cfg"
        f" hydra.run.dir={train_dir}",
        shell=True,
        capture_output=True,
        text=True,
    )
    ckpt = train_dir / "last.ckpt"
    if result.returncode == 0 and ckpt.exists():
        print(f"  PASS")
        return (label, "PASS")
    else:
        print(f"  FAIL")
        err_lines = result.stderr.strip().splitlines()
        for line in err_lines[-5:]:
            print(f"    {line}")
        return (label, "FAIL")


def main():
    import argparse
    import functools
    print = functools.partial(__builtins__["print"] if isinstance(__builtins__, dict) else getattr(__builtins__, "print"), flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nequip", action="store_true", help="Run NequIP configs only")
    parser.add_argument("--allegro", action="store_true", help="Run Allegro configs only")
    parser.add_argument("--all", action="store_true", help="Run all configs (default)")
    args = parser.parse_args()

    run_nequip = args.nequip or args.all or (not args.nequip and not args.allegro)
    run_allegro = args.allegro or args.all or (not args.nequip and not args.allegro)

    work_root = Path(tempfile.mkdtemp(prefix="test_arch_matrix_"))
    print(f"Work root: {work_root}")

    results = []
    idx = 0

    if run_nequip:
        print(f"\n=== NequIP configs: {len(NEQUIP_CONFIGS)} ===\n")
        for num_layers, l_max, per_head_l_max in NEQUIP_CONFIGS:
            label = desc(num_layers, l_max, per_head_l_max)
            yaml_content = make_yaml(num_layers, l_max, per_head_l_max)
            results.append(run_config(label, yaml_content, work_root, idx))
            idx += 1

    if run_allegro:
        print(f"\n=== Allegro configs: {len(ALLEGRO_CONFIGS)} ===\n")
        for num_layers, l_max in ALLEGRO_CONFIGS:
            label = allegro_desc(num_layers, l_max)
            yaml_content = make_allegro_yaml(num_layers, l_max)
            results.append(run_config(label, yaml_content, work_root, idx))
            idx += 1

    # Summary
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass} passed, {n_fail} failed out of {len(results)}")
    print(f"{'='*60}")
    for label, status in results:
        marker = "OK" if status == "PASS" else "XX"
        print(f"  [{marker}] {label}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
