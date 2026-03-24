# EMA Checkpoint Bug: `best.ckpt` Fails on Load

## Bug

Loading a `best.ckpt` saved during training with `EMALightningModule` fails with:

```
AssertionError: EMA module loaded in a state where it does not contain EMA weights -- the checkpoint file is likely corrupted.
```

This affects `nequip-package`, `nequip-compile`, and any code that loads a checkpoint via `load_from_checkpoint` → `evaluation_model`.

## Root cause

Lightning's `ModelCheckpoint` callback fires `on_validation_end` **before** `LightningModule.on_validation_end`. During validation, `EMALightningModule` swaps EMA weights into the model (`on_validation_start`), then swaps back (`on_validation_end`). But `ModelCheckpoint` saves the checkpoint between these two hooks — while the weights are swapped.

### Hook ordering (verified with Lightning 2.6.1)

```
1. LightningModule.on_validation_start()  → swaps EMA into model
2. [validation runs with EMA weights]
3. ModelCheckpoint.on_validation_end()     → saves checkpoint (SWAPPED STATE)
4. LightningModule.on_validation_end()     → swaps back (TOO LATE)
```

### Verification script

```python
import lightning.pytorch as pl
import torch, sys

class TestModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def on_validation_end(self):
        sys.stderr.write("MODULE on_validation_end\n")
    def training_step(self, batch, batch_idx):
        return self.layer(batch).sum()
    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    def train_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([1.0])], batch_size=1)
    def val_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([1.0])], batch_size=1)

class TestCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        sys.stderr.write("CALLBACK on_validation_end\n")

model = TestModule()
trainer = pl.Trainer(max_epochs=1, callbacks=[TestCallback()],
                     enable_progress_bar=False, enable_model_summary=False,
                     logger=False, accelerator="cpu")
trainer.fit(model)
```

Output:
```
CALLBACK on_validation_end    ← ModelCheckpoint saves here
MODULE on_validation_end      ← EMA swaps back here (too late)
```

## Reproduction

1. Train any NequIP model with `EMALightningModule` and `ModelCheckpoint` monitoring a validation metric
2. Wait for `best.ckpt` to be saved
3. Try to load it:

```bash
nequip-package build best.ckpt model.nequip.zip
```

Result:
```
AssertionError: EMA module loaded in a state where it does not contain EMA weights -- the checkpoint file is likely corrupted.
```

### Actual reproduction from our integration test (March 24, 2026)

```
$ nequip-train --config-dir=. --config-name=config  # trains 10 epochs with EMA
$ nequip-package build output_test/best.ckpt output_test/model.nequip.zip

File ".../nequip/train/ema.py", line 228, in set_extra_state
    assert self.is_holding_ema_weights, (
AssertionError: EMA module loaded in a state where it does not contain EMA weights
```

`last.ckpt` (saved at end of epoch, outside validation) works fine. Only `best.ckpt` (saved by `ModelCheckpoint` during validation) is affected.

## Why nobody noticed before

- `last.ckpt` is saved at epoch end (outside validation) → correct EMA state
- Training restarts use `last.ckpt` by default
- `nequip-compile` from `last.ckpt` works
- `best.ckpt` is only loaded when explicitly packaging/compiling the best model

## Fix

`EMALightningModule.on_save_checkpoint()` detects the swapped state and corrects it before saving. `EMAWeights.set_extra_state()` tolerates old checkpoints via `_needs_post_load_swap` flag.
