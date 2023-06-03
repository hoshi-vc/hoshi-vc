# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).

# %%

import random
from pathlib import Path

import lightning.pytorch as L
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import Tensor
from torch.optim import AdamW
from wandb.wandb_run import Run

from engine.dataset_feats import IntraDomainDataset2, IntraDomainEntry2
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.lib.utils import clamp
from engine.lib.utils_ui import plot_spectrograms
from engine.models import Input04, Model04
from engine.preparation import Preparation
from engine.utils import (IntraDomainDataModule, new_checkpoint_callback, new_wandb_logger, setup_train_environment)

class VCModule(L.LightningModule):
  def __init__(self, warmup_steps: int, total_steps: int, milestones: tuple[int, int], exclusive_rate: float, lr: float):
    super().__init__()
    self.model = Model04(hdim=512)

    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.milestones = milestones
    self.exclusive_rate = exclusive_rate
    self.lr = lr

    self.save_hyperparameters()

  def log_wandb(self, item: dict):
    wandb_logger: Run = self.logger.experiment
    wandb_logger.log(item, step=self.trainer.global_step)

  def _process_batch(self, batch: IntraDomainEntry2, self_exclude: float, ref_included: bool):
    src = batch[0]

    refs = [src]
    if ref_included:
      if random.random() >= self_exclude:
        refs = batch
      else:
        refs = batch[1:]

    y = src.mel
    y_hat: Tensor
    y_hat, (ref_key, ref_value) = self.model(
        Input04(
            src_energy=src.energy,
            src_phoneme_i=src.phoneme_i,
            src_phoneme_v=src.phoneme_v,
            src_pitch_i=src.pitch_i,
            ref_energy=torch.cat([o.energy for o in refs], dim=1),
            ref_phoneme_i=torch.cat([o.phoneme_i for o in refs], dim=1),
            ref_phoneme_v=torch.cat([o.phoneme_v for o in refs], dim=1),
            ref_pitch_i=torch.cat([o.pitch_i for o in refs], dim=1),
            ref_mel=torch.cat([o.mel for o in refs], dim=1),
        ))

    assert y.shape == y_hat.shape

    loss_reconst = F.l1_loss(y_hat, y)
    # loss_kv = F.l1_loss(ref_key, ref_value.detach())

    loss = loss_reconst

    return y_hat, loss, (loss_reconst,)

  def training_step(self, batch: IntraDomainEntry2, batch_idx: int):
    step = self.trainer.global_step
    milestones = self.milestones
    milestone_progress = (step - milestones[0]) / (milestones[1] - milestones[0])
    self_exclude = clamp(milestone_progress, 0.0, 1.0) * self.exclusive_rate
    ref_included = step >= milestones[0]

    y_hat, loss, (loss_reconst,) = self._process_batch(batch, self_exclude, ref_included)

    self.log("train_loss_reconst", loss_reconst)
    self.log("train_loss", loss)
    self.log("self_exclude_rate", self_exclude)
    self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
    return loss

  def validation_step(self, batch: IntraDomainEntry2, batch_idx: int):
    y = batch[0].mel
    y_hat, loss, (loss_reconst,) = self._process_batch(batch, self_exclude=1.0, ref_included=True)

    self.log("valid_loss_reconst", loss_reconst)
    self.log("valid_loss", loss)
    if batch_idx == 0:
      for i in range(4):
        self.log_wandb({f"spectrogram/{i:02d}": wandb.Image(plot_spectrograms(y[i], y_hat[i]))})
      plt.close("all")

    return loss

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.total_steps)

    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# init the module

PROJECT = Path(__file__).stem

setup_train_environment()
P = Preparation("cuda")
datamodule = IntraDomainDataModule(P, frames=256, n_samples=3, batch_size=8, dataset_class=IntraDomainDataset2)
model = VCModule(
    warmup_steps=500,
    total_steps=50000,
    milestones=(10000, 20000),
    exclusive_rate=1.0,
    lr=1e-3,
)

wandb_logger = new_wandb_logger(PROJECT)

trainer = L.Trainer(
    max_steps=model.total_steps,
    logger=wandb_logger,
    callbacks=[
        new_checkpoint_callback(PROJECT, wandb_logger.experiment.name),
    ],
    accelerator="gpu",
    precision="16-mixed",
)

# train the model
trainer.fit(model, datamodule=datamodule)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
