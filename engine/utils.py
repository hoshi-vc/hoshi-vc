# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from os import path
from typing import Optional

import matplotlib
import torch
import wandb
from lightning import seed_everything
from lightning.pytorch import callbacks as C
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor

from engine.lib.utils import DATA_DIR
from engine.lib.utils_ui import plot_spectrograms, plot_spectrograms2
from engine.prepare import Preparation

def setup_train_environment():
  seed_everything(90212374, workers=True)
  matplotlib.use("Agg")
  torch.set_float32_matmul_precision("medium")  # TODO: 精度落として問題ない？

def new_wandb_logger(project: str):
  (DATA_DIR / "wandb").mkdir(parents=True, exist_ok=True)
  return WandbLogger(entity="hoshi-vc", project=project, save_dir=DATA_DIR)

def new_checkpoint_callback(project: str, run_path: str, **kwargs):
  return C.ModelCheckpoint(
      dirpath=DATA_DIR / project / "checkpoints" / run_path,
      filename="{step:08d}-{valid_loss:.4f}",
      monitor="valid_loss",
      mode="min",
      save_top_k=3,
      save_last=True,
      **kwargs,
  )

def new_checkpoint_callback_wandb(project: str, wandb_logger: WandbLogger, **kwargs):
  run_name = wandb_logger.experiment.name
  run_id = wandb_logger.experiment.id
  run_path = run_name + path.sep + run_id
  return new_checkpoint_callback(project, run_path, **kwargs)

def log_spectrograms(self, names: list[str], y: Tensor, y_hat: Tensor, y_hat_cheat: Optional[Tensor] = None):
  for i, name in enumerate(names):
    for i in range(4):
      if y_hat_cheat is None:
        self.log_wandb({f"Spectrogram/{name}": wandb.Image(plot_spectrograms2(y[i], y_hat[i], y_hat_cheat[i]))})
      else:
        self.log_wandb({f"Spectrogram/{name}": wandb.Image(plot_spectrograms(y[i], y_hat[i]))})
    plt.close("all")

def log_audios(self, P: Preparation, names: list[str], y: Tensor, y_hat: Tensor, y_hat_cheat: Optional[Tensor] = None):
  step = self.batches_that_stepped()

  columns = ["index", "original", "reconstructed"]
  if y_hat_cheat is not None:
    columns.append("reconstructed_cheat")

  data = []
  for i, name in enumerate(names):
    audio, sr = P.vocoder(y[i])
    audio_hat, sr_hat = P.vocoder(y_hat[i])
    if y_hat_cheat is None:
      data.append([
          name,
          wandb.Audio(audio.cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(audio_hat.cpu().to(torch.float32), sample_rate=sr_hat),
      ])
    else:
      audio_hat_cheat, sr_hat_cheat = P.vocoder(y_hat_cheat[i])
      data.append([
          name,
          wandb.Audio(audio.cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(audio_hat.cpu().to(torch.float32), sample_rate=sr_hat),
          wandb.Audio(audio_hat_cheat.cpu().to(torch.float32), sample_rate=sr_hat_cheat),
      ])

  self.log_wandb({f"Audio/{step:08d}": wandb.Table(data=data, columns=columns)})
