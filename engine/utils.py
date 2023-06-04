from os import path

import matplotlib
import torch
import wandb
from lightning.pytorch import callbacks as C
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor

from engine.lib.utils import DATA_DIR
from engine.lib.utils_ui import plot_spectrograms
from engine.preparation import Preparation

def setup_train_environment():
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

def log_spectrograms(self, names: list[str], y: Tensor, y_hat: Tensor):
  for i, name in enumerate(names):
    for i in range(4):
      self.log_wandb({f"spectrogram/{name}": wandb.Image(plot_spectrograms(y[i], y_hat[i]))})
    plt.close("all")

def log_audios(self, P: Preparation, names: list[str], y: Tensor, y_hat: Tensor):
  step = self.batches_that_stepped()

  data = []
  for i, name in enumerate(names):
    audio, sr = P.vocoder(y[i])
    audio_hat, sr_hat = P.vocoder(y_hat[i])
    data.append([
        name,
        wandb.Audio(audio.cpu().to(torch.float32), sample_rate=sr),
        wandb.Audio(audio_hat.cpu().to(torch.float32), sample_rate=sr_hat),
    ])
  self.log_wandb({f"audio/{step:08d}": wandb.Table(data=data, columns=["index", "original", "reconstructed"])})
