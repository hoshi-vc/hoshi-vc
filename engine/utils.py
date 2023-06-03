from os import path
from typing import Any

import lightning.pytorch as L
import matplotlib
import torch
from lightning.pytorch import callbacks as C
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from engine.dataset_feats import IntraDomainDataset, IntraDomainDataset2
from engine.lib.utils import DATA_DIR
from engine.preparation import FEATS_DIR, Preparation

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

class IntraDomainDataModule(L.LightningDataModule):
  # 公式実装では intra_valid からも intra_train の音声を参照用にサンプリングしてる。
  # TODO: うまくいかなかったら、公式実装と同じ挙動にしてみる。
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4, dataset_class=IntraDomainDataset):
    super().__init__()
    self.P = P
    self.frames = frames
    self.n_samples = n_samples
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.dataset_class = dataset_class
    self.intra_train: Any = None
    self.intra_valid: Any = None

  def setup(self, stage: str):
    self.P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in self.P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in self.P.dataset.speaker_ids]
    self.intra_train = self.dataset_class(train_dirs, self.frames, self.frames, self.n_samples, random_offset=True)
    self.intra_valid = self.dataset_class(valid_dirs, self.frames, self.frames, self.n_samples, random_offset=False)

  def train_dataloader(self):
    return DataLoader(self.intra_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.intra_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class IntraDomainDataModule2(IntraDomainDataModule):
  def __init__(self, P: Preparation, frames: int, n_samples: int, batch_size: int, num_workers=4):
    super().__init__(P, frames, n_samples, batch_size, num_workers, dataset_class=IntraDomainDataset2)
