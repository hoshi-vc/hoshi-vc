# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).
#
# ## Attempt 03: FragmentVC を再現する
#
# 類似性検索を使って Fragment VC を実装してみたら、どれくらいうまくいくのか確かめてみる。

# %%

import random
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import callbacks as C
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from engine.dataset_feats import (FeatureDataset, FeatureEntry, IntraDomainDataset, IntraDomainEntry)
from engine.fragment_vc.model import FragmentVC
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.lib.grad_rev import GradientReversal
from engine.lib.utils import DATA_DIR, NPArray, clamp, make_parents, save_ckpt
from engine.lib.utils_ui import play_audio, plot_spectrogram, plot_spectrograms
from engine.preparation import (CREPE_MODEL, FEATS_DIR, PITCH_TOPK, Preparation, pad_clip)

device = "cuda"

P = Preparation(device)

# Pytorch Lightning
#   https://lightning.ai/docs/pytorch/latest/starter/introduction.html
# Module
#   https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
# DataModule
#   https://lightning.ai/docs/pytorch/stable/data/datamodule.html#what-is-a-datamodule

class IntraDomainDataModule(L.LightningDataModule):
  # 公式実装では intra_valid からも intra_train の音声を参照用にサンプリングしてる。
  # TODO: うまくいかなかったら、公式実装と同じ挙動にしてみる。
  def __init__(self, frames: int, n_samples: int, batch_size: int, num_workers: int = 4):
    super().__init__()
    self.frames = frames
    self.n_samples = n_samples
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.intra_train: Optional[IntraDomainDataset] = None
    self.intra_valid: Optional[IntraDomainDataset] = None

  def setup(self, stage: str):
    P.prepare_feats()
    train_dirs = [FEATS_DIR / "parallel100" / sid for sid in P.dataset.speaker_ids]
    valid_dirs = [FEATS_DIR / "nonpara30" / sid for sid in P.dataset.speaker_ids]
    self.intra_train = IntraDomainDataset(train_dirs, self.frames, self.frames, self.n_samples, random_offset=True)
    self.intra_valid = IntraDomainDataset(valid_dirs, self.frames, self.frames, self.n_samples, random_offset=False)

  def train_dataloader(self):
    return DataLoader(self.intra_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.intra_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class FragmentVCModule(L.LightningModule):
  def __init__(self, warmup_steps: int, total_steps: int, milestones: tuple[int, int], exclusive_rate: float):
    super().__init__()
    self.fragment_vc = FragmentVC()

    # save_hyperparameters() によって self.hparams.lr = lr が自動的に行われるらしいけど、
    # 型チェックやオートコンプリートが働かないので self.lr から値を参照することにした。
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.milestones = milestones
    self.exclusive_rate = exclusive_rate

    # save hyper-parameters to self.hparams auto-logged by wandb
    self.save_hyperparameters()  # __init__ のすべての引数を self.hparams に保存する

  def _process_batch(self, batch: IntraDomainEntry, self_exclude: float, ref_included: bool) -> tuple[Tensor, Tensor]:
    src = batch[0].w2v2
    tgt = batch[0].mel
    ref = [o.mel for o in batch[1:]]
    if ref_included:
      if random.random() >= self_exclude:
        ref = torch.cat([*ref, tgt], dim=1)
      else:
        ref = torch.cat(ref, dim=1)
    else:
      ref = tgt

    y = tgt
    y_hat, _ = self.fragment_vc(src, ref.transpose(1, 2))
    y_hat = y_hat.transpose(1, 2)

    assert y.shape == y_hat.shape

    return y, y_hat

  def training_step(self, batch: IntraDomainEntry, batch_idx: int):
    step = self.trainer.global_step
    milestones = self.milestones
    milestone_progress = (step - milestones[0]) / (milestones[1] - milestones[0])
    self_exclude = clamp(milestone_progress, 0.0, 1.0) * self.exclusive_rate
    ref_included = step > milestones[0]

    if step == milestones[0]:
      print("Changing the optimizer learning rate.")
      assert len(self.trainer.optimizers) == 1
      assert len(self.trainer.optimizers[0].param_groups) == 2

      optimizer = self.trainer.optimizers[0]
      optimizer.param_groups[0]["initial_lr"] = 1e-6
      optimizer.param_groups[1]["initial_lr"] = 1e-4

    y_hat, y = self._process_batch(batch, self_exclude, ref_included)
    loss = F.l1_loss(y_hat, y)

    self.log("train_loss", loss)
    self.log("self_exclude_rate", self_exclude)
    self.log("lr_unet", self.trainer.optimizers[0].param_groups[0]["lr"])
    self.log("lr_others", self.trainer.optimizers[0].param_groups[1]["lr"])
    return loss

  def validation_step(self, batch: IntraDomainEntry, batch_idx: int):
    y = batch[0].mel
    y_hat, y = self._process_batch(batch, self_exclude=1.0, ref_included=True)
    loss = F.l1_loss(y_hat, y)

    self.log("valid_loss", loss)
    if batch_idx == 0:
      wandb_logger: Run = self.logger.experiment
      for i in range(4):
        wandb_logger.log({f"spectrogram/{i:02d}": wandb.Image(plot_spectrograms(y[i], y_hat[i]))})

    return loss

  def configure_optimizers(self):
    # NOTE: training_step にて lr を調節するときに params_groups にアクセスしているので、順番などを変えるときは気をつける。
    params_unet = self.fragment_vc.unet.parameters()
    parameters_unet_set = set(params_unet)
    params_others = [p for p in self.parameters() if p not in parameters_unet_set]
    params = [
        {
            "params": params_unet,
        },
        {
            "params": params_others,
        },
    ]
    optimizer = AdamW(params, lr=1e-3)

    scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.total_steps)

    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# init the module

torch.set_float32_matmul_precision("medium")  # TODO: 精度落として問題ない？

fragment_vc = FragmentVCModule(warmup_steps=500, total_steps=250000, milestones=(50000, 150000), exclusive_rate=1.0)

intra_datamodule = IntraDomainDataModule(frames=256, n_samples=10, batch_size=8, num_workers=8)

(DATA_DIR / "wandb").mkdir(parents=True, exist_ok=True)
wandb_logger = WandbLogger(entity="hoshi-vc", project="attempt03", save_dir=DATA_DIR)
run_name = wandb_logger.experiment.name

checkpoint_callback = C.ModelCheckpoint(
    dirpath=DATA_DIR / "attempt03" / "checkpoints" / run_name,
    filename="{step:08d}-{valid_loss:.2f}",
    monitor="valid_loss",
    mode="min",
    save_top_k=3,
)

trainer = L.Trainer(
    max_steps=fragment_vc.total_steps,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    precision="16-mixed",
)

# train the model
trainer.fit(model=fragment_vc, datamodule=intra_datamodule)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
