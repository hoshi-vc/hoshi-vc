# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).
#
# 各種パラメーターは、モデル定義の `__init__` 内に埋め込むか、あるいは一番下の呼び出し部分で指定するようにしている。
#
# 冗長なバケツリレーを避けたいので、モデル定義の `__init__` 内にもマジックナンバーを埋め込むことにした。

# %%

import random
from pathlib import Path
from typing import NamedTuple, Optional

import lightning.pytorch as L
import torch
import torch.functional as F
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.optim import AdamW
from wandb.wandb_run import Run

from engine.dataset_feats import IntraDomainEntry2
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.lib.layers import Buckets, GetNth, Transpose
from engine.lib.utils import clamp
from engine.lib.utils_ui import plot_spectrograms
from engine.preparation import Preparation
from engine.utils import (IntraDomainDataModule2, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment)

class Input04(NamedTuple):
  src_energy: Tensor  #    (batch, src_len, 1)
  src_phoneme_i: Tensor  # (batch, src_len, topk)
  src_phoneme_v: Tensor  # (batch, src_len, topk)
  src_pitch_i: Tensor  #   (batch, src_len, 1+)
  ref_energy: Tensor  #    (batch, ref_len, 1)
  ref_phoneme_i: Tensor  # (batch, ref_len, topk)
  ref_phoneme_v: Tensor  # (batch, src_len, topk)
  ref_pitch_i: Tensor  #   (batch, ref_len, 1+)
  ref_mel: Tensor  #       (batch, ref_len, 80)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    others_dim = hdim - energy_dim - pitch_dim
    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.phoneme_embed = nn.Embedding(400, others_dim)
    self.encode_key = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.RNN(hdim, hdim, batch_first=True),
        GetNth(0),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
    )

    self.mel_encode = nn.Linear(80, others_dim)
    self.encode_value = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
    )

    self.lookup = nn.MultiheadAttention(hdim, 4, dropout=0.1, batch_first=True)

    self.decode = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, 80, kernel_size=1),
        Transpose(1, 2),
    )

  def forward_key(self, energy_i: Tensor, pitch_i: Tensor, phoneme_i: Tensor, phoneme_v: Tensor):
    energy = self.energy_embed(self.energy_bins(energy_i[:, :, 0]))
    pitch = self.pitch_embed(pitch_i[:, :, 0])

    phoneme: Optional[Tensor] = None
    for k in range(phoneme_v.shape[-1]):
      emb_k = self.phoneme_embed(phoneme_i[:, :, k])
      emb_k *= phoneme_v[:, :, k].exp().unsqueeze(-1)
      phoneme = emb_k if phoneme is None else phoneme + emb_k

    key = torch.cat([energy, pitch, phoneme], dim=-1)
    key = self.encode_key(key)

    return key

  def forward_value(self, energy_i: Tensor, pitch_i: Tensor, mel: Tensor):
    energy = self.energy_embed(self.energy_bins(energy_i[:, :, 0]))
    pitch = self.pitch_embed(pitch_i[:, :, 0])
    mel = self.mel_encode(mel)

    value = torch.cat([energy, pitch, mel], dim=-1)
    value = self.encode_value(value)

    return value

  def forward(self, o: Input04):
    # shape: (batch, src_len, hdim)
    src_key = self.forward_key(o.src_energy, o.src_pitch_i, o.src_phoneme_i, o.src_phoneme_v)
    ref_key = self.forward_key(o.ref_energy, o.ref_pitch_i, o.ref_phoneme_i, o.ref_phoneme_v)
    ref_value = self.forward_value(o.ref_energy, o.ref_pitch_i, o.ref_mel)

    tgt_value, _ = self.lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(tgt_value)

    return tgt_mel, (ref_key, ref_value)

class VCModule(L.LightningModule):
  def __init__(self, hdim: int, lr: float, warmup_steps: int, total_steps: int, milestones: tuple[int, int], exclusive_rate: float):
    super().__init__()
    self.model = VCModel(hdim=hdim)

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

if __name__ == "__main__":

  PROJECT = Path(__file__).stem

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModule2(P, frames=256, n_samples=3, batch_size=8)

  model = VCModule(
      hdim=512,
      lr=1e-3,
      warmup_steps=500,
      total_steps=50000,
      milestones=(10000, 20000),
      exclusive_rate=1.0,
  )

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_steps=model.total_steps,
      logger=wandb_logger,
      callbacks=[
          new_checkpoint_callback_wandb(PROJECT, wandb_logger),
      ],
      accelerator="gpu",
      precision="16-mixed",
  )

  # train the model
  trainer.fit(model, datamodule=datamodule)

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()
