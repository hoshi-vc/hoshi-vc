# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

# 各種パラメーターは、モデル定義の `__init__` 内に埋め込むか、あるいは一番下の呼び出し部分で指定するようにしている。
# 冗長なバケツリレーを避けたいので、モデル定義の `__init__` 内にもマジックナンバーを埋め込むことにした。

from pathlib import Path
from random import Random
from typing import NamedTuple, Optional

import lightning.pytorch as L
import torch
import torch.functional as F
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.optim import AdamW
from wandb.wandb_run import Run

from engine.dataset_feats import IntraDomainDataModule2, IntraDomainEntry2
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.lib.layers import Buckets, GetNth, Transpose
from engine.lib.utils import clamp
from engine.prepare import Preparation
from engine.utils import (log_audios, log_spectrograms, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment)

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
    phoneme_dim = hdim // 2
    mel_dim = hdim // 2

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.phoneme_embed = nn.Embedding(400, phoneme_dim)
    self.encode_key = nn.Sequential(
        # input: (batch, src_len, hdim)
        nn.Linear(energy_dim + pitch_dim + phoneme_dim, hdim),
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

    self.mel_encode = nn.Linear(80, mel_dim)
    self.encode_value = nn.Sequential(
        # input: (batch, src_len, hdim)
        nn.Linear(energy_dim + pitch_dim + mel_dim, hdim),
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
        # input: (batch, src_len, hdim + energy_dim + pitch_dim)
        Transpose(1, 2),
        nn.Conv1d(hdim + energy_dim + pitch_dim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.Linear(hdim, 80),
    )

  def forward_energy(self, energy_i: Tensor):
    return self.energy_embed(self.energy_bins(energy_i[:, :, 0]))

  def forward_pitch(self, pitch_i: Tensor):
    return self.pitch_embed(pitch_i[:, :, 0])

  def forward_phoneme(self, phoneme_i: Tensor, phoneme_v: Tensor):
    phoneme: Optional[Tensor] = None
    for k in range(phoneme_v.shape[-1]):
      emb_k = self.phoneme_embed(phoneme_i[:, :, k])
      emb_k *= phoneme_v[:, :, k].exp().unsqueeze(-1)
      phoneme = emb_k if phoneme is None else phoneme + emb_k
    return phoneme

  def forward_mel(self, mel: Tensor):
    return self.mel_encode(mel)

  def forward_key(self, energy: Tensor, pitch: Tensor, phoneme: Tensor):
    return self.encode_key(torch.cat([energy, pitch, phoneme], dim=-1))

  def forward_value(self, energy: Tensor, pitch: Tensor, mel: Tensor):
    return self.encode_value(torch.cat([energy, pitch, mel], dim=-1))

  def forward(self, o: Input04):

    # key: 似たような発音ほど近い表現になってほしい
    #      話者性が多少残ってても lookup 後の value への影響は間接的なので多分問題ない

    # value: 可能な限り多くの発音情報や話者性を含む表現になってほしい
    #        ただし、ピッチや音量によらない表現になってほしい
    #        （デコード時にピッチと音量を調節するので、そこと情報の衝突が起きないでほしい）

    # shape: (batch, src_len, hdim)
    src_energy = self.forward_energy(o.src_energy)
    src_pitch = self.forward_pitch(o.src_pitch_i)
    src_phoneme = self.forward_phoneme(o.src_phoneme_i, o.src_phoneme_v)
    src_key = self.forward_key(src_energy, src_pitch, src_phoneme)

    ref_energy = self.forward_energy(o.ref_energy)
    ref_pitch = self.forward_pitch(o.ref_pitch_i)
    ref_phoneme = self.forward_phoneme(o.ref_phoneme_i, o.ref_phoneme_v)
    ref_key = self.forward_key(ref_energy, ref_pitch, ref_phoneme)

    ref_mel = self.forward_mel(o.ref_mel)
    ref_value = self.forward_value(ref_energy, ref_pitch, ref_mel)

    tgt_value, _ = self.lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    return tgt_mel, (ref_key, ref_value)

class VCModule(L.LightningModule):
  def __init__(self, hdim: int, lr: float, warmup_steps: int, total_steps: int, milestones: tuple[int, int]):
    super().__init__()
    self.model = VCModel(hdim=hdim)

    self.batch_rand = Random(94324203)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.milestones = milestones
    self.lr = lr

    self.save_hyperparameters()

  def batches_that_stepped(self):
    # https://github.com/Lightning-AI/lightning/issues/13752
    # same as 'trainer/global_step' of wandb logger
    return self.trainer.fit_loop.epoch_loop._batches_that_stepped

  def log_wandb(self, item: dict):
    wandb_logger: Run = self.logger.experiment
    wandb_logger.log(item, step=self.batches_that_stepped())

  def _process_batch(self, batch: IntraDomainEntry2, self_exclude: float):
    src = batch[0]

    if self.batch_rand.random() >= self_exclude:
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

    reconst = F.l1_loss(y_hat, y)

    loss_model = reconst

    return y_hat, loss_model, (reconst,)

  def training_step(self, batch: IntraDomainEntry2, batch_idx: int):
    step = self.batches_that_stepped()
    milestones = self.milestones
    progress1 = (step - milestones[0]) / (milestones[1] - milestones[0])
    self_exclude = clamp(progress1, 0.0, 1.0)

    _, loss_model, (reconst,) = self._process_batch(batch, self_exclude)

    self.log("train_loss", loss_model)
    self.log("train_reconst", reconst)
    self.log("self_exclude_rate", self_exclude)
    self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
    return loss_model

  def validation_step(self, batch: IntraDomainEntry2, batch_idx: int):
    y = batch[0].mel

    y_hat_cheat, loss_model, (reconst,) = self._process_batch(batch, self_exclude=0.0)

    self.log("vcheat_loss", loss_model)
    self.log("vcheat_reconst", reconst)

    y_hat, loss_model, (reconst,) = self._process_batch(batch, self_exclude=1.0)

    self.log("valid_loss", loss_model)
    self.log("valid_reconst", reconst)
    if batch_idx == 0:
      names = [f"{i:02d}" for i in range(4)]
      log_spectrograms(self, names, y, y_hat, y_hat_cheat)
      log_audios(self, P, names, y, y_hat, y_hat_cheat)

    return loss_model

  def configure_optimizers(self):
    opt_model = AdamW(self.model.parameters(), lr=self.lr)
    sch_model = get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps)

    return [opt_model], [{"scheduler": sch_model, "interval": "step"}]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModule2(P, frames=256, n_samples=15, batch_size=16)

  total_steps = 10000

  model = VCModule(
      hdim=512,
      lr=1e-3,
      warmup_steps=500,
      total_steps=total_steps,
      milestones=(2000, 40000),
  )

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_steps=total_steps,
      logger=wandb_logger,
      callbacks=[
          new_checkpoint_callback_wandb(PROJECT, wandb_logger),
      ],
      accelerator="gpu",
      precision="16-mixed",
      deterministic=True,
  )

  # train the model
  trainer.fit(model, datamodule=datamodule)

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()
