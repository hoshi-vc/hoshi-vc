# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from os import path
from pathlib import Path
from typing import Callable, Optional

import lightning.pytorch as L
import matplotlib
import torch
import torch._dynamo
import torch.nn.functional as F
import wandb
from attr import dataclass
from lightning import seed_everything
from lightning.pytorch import callbacks as C
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor, cosine_similarity, nn
from torch.optim import Optimizer
from wandb.wandb_run import Run

import engine.hifi_gan.models as VOC
from engine.lib.acgan import BasicDiscriminator, aux_loss
from engine.lib.layers import Transpose
from engine.lib.utils import AttrDict, Device
from engine.lib.utils_ui import (plot_attention, plot_spectrograms, plot_spectrograms2)
from engine.singleton import DATA_DIR, P

def setup_train_environment(seed=90212374):
  seed_everything(seed, workers=True)
  matplotlib.use("Agg")
  torch.set_float32_matmul_precision("medium")  # TODO: 精度落として問題ない？

def new_wandb_logger(project: str):
  (DATA_DIR / "wandb").mkdir(parents=True, exist_ok=True)
  return WandbLogger(entity="hoshi-vc", project=project, save_dir=DATA_DIR)

def new_checkpoint_callback(project: str, run_path: str, filename="{step:08d}-{valid_loss:.4f}", monitor="valid_loss", mode="min", **kwargs):
  return C.ModelCheckpoint(
      dirpath=DATA_DIR / project / "checkpoints" / run_path,
      filename=filename,
      monitor=monitor,
      mode=mode,
      save_top_k=3,
      save_last=True,
      **kwargs,
  )

def new_checkpoint_callback_wandb(project: str, wandb_logger: WandbLogger, **kwargs):
  run_name = wandb_logger.experiment.name
  run_id = wandb_logger.experiment.id
  run_path = run_name + path.sep + run_id
  return new_checkpoint_callback(project, run_path, **kwargs)

@torch._dynamo.disable()
def log_spectrograms(self, names: list[str], y: Tensor, y_hat: Tensor, y_hat_cheat: Optional[Tensor] = None, folder="Spectrogram"):
  for i, name in enumerate(names):
    if y_hat_cheat is not None:
      self.log_wandb({f"{folder}/{name}": wandb.Image(plot_spectrograms2(y[i], y_hat[i], y_hat_cheat[i]))})
    else:
      self.log_wandb({f"{folder}/{name}": wandb.Image(plot_spectrograms(y[i], y_hat[i]))})
  plt.close("all")

@torch._dynamo.disable()
def log_attentions(self, names: list[str], attn: Tensor, folder="Attention"):
  for i, name in enumerate(names):
    self.log_wandb({f"{folder}/{name}": wandb.Image(plot_attention(attn[i]))})
  plt.close("all")

@torch._dynamo.disable()
def log_audios(self, names: list[str], sr: int, ys: list[Tensor], columns: list[str], folder="Audio"):
  step = self.batches_that_stepped()

  assert len(ys) == len(columns)

  columns = ["index"] + columns

  data = []
  for i, name in enumerate(names):
    audios = [wandb.Audio(y[i].cpu().to(torch.float32), sample_rate=sr) for y in ys]
    data.append([name] + audios)

  self.log_wandb({f"{folder}/{step:08d}": wandb.Table(data=data, columns=columns)})

def log_spksim0(self, y: Tensor, yv: Tensor, folder="Charts (SpkSim)"):
  y_spkemb = P.spkemb(y, 22050)
  y_v_spkemb = P.spkemb(yv, 22050)
  v_spksim = cosine_similarity(y_spkemb, y_v_spkemb)
  self.log(f"{folder}/valid_spksim", v_spksim.mean())

  return {
      "valid_spksim": v_spksim,
  }

def log_spksim1(self, y: Tensor, yv: Tensor, yc: Tensor, folder="Charts (SpkSim)"):
  y_spkemb = P.spkemb(y, 22050)
  y_v_spkemb = P.spkemb(yv, 22050)
  y_c_spkemb = P.spkemb(yc, 22050)
  v_spksim = cosine_similarity(y_spkemb, y_v_spkemb)
  c_spksim = cosine_similarity(y_spkemb, y_c_spkemb)
  self.log(f"{folder}/valid_spksim", v_spksim.mean())
  self.log(f"{folder}/cheat_spksim", c_spksim.mean())

  return {
      "valid_spksim": v_spksim,
      "cheat_spksim": c_spksim,
  }

class BaseLightningModule(L.LightningModule):
  def batches_that_stepped(self):
    # https://github.com/Lightning-AI/lightning/issues/13752
    # same as 'trainer/global_step' of wandb logger
    return self.trainer.fit_loop.epoch_loop._batches_that_stepped

  def log_wandb(self, item: dict):
    wandb_logger: Run = self.logger.experiment
    wandb_logger.log(item, step=self.batches_that_stepped())

def step_optimizer(self, opt, loss: Tensor, grad_clip=0.0, *, retain_graph=False):
  self.toggle_optimizer(opt)
  opt.zero_grad()
  self.manual_backward(loss, retain_graph=retain_graph)
  self.clip_gradients(opt, gradient_clip_val=grad_clip)
  opt.step()
  self.untoggle_optimizer(opt)

def step_optimizers(self, opts, loss: Tensor, grad_clip=0.0, *, retain_graph=False):
  for opt in opts:
    opt.zero_grad()
  self.manual_backward(loss, retain_graph=retain_graph)
  for opt in opts:
    self.clip_gradients(opt, gradient_clip_val=grad_clip)
    opt.step()

def fm_loss(fs1: list[Tensor], fs2: list[Tensor], fn: Callable[[Tensor, Tensor], Tensor]):
  """ Feature Matching Loss """
  loss = torch.as_tensor(0.0, device=fs1[0].device)
  for f1, f2 in zip(fs1, fs2):
    loss += fn(f1, f2)
  return loss

def shuffle_dim0(x: Tensor):
  return x[torch.randperm(x.shape[0])]

def rotate_dim0(x: Tensor):
  return torch.cat([x[1:], x[:1]], dim=0)

class LinearSchedule:
  def __init__(self, pos: list[tuple[int, float]]):
    self.pos = pos
    self.pos.sort(key=lambda x: x[0])

  def __call__(self, step: int):
    for i, (s, v) in enumerate(self.pos):
      if step < s:
        if i == 0: return v
        s0, v0 = self.pos[i - 1]
        if s == s0: return v
        return v0 + (v - v0) * (step - s0) / (s - s0)
    return self.pos[-1][1]

class BinarySchedule:
  def __init__(self, pos: list[tuple[int, bool]]):
    self.pos = pos
    self.pos.sort(key=lambda x: x[0])

  def __call__(self, step: int):
    for i, (s, v) in enumerate(self.pos):
      if step < s:
        if i == 0: return v
        s0, v0 = self.pos[i - 1]
        if s == s0: return v
        return v0
    return self.pos[-1][1]

@dataclass
class HifiMetadata:
  steps: int
  epoch: int

def load_hifigan_g(g: Path, generator: Optional[nn.Module], device: Device):
  g_data = torch.load(g, map_location=device)
  if generator: generator.load_state_dict(g_data["generator"])

def load_hifigan_do(do: Path, mpd: Optional[nn.Module], msd: Optional[nn.Module], optim_g: Optional[nn.Module], optim_d: Optional[nn.Module], device: Device):
  do_data = torch.load(do, map_location=device)
  if mpd: mpd.load_state_dict(do_data['mpd'])
  if msd: msd.load_state_dict(do_data['msd'])
  if optim_g: optim_g.load_state_dict(do_data['optim_g'])
  if optim_d: optim_d.load_state_dict(do_data['optim_d'])
  return HifiMetadata(steps=do_data['steps'], epoch=do_data['epoch'])

def log_lr(self, opt: Optimizer | LightningOptimizer, name: str, folder="Charts (General)"):
  if isinstance(opt, LightningOptimizer): opt = opt.optimizer
  lr = opt.param_groups[0]['lr']
  self.log(f"{folder}/{name}", lr)
  return lr

def club_ksp_net(xdim: int, ynum: int, hdim: int):
  return nn.Sequential(
      nn.Linear(xdim, hdim),
      nn.ReLU(),
      Transpose(1, 2),
      nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
      Transpose(1, 2),
      nn.ReLU(),
      nn.Linear(hdim, ynum),
  )

def spd_net():
  return BasicDiscriminator(
      dims=[64, 128, 512, 128, 64],
      kernels=[3, 5, 5, 5, 3],
      strides=[1, 2, 2, 1, 1],
      use_spectral_norm=False,  # spectral norm の weight / sigma で div by zero になってたので
  )

@dataclass
class HifiganFiledata:
  ckpts: tuple[str, str]
  config: AttrDict

def default_hifigan():
  return HifiganFiledata(
      ckpts=(DATA_DIR / "vocoder" / "g_02500000", DATA_DIR / "vocoder" / "do_02500000"),
      config=AttrDict({
          "resblock": "1",
          "learning_rate": 0.0002,
          "adam_b1": 0.8,
          "adam_b2": 0.99,
          "lr_decay": 0.999,
          "upsample_rates": [8, 8, 2, 2],
          "upsample_kernel_sizes": [16, 16, 4, 4],
          "upsample_initial_channel": 512,
          "resblock_kernel_sizes": [3, 7, 11],
          "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      }))

@dataclass
class LossSPD:
  real: Tensor
  fake: Tensor

def loss_spd_adcgan(speaker_d: nn.Module, mel_real: Tensor, mel_fake: Tensor, speaker: Tensor):
  real, _ = speaker_d(mel_real.detach())
  fake, _ = speaker_d(mel_fake.detach())
  loss_real = aux_loss(real, speaker * 2)
  loss_fake = aux_loss(fake, speaker * 2 + 1)

  return LossSPD(loss_real, loss_fake)

@dataclass
class LossSPD_G:
  fm: Tensor
  pos: Tensor
  neg: Tensor

def loss_spd_adcgan_g(speaker_d: nn.Module, mel: Tensor, mel_hat: Tensor, speaker: Tensor):
  with torch.no_grad():
    _, spd_f_real = speaker_d(mel)
  spd_c_fake, spd_f_fake = speaker_d(mel_hat)

  spd_g_fm = fm_loss(spd_f_real, spd_f_fake, F.l1_loss)
  spd_g_pos = aux_loss(spd_c_fake, speaker * 2)
  spd_g_neg = aux_loss(spd_c_fake, speaker * 2 + 1)

  return LossSPD_G(spd_g_fm, spd_g_pos, spd_g_neg)

@dataclass
class LossHifiganD:
  f: Tensor
  s: Tensor

def loss_hifigan_d(mpd: nn.Module, msd: nn.Module, y: Tensor, y_hat: Tensor):
  y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_hat.detach())
  loss_disc_f = VOC.discriminator_loss(y_df_hat_r, y_df_hat_g)

  y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_hat.detach())
  loss_disc_s = VOC.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

  return LossHifiganD(loss_disc_f, loss_disc_s)

@dataclass
class LossHifiganG:
  f: Tensor
  s: Tensor
  fm_f: Tensor
  fm_s: Tensor

def loss_hifigan_g(mpd: nn.Module, msd: nn.Module, y: Tensor, y_hat: Tensor):
  _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
  _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_hat)
  loss_fm_f = VOC.feature_loss(fmap_f_r, fmap_f_g)
  loss_fm_s = VOC.feature_loss(fmap_s_r, fmap_s_g)
  loss_gen_f, _ = VOC.generator_loss(y_df_hat_g)
  loss_gen_s, _ = VOC.generator_loss(y_ds_hat_g)

  return LossHifiganG(loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s)
