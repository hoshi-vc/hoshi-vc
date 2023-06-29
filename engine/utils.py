# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from os import path
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import lightning.pytorch as L
import matplotlib
import torch
import torch._dynamo
import wandb
from lightning import seed_everything
from lightning.pytorch import callbacks as C
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor, cosine_similarity
from wandb.wandb_run import Run

from engine.lib.utils_ui import (plot_attention, plot_spectrograms, plot_spectrograms2)

if TYPE_CHECKING: from engine.prepare import Preparation

DATA_DIR = Path(__file__).parent.parent / "data"

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
def log_audios(self, P: "Preparation", names: list[str], y: Tensor, y_hat: Tensor, y_hat_cheat: Optional[Tensor] = None, folder="Audio"):
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

  self.log_wandb({f"{folder}/{step:08d}": wandb.Table(data=data, columns=columns)})

@torch._dynamo.disable()
def log_audios2(self,
                P: "Preparation",
                names: list[str],
                sr: int,
                y: Tensor,
                y_hat: Tensor,
                y_hat_cheat: Optional[Tensor] = None,
                cols: Optional[list[str]] = None,
                folder="Audio"):
  step = self.batches_that_stepped()

  columns = ["index", "original", "reconstructed"]
  if y_hat_cheat is not None:
    columns.append("reconstructed_cheat")
  if cols: columns = cols

  data = []
  for i, name in enumerate(names):
    if y_hat_cheat is None:
      data.append([
          name,
          wandb.Audio(y[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat[i].cpu().to(torch.float32), sample_rate=sr),
      ])
    else:
      data.append([
          name,
          wandb.Audio(y[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat_cheat[i].cpu().to(torch.float32), sample_rate=sr),
      ])

  self.log_wandb({f"{folder}/{step:08d}": wandb.Table(data=data, columns=columns)})

@torch._dynamo.disable()
def log_audios3(self,
                P: "Preparation",
                names: list[str],
                sr: int,
                y: Tensor,
                y_aug: Tensor,
                y_hat: Tensor,
                y_hat_cheat: Optional[Tensor] = None,
                folder="Audio"):
  step = self.batches_that_stepped()

  columns = ["index", "original", "augmented", "reconstructed"]
  if y_hat_cheat is not None:
    columns.append("reconstructed_cheat")

  data = []
  for i, name in enumerate(names):
    if y_hat_cheat is None:
      data.append([
          name,
          wandb.Audio(y[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_aug[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat[i].cpu().to(torch.float32), sample_rate=sr),
      ])
    else:
      data.append([
          name,
          wandb.Audio(y[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_aug[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat[i].cpu().to(torch.float32), sample_rate=sr),
          wandb.Audio(y_hat_cheat[i].cpu().to(torch.float32), sample_rate=sr),
      ])

  self.log_wandb({f"{folder}/{step:08d}": wandb.Table(data=data, columns=columns)})

# TODO: この関数の汎用性が低すぎて、良くない処理のくくりだしに思える
def log_spksim(self, P: "Preparation", y: Tensor, yv: Tensor, yc: Tensor, yv_rot: Tensor, yc_rot: Tensor, folder="Charts (SpkSim)"):
  """
  Args:
      y: ground truth audio
      yv: converted audio without cheat
      yc: converted audio with cheat
      yv_rot: converted audio without cheat (rotated source speakers)
      yc_rot: converted audio with cheat (rotated source speakers)
  """

  y_spkemb = P.spkemb(y, 22050)
  y_v_spkemb = P.spkemb(yv, 22050)
  y_c_spkemb = P.spkemb(yc, 22050)
  v_spksim = cosine_similarity(y_spkemb, y_v_spkemb)
  c_spksim = cosine_similarity(y_spkemb, y_c_spkemb)
  self.log(f"{folder}/valid_spksim", v_spksim.mean())
  self.log(f"{folder}/vcheat_spksim", c_spksim.mean())

  # speaker-conversion
  yv2_spkemb = P.spkemb(yv_rot, 22050)
  yc2_spkemb = P.spkemb(yc_rot, 22050)
  v2_spksim = cosine_similarity(y_spkemb, yv2_spkemb)
  c2_spksim = cosine_similarity(y_spkemb, yc2_spkemb)
  v2_spksim_leak = cosine_similarity(rotate_dim0(y_spkemb), yv2_spkemb)
  c2_spksim_leak = cosine_similarity(rotate_dim0(y_spkemb), yc2_spkemb)
  v2_spksim_irrelevent = cosine_similarity(y_spkemb, rotate_dim0(yv2_spkemb))
  c2_spksim_irrelevent = cosine_similarity(y_spkemb, rotate_dim0(yc2_spkemb))
  self.log(f"{folder}/valid_spksim_vc", v2_spksim.mean())
  self.log(f"{folder}/vcheat_spksim_vc", c2_spksim.mean())
  self.log(f"{folder}/valid_spksim_leak", v2_spksim_leak.mean())
  self.log(f"{folder}/vcheat_spksim_leak", c2_spksim_leak.mean())
  self.log(f"{folder}/valid_spksim_base", v2_spksim_irrelevent.mean())
  self.log(f"{folder}/vcheat_spksim_base", c2_spksim_irrelevent.mean())

  return {
      "valid_spksim": v_spksim,
      "vcheat_spksim": c_spksim,
      "valid_spksim_vc": v2_spksim,
      "vcheat_spksim_vc": c2_spksim,
  }

# TODO: この関数の汎用性が低すぎて、良くない処理のくくりだしに思える
def log_spksim1(self, P: "Preparation", y: Tensor, yv: Tensor, yc: Tensor, folder="Charts (SpkSim)"):
  y_spkemb = P.spkemb(y, 22050)
  y_v_spkemb = P.spkemb(yv, 22050)
  y_c_spkemb = P.spkemb(yc, 22050)
  v_spksim = cosine_similarity(y_spkemb, y_v_spkemb)
  c_spksim = cosine_similarity(y_spkemb, y_c_spkemb)
  self.log(f"{folder}/valid_spksim", v_spksim.mean())
  self.log(f"{folder}/vcheat_spksim", c_spksim.mean())

  return {
      "valid_spksim": v_spksim,
      "vcheat_spksim": c_spksim,
  }

# TODO: この関数の汎用性が低すぎて、良くない処理のくくりだしに思える
def log_spksim0(self, P: "Preparation", y: Tensor, yv: Tensor, folder="Charts (SpkSim)"):
  y_spkemb = P.spkemb(y, 22050)
  y_v_spkemb = P.spkemb(yv, 22050)
  v_spksim = cosine_similarity(y_spkemb, y_v_spkemb)
  self.log(f"{folder}/valid_spksim", v_spksim.mean())

  return {
      "valid_spksim": v_spksim,
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
