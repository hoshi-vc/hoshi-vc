# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

# Based on attempt08
# 次元下げれば、それだけでいいボトルネックになるかもしれないことに今気がついたので。

from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Optional

import lightning.pytorch as L
import torch
import torch._dynamo
import torch.functional as F
import torch.nn.functional as F
import torch.optim.lr_scheduler as S
import wandb
from lightning.pytorch import profilers
from torch import Tensor, nn
from torch.optim import AdamW

import engine.hifi_gan.models as VOC
from engine.attempt08_dataset import DataModule08, Entry08
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.club import CLUBSampleForCategorical
from engine.lib.fastspeech import FFNBlock, PosFFT
from engine.lib.layers import Buckets, Transpose
from engine.lib.utils import AttrDict, clamp, hide_warns
from engine.prepare import Preparation
from engine.utils import (DATA_DIR, BaseLightningModule, log_attentions, log_audios2, log_spectrograms, log_spksim0, log_spksim1, new_checkpoint_callback_wandb,
                          new_wandb_logger, setup_train_environment, shuffle_dim0, step_optimizer)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim // 2
    mel_dim = hdim // 2
    kv_dim = hdim // 2

    kdim = 8
    vdim = 64

    self.kv_dim = kv_dim

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.w2v2_embed = nn.Linear(256, w2v2_dim)
    self.encode_key = nn.Sequential(
        nn.Linear(energy_dim + pitch_dim + w2v2_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kdim),
        nn.Linear(kdim, kv_dim),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.mel_encode = nn.Linear(80, mel_dim)
    self.encode_value = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, vdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(vdim),
        nn.Linear(vdim, kv_dim),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.lookup = nn.MultiheadAttention(kv_dim, 16, dropout=0.2, batch_first=True)

    self.decode = nn.Sequential(
        nn.Linear(kv_dim + energy_dim + pitch_dim, hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        # PosFFT(hdim, layers=2, heads=2, hdim=256, kernels=(3, 3), dropout=0.2, posenc_len=2048),
        nn.Linear(hdim, 80),
    )

  def forward_energy(self, energy_i: Tensor):
    return self.energy_embed(self.energy_bins(energy_i[:, :, 0]))

  def forward_pitch(self, pitch_i: Tensor):
    return self.pitch_embed(pitch_i[:, :, 0])

  def forward_w2v2(self, w2v2: Tensor):
    return self.w2v2_embed(w2v2)

  def forward_mel(self, mel: Tensor):
    return self.mel_encode(mel)

  def forward_key(self, energy: Tensor, pitch: Tensor, w2v2: Tensor):
    return self.encode_key(torch.cat([energy, pitch, w2v2], dim=-1))

  def forward_value(self, energy: Tensor, pitch: Tensor, mel: Tensor):
    return self.encode_value(torch.cat([energy, mel], dim=-1))

  def forward(self, batch: Entry08):
    n_refs = len(batch.ref)
    n_batch = len(batch.src.energy)
    ref_len = batch.ref[0].energy.shape[1]

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_energy = torch.stack([o.energy for o in batch.ref]).flatten(0, 1)
    ref_pitch_i = torch.stack([o.pitch_i for o in batch.ref]).flatten(0, 1)
    ref_w2v2 = torch.stack([o.soft for o in batch.ref]).flatten(0, 1)
    ref_mel = torch.stack([o.mel for o in batch.ref]).flatten(0, 1)

    src_energy = batch.src.energy
    src_pitch_i = batch.src.pitch_i
    src_w2v2 = batch.src.soft
    src_mel = batch.src.mel

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_energy = self.forward_energy(ref_energy)
    ref_pitch = self.forward_pitch(ref_pitch_i)
    ref_w2v2 = self.forward_w2v2(ref_w2v2)
    ref_mel = self.forward_mel(ref_mel)
    ref_key = self.forward_key(ref_energy, ref_pitch, ref_w2v2)
    ref_value = self.forward_value(ref_energy, ref_pitch, ref_mel)

    src_energy = self.forward_energy(src_energy)
    src_pitch = self.forward_pitch(src_pitch_i)
    src_w2v2 = self.forward_w2v2(src_w2v2)
    src_mel = self.forward_mel(src_mel)
    src_key = self.forward_key(src_energy, src_pitch, src_w2v2)
    # src_value = self.forward_value(src_energy, src_pitch, src_mel)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim) -> (n_batch, n_refs*seq_len, feat_dim)
    ref_energy = ref_energy.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_pitch = ref_pitch.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_key = ref_key.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_value = ref_value.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_pitch_i = ref_pitch_i.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)

    assert ref_key.shape[1] == ref_len * n_refs, f"ref_key.shape={ref_key.shape}, ref_len={ref_len}, n_refs={n_refs}"

    tgt_value, attn = self.lookup(src_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    return tgt_mel, (ref_key, ref_value, ref_pitch_i, attn), []

class VCModule(BaseLightningModule):
  def __init__(self,
               hdim: int,
               lr: float,
               lr_club: float,
               warmup_steps: int,
               total_steps: int,
               grad_clip: float,
               e2e_frames: int,
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.vc_model = VCModel(hdim=hdim)
    self.vocoder = VOC.Generator(hifi_gan)

    self.club_val = CLUBSampleForCategorical(xdim=self.vc_model.kv_dim, ynum=360, hdim=hdim, fast_sampling=True)
    self.club_key = CLUBSampleForCategorical(xdim=self.vc_model.kv_dim, ynum=360, hdim=hdim, fast_sampling=True)
    self.club_ksp = CLUBSampleForCategorical(xdim=self.vc_model.kv_dim, ynum=len(P.dataset.speaker_ids), hdim=hdim)

    self.batch_rand = Random(94324203)
    self.clip_rand = Random(76482573)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.lr = lr
    self.lr_club = lr_club
    self.grad_clip = grad_clip
    self.e2e_frames = e2e_frames
    self.hifi_gan = hifi_gan
    self.hifi_gan_ckpt = hifi_gan_ckpt

    self.val_outputs = []

    self.save_hyperparameters()

    # NOTE: activates manual optimization.
    # https://lightning.ai/docs/pytorch/stable/common/optimization.html
    # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
    self.automatic_optimization = False

  def _process_batch(
      self,
      batch: Entry08,
      step: int,
      e2e: bool,
      e2e_frames: Optional[int],
      train=False,
      log: Optional[str] = None,
  ):
    opt_model, opt_club = self.optimizers()

    # NOTE: step 0 ではバグ確認のためすべてのロスを使って backward する。
    debug = step == 0 and train

    # inputs

    mel = batch.src.mel
    y = batch.src.audio

    # vc model

    mel_hat, (ref_key, ref_value, ref_pitch_i, vc_attn), _ = self.vc_model(batch)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"Charts (Main)/{log}_reconst", vc_reconst)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = ref_pitch_i[:, :, 0]
    club_sp_y = batch.src.speaker.repeat(1, ref_key.shape[1])
    club_sp_n = shuffle_dim0(batch.src.speaker).repeat(1, ref_key.shape[1])

    mi_val = self.club_val(club_x_val, club_y)
    mi_key = self.club_key(club_x_key, club_y)
    mi_ksp = self.club_ksp(club_x_key, club_sp_y, club_sp_n)

    club_val = self.club_val.learning_loss(club_x_val.detach(), club_y.detach())
    club_key = self.club_key.learning_loss(club_x_key.detach(), club_y.detach())
    club_ksp = self.club_ksp.learning_loss(club_x_key.detach(), club_sp_y.detach())

    total_club = club_val + club_key + club_ksp

    if train:
      step_optimizer(self, opt_club, total_club, self.grad_clip, retain_graph=True)

    if log:
      self.log(f"Charts (Main)/{log}_mi_val", mi_val)
      self.log(f"Charts (Main)/{log}_mi_key", mi_key)
      self.log(f"Charts (Main)/{log}_mi_ksp", mi_ksp)
      self.log(f"Charts (CLUB)/{log}_club_val", club_val)
      self.log(f"Charts (CLUB)/{log}_club_key", club_key)
      self.log(f"Charts (CLUB)/{log}_club_ksp", club_ksp)

    # e2e

    y_g_hat = None
    y_g_hat_mel = None
    if debug or e2e:

      # vocoder

      if e2e_frames is None:
        e2e_start = 0
        e2e_end = mel.shape[1]
      else:
        e2e_start = self.clip_rand.randint(0, mel.shape[1] - e2e_frames)
        e2e_end = e2e_start + e2e_frames

      e2e_y = y[:, e2e_start * 256:e2e_end * 256].unsqueeze(1)
      e2e_mel = mel[:, e2e_start:e2e_end].transpose(1, 2)
      e2e_mel_hat = mel_hat[:, e2e_start:e2e_end].transpose(1, 2)

      y_g_hat = self.vocoder(e2e_mel_hat)
      y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), sampling_rate=22050, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=8000, fast=True)

      loss_mel = F.l1_loss(e2e_mel, y_g_hat_mel)

      if log:
        self.log(f"Charts (Main)/{log}_e2e_reconst", loss_mel)

      y_g_hat = y_g_hat.squeeze(1)
      y_g_hat_mel = y_g_hat_mel.transpose(1, 2)

    # generator

    total_model = vc_reconst * 150.0
    total_model += mi_val
    total_model += mi_ksp

    if train:
      step_optimizer(self, opt_model, total_model, self.grad_clip)

    if log:
      self.log(f"Charts (Main)/{log}_loss", total_model)

    return mel_hat, y_g_hat, y_g_hat_mel, vc_attn, []

  def vocoder_forward(self, mel: Tensor):
    return self.vocoder(mel.transpose(1, 2)).squeeze(1)

  def training_step(self, batch: Entry08, batch_idx: int):
    step = self.batches_that_stepped()

    opt_model, opt_club = self.optimizers()
    sch_model, sch_club = self.lr_schedulers()
    self.log("Charts (General)/lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_club", opt_club.optimizer.param_groups[0]["lr"])

    self._process_batch(batch, step, e2e=False, e2e_frames=self.e2e_frames, train=True, log="train")

    with hide_warns():
      sch_club.step()
      sch_model.step()

  def validation_step(self, batch: Entry08, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch.src.mel
    y = batch.src.audio

    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics
    _, y_v, _, _, _ = self._process_batch(batch, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

    if complex_metrics:
      spksim = log_spksim0(self, P, y, y_v)
      self.log("valid_spksim", spksim["valid_spksim"].mean())
      self.val_outputs.append(spksim)

    if batch_idx == 0:
      mel_v, y_v, ymel_v, attn_v, _ = self._process_batch(batch, step=step, e2e=True, e2e_frames=None)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_v, "Attention")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_audios2(self, P, names, 22050, y, y_v)

  def on_validation_epoch_end(self):
    if len(self.val_outputs) > 0:
      v_spksim = torch.cat([x["valid_spksim"] for x in self.val_outputs]).cpu().numpy()
      self.log_wandb({"Charts (SpkSim)/valid_hist": wandb.Histogram(v_spksim)})

    self.val_outputs.clear()

  def on_validation_end(self):
    P.release_spkemb()
    P.release_mosnet()
    torch.cuda.empty_cache()

  def configure_optimizers(self):
    opt_model = AdamW(self.vc_model.parameters(), lr=self.lr)
    opt_club = AdamW([*self.club_val.parameters(), *self.club_key.parameters(), *self.club_ksp.parameters()], lr=self.lr_club)

    # TODO: こんな方法とタイミングで事前学習済みモデルを読み込むのが最善とは思えない
    if self.hifi_gan_ckpt is not None:
      if self.trainer.ckpt_path is not None:
        print("Skipped loading pretrained HiFi-GAN weights because the training was resumed from a checkpoint.")
      else:
        print("Loading pretrained HiFi-GAN weights...")
        g, do = self.hifi_gan_ckpt
        g_data = torch.load(g, map_location=self.device)
        self.vocoder.load_state_dict(g_data["generator"])

    # TODO: resume 時に step に渡される値がリセットされていそう
    sch_model = S.ChainedScheduler([
        get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps),
    ])
    sch_club = S.LambdaLR(opt_club, lambda step: 1.0)

    return [opt_model, opt_club], [sch_model, sch_club]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem.split("_")[0].split(" ")[0]
  assert PROJECT.startswith("attempt08")
  PROJECT = "attempt08"

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = DataModule08(P, frames=256, frames_ref=32, n_refs=64, ref_max_kth=64, batch_size=8, n_batches=2000, n_batches_val=200)

  total_steps = 40000
  total_actual_steps = 20000
  complex_metrics_batches = 50  # see: validation_step

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_club=1e-4,
      warmup_steps=500,
      total_steps=total_steps,
      e2e_frames=64,
      grad_clip=1.0,
      hifi_gan=AttrDict({
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
      }),
      hifi_gan_ckpt=(g_ckpt, do_ckpt),
  )

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_epochs=int(ceil(total_actual_steps / datamodule.n_batches)),
      logger=wandb_logger,
      callbacks=[
          new_checkpoint_callback_wandb(
              PROJECT,
              wandb_logger,
              filename="{step:08d}-{valid_spksim:.4f}-{valid_vc_spksim:.4f}",
              monitor="valid_spksim",
              mode="max",
          ),
      ],
      accelerator="gpu",
      precision="16-mixed",
      benchmark=True,
      # # not benchmark and deterministic にしたら spkemb の計算の conv1d が 85％ の validation 時間を占めた
      # detect_anomaly=True,
      # deterministic=True,
      # profiler=profilers.PyTorchProfiler(
      #     DATA_DIR / "profiler",
      #     schedule=torch.profiler.schedule(wait=0, warmup=100, active=10, repeat=1),
      #     on_trace_ready=torch.profiler.tensorboard_trace_handler(DATA_DIR / "profiler"),
      #     with_stack=False,
      # ),
  )

  # train the model
  trainer.fit(
      model,
      datamodule=datamodule,
      ckpt_path=None,
  )

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()

  print(f"Saved model to {Path(trainer.checkpoint_callback.last_model_path).relative_to(DATA_DIR)}")
