# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

import json
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, NamedTuple, Optional

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
from wandb.wandb_run import Run

import engine.hifi_gan.models as VOC
from engine.dataset_feats import IntraDomainDataModule4, IntraDomainEntry4
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.layers import Buckets, CLUBSampleForCategorical, Transpose
from engine.lib.utils import DATA_DIR, AttrDict, clamp
from engine.prepare import Preparation
from engine.utils import (log_audios2, log_spectrograms, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment)

class VCInputs(NamedTuple):
  src_energy: Tensor  #    (batch, src_len, 1)
  src_w2v2: Tensor  #      (batch, src_len, 768)
  src_pitch_i: Tensor  #   (batch, src_len, 1+)
  ref_energy: Tensor  #    (batch, ref_len, 1)
  ref_w2v2: Tensor  #      (batch, ref_len, 768)
  ref_pitch_i: Tensor  #   (batch, ref_len, 1+)
  ref_mel: Tensor  #       (batch, ref_len, 80)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim
    mel_dim = hdim // 2

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.w2v2_embed = nn.Linear(768, w2v2_dim)
    self.encode_key = nn.Sequential(
        # input: (batch, src_len, hdim)
        Transpose(1, 2),
        nn.Conv1d(energy_dim + pitch_dim + w2v2_dim, hdim, kernel_size=1),
        Transpose(1, 2),
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
        Transpose(1, 2),
        nn.Conv1d(energy_dim + pitch_dim + mel_dim, hdim, kernel_size=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
    )

    self.lookup = nn.MultiheadAttention(hdim, 4, batch_first=True)

    self.decode = nn.Sequential(
        # input: (batch, src_len, hdim + energy_dim + pitch_dim)
        Transpose(1, 2),
        nn.Conv1d(hdim + energy_dim + pitch_dim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, 80, kernel_size=1),
        Transpose(1, 2),
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
    return self.encode_value(torch.cat([energy, pitch, mel], dim=-1))

  def forward(self, o: VCInputs):

    # key: 似たような発音ほど近い表現になってほしい
    #      話者性が多少残ってても lookup 後の value への影響は間接的なので多分問題ない

    # value: 可能な限り多くの発音情報や話者性を含む表現になってほしい
    #        ただし、ピッチや音量によらない表現になってほしい
    #        （デコード時にピッチと音量を調節するので、そこと情報の衝突が起きないでほしい）

    # shape: (batch, src_len, hdim)
    src_energy = self.forward_energy(o.src_energy)
    src_pitch = self.forward_pitch(o.src_pitch_i)
    src_w2v2 = self.forward_w2v2(o.src_w2v2)
    src_key = self.forward_key(src_energy, src_pitch, src_w2v2)

    ref_energy = self.forward_energy(o.ref_energy)
    ref_pitch = self.forward_pitch(o.ref_pitch_i)
    ref_w2v2 = self.forward_w2v2(o.ref_w2v2)
    ref_key = self.forward_key(ref_energy, ref_pitch, ref_w2v2)

    ref_mel = self.forward_mel(o.ref_mel)
    ref_value = self.forward_value(ref_energy, ref_pitch, ref_mel)

    tgt_value, _ = self.lookup(src_key, ref_key, ref_value)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    return tgt_mel, (ref_key, ref_value)

class VCModule(L.LightningModule):
  def __init__(self,
               hdim: int,
               lr: float,
               lr_club: float,
               warmup_steps: int,
               total_steps: int,
               milestones: tuple[int, int, int, int],
               grad_clip: float,
               e2e_milestones: int,
               e2e_frames: int,
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.vc_model = VCModel(hdim=hdim)
    self.vocoder = VOC.Generator(hifi_gan)
    self.vocoder_mpd = VOC.MultiPeriodDiscriminator()
    self.vocoder_msd = VOC.MultiScaleDiscriminator()

    self.club_val = CLUBSampleForCategorical(xdim=hdim, ynum=360, hdim=hdim, fast_sampling=True)
    self.club_key = CLUBSampleForCategorical(xdim=hdim, ynum=360, hdim=hdim, fast_sampling=True)

    self.batch_rand = Random(94324203)
    self.clip_rand = Random(76482573)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.milestones = milestones
    self.lr = lr
    self.lr_club = lr_club
    self.grad_clip = grad_clip
    self.e2e_milestones = e2e_milestones
    self.e2e_frames = e2e_frames
    self.hifi_gan = hifi_gan
    self.hifi_gan_ckpt = hifi_gan_ckpt

    self.save_hyperparameters()

    # NOTE: activates manual optimization.
    # https://lightning.ai/docs/pytorch/stable/common/optimization.html
    # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
    self.automatic_optimization = False

  def batches_that_stepped(self):
    # https://github.com/Lightning-AI/lightning/issues/13752
    # same as 'trainer/global_step' of wandb logger
    return self.trainer.fit_loop.epoch_loop._batches_that_stepped

  def log_wandb(self, item: dict):
    wandb_logger: Run = self.logger.experiment
    wandb_logger.log(item, step=self.batches_that_stepped())

  def _make_vc_inputs(self, batch: IntraDomainEntry4, self_ratio: float, ref_ratio: float):
    src, refs = batch[0], batch[1:]

    base_len = src.mel.shape[1]
    n_refs = len(refs)
    s_len = int(base_len * self_ratio)
    r_len = int(base_len * n_refs * ref_ratio)
    rs = self.batch_rand.randint(0, base_len * n_refs - r_len)
    re = rs + r_len

    # yapf: disable
    ref_energy= torch.cat([ src.energy[:, :s_len, :], torch.cat([ o.energy for o in refs], dim=1)[:, rs:re, :]], dim=1)
    ref_w2v2=   torch.cat([   src.w2v2[:, :s_len, :], torch.cat([   o.w2v2 for o in refs], dim=1)[:, rs:re, :]], dim=1)
    ref_pitch_i=torch.cat([src.pitch_i[:, :s_len, :], torch.cat([o.pitch_i for o in refs], dim=1)[:, rs:re, :]], dim=1)
    ref_mel=    torch.cat([    src.mel[:, :s_len, :], torch.cat([    o.mel for o in refs], dim=1)[:, rs:re, :]], dim=1)
    # yapf: enable

    return VCInputs(
        src_energy=src.energy,
        src_w2v2=src.w2v2,
        src_pitch_i=src.pitch_i,
        ref_energy=ref_energy,
        ref_w2v2=ref_w2v2,
        ref_pitch_i=ref_pitch_i,
        ref_mel=ref_mel,
    )

  def _process_batch(self,
                     batch: IntraDomainEntry4,
                     self_ratio: float,
                     ref_ratio: float,
                     step: int,
                     e2e: bool,
                     e2e_frames: Optional[int],
                     train=False,
                     log: Optional[str] = None):
    h = self.hifi_gan
    opt_model, opt_club, opt_d = self.optimizers()

    # TODO: あとできれいに書き直したい...

    # NOTE: step 0 ではバグ確認のためすべてのロスを使って backward する。
    debug = step == 0 and train
    if debug: print('process_batch: debug mode')

    # inputs
    mel = batch[0].mel
    y = batch[0].audio
    vc_inputs = self._make_vc_inputs(batch, self_ratio, ref_ratio)

    # vc model

    mel_hat: Tensor
    mel_hat, (ref_key, ref_value) = self.vc_model(vc_inputs)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"{log}_reconst", vc_reconst)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = vc_inputs.ref_pitch_i[:, :, 0]

    mi_val = self.club_val(club_x_val, club_y)
    mi_key = self.club_key(club_x_key, club_y)

    club_val = self.club_val.learning_loss(club_x_val, club_y)
    club_key = self.club_key.learning_loss(club_x_key, club_y)

    total_club = club_val + club_key

    if train:
      opt_club.zero_grad()
      self.toggle_optimizer(opt_club)
      self.manual_backward(total_club, retain_graph=True)
      self.untoggle_optimizer(opt_club)
      self.clip_gradients(opt_club, gradient_clip_val=self.grad_clip)
      opt_club.step()

    if log:
      self.log(f"{log}_mi_val", mi_val)
      self.log(f"{log}_mi_key", mi_key)
      self.log(f"{log}_loss_club_val", club_val)
      self.log(f"{log}_loss_club_key", club_key)

    # discriminator, e2e

    total_model = vc_reconst / 0.4
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
      y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss, fast=True)

      # discriminator

      y_df_hat_r, y_df_hat_g, _, _ = self.vocoder_mpd(e2e_y, y_g_hat.detach())
      loss_disc_f = VOC.discriminator_loss(y_df_hat_r, y_df_hat_g)

      y_ds_hat_r, y_ds_hat_g, _, _ = self.vocoder_msd(e2e_y, y_g_hat.detach())
      loss_disc_s = VOC.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

      total_disc = loss_disc_s + loss_disc_f

      if train:
        opt_d.zero_grad()
        self.toggle_optimizer(opt_d)
        self.manual_backward(total_disc, retain_graph=True)
        self.untoggle_optimizer(opt_d)
        self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip)
        opt_d.step()

      if log:
        self.log(f"{log}_loss_disc", total_disc)
        self.log(f"{log}_e2e_disc_f", loss_disc_f)
        self.log(f"{log}_e2e_disc_s", loss_disc_s)

      # e2e generator loss

      loss_mel = F.l1_loss(e2e_mel, y_g_hat_mel)

      y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.vocoder_mpd(e2e_y, y_g_hat)
      y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.vocoder_msd(e2e_y, y_g_hat)
      loss_fm_f = VOC.feature_loss(fmap_f_r, fmap_f_g)
      loss_fm_s = VOC.feature_loss(fmap_s_r, fmap_s_g)
      loss_gen_f, losses_gen_f = VOC.generator_loss(y_df_hat_g)
      loss_gen_s, losses_gen_s = VOC.generator_loss(y_ds_hat_g)

      e2e_model_loss = (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45.0) / 60.0

      if debug: total_model = total_model * 0.2 + e2e_model_loss * 0.8
      else:
        if step < self.e2e_milestones[1]: pass
        elif step < self.e2e_milestones[2]: total_model = total_model * 0.2 + e2e_model_loss * 0.8
        else: total_model = e2e_model_loss

      if log:
        self.log(f"{log}_e2e_gen_f", loss_gen_f)
        self.log(f"{log}_e2e_gen_s", loss_gen_s)
        self.log(f"{log}_e2e_fm_f", loss_fm_f)
        self.log(f"{log}_e2e_fm_s", loss_fm_s)
        self.log(f"{log}_e2e_mel", loss_mel)

      y_g_hat = y_g_hat.squeeze(1)
      y_g_hat_mel = y_g_hat_mel.transpose(1, 2)

    # generator

    if train:
      opt_model.zero_grad()
      self.toggle_optimizer(opt_model)
      self.manual_backward(total_model)
      self.untoggle_optimizer(opt_model)
      self.clip_gradients(opt_model, gradient_clip_val=self.grad_clip)
      opt_model.step()

    if log:
      self.log(f"{log}_loss", total_model)

    return mel_hat, y_g_hat, y_g_hat_mel

  def training_step(self, batch: IntraDomainEntry4, batch_idx: int):
    step = self.batches_that_stepped()
    milestones = self.milestones
    progress1 = (step - milestones[0]) / (milestones[1] - milestones[0])
    progress2 = (step - milestones[2]) / (milestones[3] - milestones[2])
    ref_ratio = clamp(progress1, 0.0, 1.0)
    self_ratio = 1 - clamp(progress2, 0.0, 1.0)
    self.log("self_ratio", self_ratio)
    self.log("ref_ratio", ref_ratio)

    opt_model, opt_club, opt_d = self.optimizers()
    sch_model, sch_club, sch_d = self.lr_schedulers()
    self.log("lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("lr_club", opt_club.optimizer.param_groups[0]["lr"])
    self.log("lr_d", opt_d.optimizer.param_groups[0]["lr"])

    self._process_batch(batch, self_ratio, ref_ratio, step, e2e=step >= self.e2e_milestones[0], e2e_frames=self.e2e_frames, train=True, log="train")

    sch_club.step()
    sch_model.step()
    if step % 25000 == 0: sch_d.step()  # 25000: do_02500000 の steps/epoch

  def validation_step(self, batch: IntraDomainEntry4, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch[0].mel
    y = batch[0].audio

    # vcheat: validation with cheating
    self._process_batch(batch, self_ratio=1.0, ref_ratio=1.0, step=step, e2e=True, e2e_frames=self.e2e_frames, log="vcheat")
    self._process_batch(batch, self_ratio=0.0, ref_ratio=1.0, step=step, e2e=True, e2e_frames=self.e2e_frames, log="valid")

    if batch_idx == 0:
      mel_hat_cheat, y_hat_cheat, y_hat_mel_cheat = self._process_batch(batch, self_ratio=1.0, ref_ratio=1.0, step=step, e2e=True, e2e_frames=None)
      mel_hat, y_hat, y_hat_mel = self._process_batch(batch, self_ratio=0.0, ref_ratio=1.0, step=step, e2e=True, e2e_frames=None)
      names = [f"{i:02d}" for i in range(8)]
      log_spectrograms(self, names, mel, mel_hat_cheat, y_hat_mel_cheat, "Spectrogram (Cheat)")
      log_spectrograms(self, names, mel, mel_hat, y_hat_mel, "Spectrogram")
      log_audios2(self, P, names, 22050, y, y_hat, y_hat_cheat)

  def configure_optimizers(self):
    h = self.hifi_gan

    # TODO: 今はとりあえず vocoder の重みの更新はしないで様子を見る

    opt_model = AdamW(self.vc_model.parameters(), lr=self.lr)
    opt_club = AdamW([*self.club_val.parameters(), *self.club_key.parameters()], lr=self.lr_club)
    opt_d = AdamW([*self.vocoder_msd.parameters(), *self.vocoder_mpd.parameters()], h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # TODO: こんな方法とタイミングで事前学習済みモデルを読み込むのが最善とは思えない
    last_epoch = -1
    if self.hifi_gan_ckpt is not None:
      g, do = self.hifi_gan_ckpt
      g_data = torch.load(g, map_location=self.device)
      do_data = torch.load(do, map_location=self.device)
      last_epoch = do_data['epoch']
      self.vocoder.load_state_dict(g_data["generator"])
      self.vocoder_mpd.load_state_dict(do_data['mpd'])
      self.vocoder_msd.load_state_dict(do_data['msd'])
      opt_d.load_state_dict(do_data['optim_d'])

      # 1 epoch = 500 steps として、速度が安定してきた epoch 3 後半あたりの処理速度を雑に測った
      # with compile:    3.52 it/s
      # without compile: 3.52 it/s
      # という結果だったので、モデルだけコンパイルしても学習には大差ないのかもしれない

      # self.vc_model = torch.compile(self.vc_model)
      # self.vocoder = torch.compile(self.vocoder)
      # self.vocoder_mpd = torch.compile(self.vocoder_mpd)
      # self.vocoder_msd = torch.compile(self.vocoder_msd)

      # これは逆に遅くなる & プロファイラで見る限り途中から fused じゃなくなる
      # self.vocoder = torch.jit.trace(self.vocoder, torch.randn(1, 80, 256, device=self.device))

    sch_model = get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps)
    sch_club = S.MultiplicativeLR(opt_club, lambda step: 1.0)
    sch_d = S.ExponentialLR(opt_d, gamma=h.lr_decay, last_epoch=last_epoch)

    return [opt_model, opt_club, opt_d], [sch_model, sch_club, sch_d]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModule4(P, frames=256, n_samples=7, batch_size=8, n_batches=1000, n_batches_val=200)

  total_steps = 30000

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_club=1e-3,
      warmup_steps=500,
      total_steps=total_steps,
      milestones=(4000, 6001, 10000, 20000),
      e2e_milestones=(1000, 2000, 2000),
      e2e_frames=64,  # same as JETS https://arxiv.org/pdf/2203.16852.pdf
      grad_clip=0.5,
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
          "num_mels": 80,
          "n_fft": 1024,
          "hop_size": 256,
          "win_size": 1024,
          "sampling_rate": 22050,
          "fmin": 0,
          "fmax_for_loss": None,
      }),
      hifi_gan_ckpt=(g_ckpt, do_ckpt),
  )

  # NOTE: pytorch 2.0 では torch._functorch.partitioners._tensor_nbytes が complex64 をサポートしないのでエラーが出る
  #       https://pytorch.org/functorch/2.0/_modules/torch/_functorch/partitioners.html

  # https://pytorch.org/get-started/pytorch-2.0/
  # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  # https://pytorch.org/docs/stable/dynamo/faq.html
  # https://pytorch.org/docs/stable/dynamo/troubleshooting.html
  # https://github.com/pytorch/pytorch/blob/f7608998649d96ace4d2b56dc392ad36177791e2/docs/source/compile/fine_grained_apis.rst
  # https://grep.app/search?q=torch._dynamo

  # use the latest lagic : なぜかエラーするので、モデル群だけをコンパイルすることにした : see configure_optimizers
  # model = torch.compile(model)  # -> RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation （なんで...）

  # torch._dynamo.config.verbose = True
  # from torch._dynamo.utils import CompileProfiler
  # prof = CompileProfiler()

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_epochs=int(ceil(total_steps / datamodule.n_batches)),
      logger=wandb_logger,
      callbacks=[
          new_checkpoint_callback_wandb(PROJECT, wandb_logger),
      ],
      accelerator="gpu",
      precision="16-mixed",
      benchmark=True,
      # deterministic=True,
      # detect_anomaly=True,
      # profiler=profilers.PyTorchProfiler(
      #     DATA_DIR / "profiler",
      #     schedule=torch.profiler.schedule(wait=20, warmup=10, active=6, repeat=1),
      #     on_trace_ready=torch.profiler.tensorboard_trace_handler(DATA_DIR / "profiler"),
      #     with_stack=True,
      # ),
  )

  # train the model
  trainer.fit(model, datamodule=datamodule)

  # print(prof.report())
  # print(torch._dynamo.utils.compile_times())

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()
