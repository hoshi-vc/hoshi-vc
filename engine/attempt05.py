# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

import json
from pathlib import Path
from random import Random
from typing import Any, NamedTuple, Optional

import lightning.pytorch as L
import torch
import torch.functional as F
import torch.nn.functional as F
import torch.optim.lr_scheduler as S
import wandb
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
from engine.utils import (log_audios, log_audios2, log_spectrograms, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment)

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

    self.lookup = nn.MultiheadAttention(hdim, 4, dropout=0.1, batch_first=True)

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
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.model = VCModel(hdim=hdim)
    self.vocoder = VOC.Generator(hifi_gan)
    self.vocoder_mpd = VOC.MultiPeriodDiscriminator()
    self.vocoder_msd = VOC.MultiScaleDiscriminator()

    self.club_val = CLUBSampleForCategorical(xdim=hdim, ynum=360, hdim=hdim, fast_sampling=True)
    self.club_key = CLUBSampleForCategorical(xdim=hdim, ynum=360, hdim=hdim, fast_sampling=True)

    self.batch_rand = Random(94324203)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.milestones = milestones
    self.lr = lr
    self.lr_club = lr_club
    self.grad_clip = grad_clip
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

  def _process_batch(self, batch: IntraDomainEntry4, self_ratio: float, ref_ratio: float, step: int, log: Optional[str] = None):
    h = self.hifi_gan

    # inputs
    mel = batch[0].mel
    y = batch[0].audio
    vc_inputs = self._make_vc_inputs(batch, self_ratio, ref_ratio)

    # vc model
    mel_hat: Tensor
    mel_hat, (ref_key, ref_value) = self.model(vc_inputs)

    vc_reconst = F.l1_loss(mel_hat, mel)

    # vocoder
    y_g_hat = self.vocoder(mel_hat.transpose(1, 2))
    y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

    # MPD
    y_df_hat_r, y_df_hat_g, _, _ = self.vocoder_mpd(y.unsqueeze(1), y_g_hat.detach())
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = VOC.discriminator_loss(y_df_hat_r, y_df_hat_g)

    # MSD
    y_ds_hat_r, y_ds_hat_g, _, _ = self.vocoder_msd(y.unsqueeze(1), y_g_hat.detach())
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = VOC.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

    loss_disc_all = loss_disc_s + loss_disc_f

    # e2e/vocoder loss

    loss_mel = F.l1_loss(mel.transpose(1, 2), y_g_hat_mel) * 45

    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.vocoder_mpd(y.unsqueeze(1), y_g_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.vocoder_msd(y.unsqueeze(1), y_g_hat)
    loss_fm_f = VOC.feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = VOC.feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = VOC.generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = VOC.generator_loss(y_ds_hat_g)
    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
    if step < 6000: loss_gen_all += vc_reconst

    # CLUB
    club_x_val = ref_value
    club_x_key = ref_key
    club_y = vc_inputs.ref_pitch_i[:, :, 0]

    mi_val = self.club_val(club_x_val, club_y)
    mi_key = self.club_key(club_x_key, club_y)

    club_val = self.club_val.learning_loss(club_x_val, club_y)
    club_key = self.club_key.learning_loss(club_x_key, club_y)

    # aggregate
    total_model = loss_gen_all
    total_disc = loss_disc_all
    total_club = club_val + club_key

    ### === log === ###

    if log is not None:
      self.log(f"{log}_loss", total_model)
      self.log(f"{log}_loss_disc", total_disc)
      self.log(f"{log}_loss_club_val", club_val)
      self.log(f"{log}_loss_club_key", club_key)
      self.log(f"{log}_e2e_disc_f", loss_disc_f)
      self.log(f"{log}_e2e_disc_s", loss_disc_s)
      self.log(f"{log}_e2e_gen_f", loss_gen_f)
      self.log(f"{log}_e2e_gen_s", loss_gen_s)
      self.log(f"{log}_e2e_fm_f", loss_fm_f)
      self.log(f"{log}_e2e_fm_s", loss_fm_s)
      self.log(f"{log}_e2e_mel", loss_mel)
      self.log(f"{log}_reconst", vc_reconst)
      self.log(f"{log}_mi_val", mi_val)
      self.log(f"{log}_mi_key", mi_key)

    return (mel_hat, y_g_hat.squeeze(1)), (total_model, total_disc, total_club)

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

    _, (total_model, total_disc, total_club) = self._process_batch(batch, self_ratio, ref_ratio, step, log="train")

    opt_d.zero_grad()
    self.manual_backward(total_disc)
    self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip)
    opt_d.step()

    opt_model.zero_grad()
    self.manual_backward(total_model, retain_graph=True)
    self.clip_gradients(opt_model, gradient_clip_val=self.grad_clip)
    opt_model.step()

    opt_club.zero_grad()
    self.manual_backward(total_club)
    self.clip_gradients(opt_club, gradient_clip_val=self.grad_clip)
    opt_club.step()

    sch_club.step()
    sch_model.step()
    if step % 25000 == 0: sch_d.step()  # 25000: do_02500000 の steps/epoch

  def validation_step(self, batch: IntraDomainEntry4, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch[0].mel
    y = batch[0].audio

    # vcheat: validation with cheating
    (mel_hat_cheat, y_hat_cheat), _ = self._process_batch(batch, self_ratio=1.0, ref_ratio=1.0, step=step, log="vcheat")
    (mel_hat, y_hat), _ = self._process_batch(batch, self_ratio=0.0, ref_ratio=1.0, step=step, log="valid")

    if batch_idx == 0:
      names = [f"{i:02d}" for i in range(4)]
      log_spectrograms(self, names, mel, mel_hat, mel_hat_cheat)
      log_audios2(self, P, names, 22050, y, y_hat, y_hat_cheat)

  def configure_optimizers(self):
    h = self.hifi_gan

    # TODO: 今はとりあえず vocoder の重みの更新はしないで様子を見る

    opt_model = AdamW(self.model.parameters(), lr=self.lr)
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

    sch_model = S.MultiplicativeLR(opt_model, lambda step: 1.0)
    sch_club = S.MultiplicativeLR(opt_club, lambda step: 1.0)
    sch_d = S.ExponentialLR(opt_d, gamma=h.lr_decay, last_epoch=last_epoch)

    return [opt_model, opt_club, opt_d], [sch_model, sch_club, sch_d]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModule4(P, frames=32, n_samples=31, batch_size=8, n_items=8 * 1000, n_items_val=8 * 500)

  total_steps = 50000

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  model = VCModule(
      hdim=512,
      lr=2e-4,
      lr_club=5e-3,
      warmup_steps=500,
      total_steps=total_steps,
      milestones=(0, 1, 10000, 20000),
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

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_steps=total_steps * 2,  # optimizer を二回呼ぶので
      logger=wandb_logger,
      callbacks=[
          new_checkpoint_callback_wandb(PROJECT, wandb_logger),
      ],
      accelerator="gpu",
      precision="16-mixed",
      deterministic=False,
  )

  # train the model
  trainer.fit(model, datamodule=datamodule)

  trainer.chec

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()
