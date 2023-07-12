# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

from functools import cache
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
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.acgan import ACDiscriminator, BasicDiscriminator, aux_loss
from engine.lib.attention import MultiHeadAttention
from engine.lib.club import CLUBSampleForCategorical, CLUBSampleForCategorical3
from engine.lib.fastspeech import FFNBlock
from engine.lib.layers import Buckets, GetNth, Transpose
from engine.lib.utils import AttrDict, hide_warns, mix
from engine.prev.attempt10_dataset import DataModule09, Entry09
from engine.singleton import DATA_DIR, FEATS_DIR, P
from engine.utils import (BaseLightningModule, BinarySchedule, LinearSchedule, fm_loss, log_attentions, log_audios2, log_spectrograms, log_spksim1,
                          new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment, step_optimizer, step_optimizers)

@cache
def speaker_pitch(speaker_id: int):
  speaker = P.dataset.speaker_ids[speaker_id]
  feat_dir = FEATS_DIR / "parallel100" / speaker

  # TODO: 面倒なので直接呼んでる
  entry = datamodule.load_entry(feat_dir, speaker_id, 0, 32 * 256)

  mask = entry.pitch_v > 0.5
  return (entry.pitch_i * mask).sum() / mask.sum()

def discrete_pitch(pitch_i: Tensor):
  pitch_i = torch.round(pitch_i / 16) * 16
  pitch_i = torch.clamp(pitch_i, 0, 360 - 1)
  return pitch_i.to(torch.int64)

def discrete_energy(energy: Tensor):
  energy = torch.round(energy * 2) / 2
  return energy

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim // 2
    mel_dim = hdim // 2

    self.kdim = kdim = 32
    self.vdim = vdim = hdim

    self.mel_encode = nn.Linear(80, mel_dim)

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    # self.phoneme_embed = nn.Embedding(400, phoneme_dim)
    self.w2v2_embed = nn.Linear(256, w2v2_dim)

    self.encode_key = nn.Sequential(
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, kdim),
    )

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

    self.encode_key = nn.Sequential(
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.RNN(hdim, hdim, batch_first=True),
        GetNth(0),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kdim),
    )

    self.encode_key = nn.Sequential(
        nn.Linear(energy_dim + pitch_dim + w2v2_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.Linear(hdim, kdim),
        nn.ReLU(),
        nn.LayerNorm(kdim),
    )

    self.encode_value = nn.Sequential(
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, vdim),
    )

    self.encode_value = nn.Sequential(
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, vdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(vdim),
    )

    self.encode_value = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.Linear(hdim, vdim),
        nn.ReLU(),
        nn.LayerNorm(vdim),
    )

    self.lookup = MultiHeadAttention(kdim, vdim, 1, dropout=0.2, hard=True)

    self.decode = nn.Sequential(
        nn.Linear(vdim, hdim),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, 80),
    )

    self.decode = nn.Sequential(
        nn.Linear(vdim, hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.Linear(hdim, 80),
    )

    self.decode = nn.Sequential(
        nn.Linear(vdim + energy_dim + pitch_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, hdim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        nn.Linear(hdim, hdim),
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

  # def forward_phoneme(self, phoneme_i: Tensor, phoneme_v: Tensor):
  #   phoneme: Optional[Tensor] = None
  #   for k in range(VC_PHONEME_TOPK):
  #     emb_k = self.phoneme_embed(phoneme_i[:, :, k])
  #     emb_k *= phoneme_v[:, :, k].exp().unsqueeze(-1)
  #     phoneme = emb_k if phoneme is None else phoneme + emb_k
  #   return phoneme

  def forward_mel(self, mel: Tensor):
    return self.mel_encode(mel)

  def forward_key(self, energy: Tensor, pitch: Tensor, w2v2: Tensor):
    return self.encode_key(torch.cat([energy, pitch, w2v2], dim=-1))

  def forward_value(self, energy: Tensor, pitch: Tensor, mel: Tensor):
    return self.encode_value(torch.cat([energy, mel], dim=-1))

  def forward_lookup(self, key: Tensor, ref_key: Tensor, ref_value: Tensor, need_weights: bool = False):
    return self.lookup(key, ref_key, ref_value, need_weights=need_weights)

  def forward_decode(self, value: Tensor, energy: Tensor, pitch: Tensor):
    return self.decode(torch.cat([value, energy, pitch], dim=-1))

  def forward(self, batch: Entry09, src_ref_start: int, src_ref_len: int):
    # key: 似たような発音ほど近い表現になってほしい
    #      話者性が多少残ってても lookup 後の value への影響は間接的なので多分問題ない

    # value: 可能な限り多くの発音情報や話者性を含む表現になってほしい
    #        ただし、ピッチや音量によらない表現になってほしい
    #        （デコード時にピッチと音量を調節するので、そこと情報の衝突が起きないでほしい）

    n_refs = len(batch.ref)
    n_batch = len(batch.src.energy)
    ref_len = batch.ref[0].energy.shape[1]

    pitch_offset = [speaker_pitch(speaker.item()) for speaker in batch.src.speaker]
    pitch_offset = 360 / 2 - torch.tensor(pitch_offset, device=batch.src.energy.device, dtype=batch.src.energy.dtype).unsqueeze(0)
    conv_pitch = lambda pitch_i: torch.clamp(pitch_i + pitch_offset, 0, 360 - 1).to(torch.int64)

    src_ref_end = src_ref_start + src_ref_len

    # import matplotlib
    # from engine.lib.utils_ui import play_audio, plot_spectrogram
    # matplotlib.use('TkAgg')
    # m1 = batch.src.mel[0].unsqueeze(0)
    # s1 = batch.src.soft[0].unsqueeze(0)
    # m2 = torch.stack([o.mel for o in batch.ref]).transpose(0, 1).flatten(1, 2)[0].unsqueeze(0)
    # s2 = torch.stack([o.soft for o in batch.ref]).transpose(0, 1).flatten(1, 2)[0].unsqueeze(0)
    # play_audio(model.vocoder_forward(m1)[0], 22050)
    # play_audio(model.vocoder_forward(m2)[0], 22050)
    # plot_spectrogram(torch.mm(s1[0] / s1[0].norm(dim=1)[:, None], (s2[0] / s2[0].norm(dim=1)[:, None]).transpose(0, 1)))
    # raise Exception()

    # from engine.lib.utils_ui import play_audio
    # play_audio(model.vocoder_forward(batch.src.mel)[0], 22050)
    # play_audio(model.vocoder_forward(batch.ref[0].mel)[0], 22050)
    # play_audio(model.vocoder_forward(batch.ref[1].mel)[0], 22050)
    # play_audio(model.vocoder_forward(batch.ref[2].mel)[0], 22050)
    # play_audio(model.vocoder_forward(batch.ref[3].mel)[0], 22050)
    # raise Exception()

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_energy = torch.stack([discrete_energy(o.energy) for o in batch.ref]).flatten(0, 1)
    ref_pitch_i = torch.stack([discrete_pitch(conv_pitch(o.pitch_i)) for o in batch.ref]).flatten(0, 1)
    ref_w2v2 = torch.stack([o.soft for o in batch.ref]).flatten(0, 1)
    ref_mel = torch.stack([o.mel for o in batch.ref]).flatten(0, 1)

    src_energy = discrete_energy(batch.src.energy)
    src_pitch_i = discrete_pitch(conv_pitch(batch.src.pitch_i))
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
    src_value = self.forward_value(src_energy, src_pitch, src_mel)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim)
    ref_energy = ref_energy.unflatten(0, (n_refs, n_batch))
    ref_pitch = ref_pitch.unflatten(0, (n_refs, n_batch))
    ref_key = ref_key.unflatten(0, (n_refs, n_batch))
    ref_value = ref_value.unflatten(0, (n_refs, n_batch))
    ref_pitch_i = ref_pitch_i.unflatten(0, (n_refs, n_batch))

    # (...) -> (n_batch, n_refs*seq_len, feat_dim)
    ref_key = torch.cat([src_key[:, src_ref_start:src_ref_end], ref_key.transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_value = torch.cat([src_value[:, src_ref_start:src_ref_end], ref_value.transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_pitch_i = torch.cat([src_pitch_i[:, src_ref_start:src_ref_end], ref_pitch_i.transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)

    assert ref_key.shape[1] == ref_len * n_refs, f"ref_key.shape={ref_key.shape}, ref_len={ref_len}, n_refs={n_refs}"

    tgt_value, attn = self.forward_lookup(src_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.forward_decode(tgt_value, src_energy, src_pitch)

    return tgt_mel, (ref_key, ref_value, ref_pitch_i, attn), []

class VCModule(BaseLightningModule):
  def __init__(self,
               hdim: int,
               lr: float,
               lr_club: float,
               lr_spd: float,
               warmup_steps: int,
               total_steps: int,
               self_ratio: list[tuple[int, float]],
               grad_clip: float,
               e2e_ratio: list[tuple[int, float]],
               voc_train: list[tuple[int, bool]],
               e2e_frames: int,
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.vc_model = VCModel(hdim=hdim)
    self.vocoder = VOC.Generator(hifi_gan)
    self.vocoder_mpd = VOC.MultiPeriodDiscriminator()
    self.vocoder_msd = VOC.MultiScaleDiscriminator()

    self.club_val = CLUBSampleForCategorical(
        xdim=self.vc_model.vdim,
        ynum=360,
        hdim=hdim,
        fast_sampling=True,
    )
    self.club_key = CLUBSampleForCategorical(
        xdim=self.vc_model.kdim,
        ynum=360,
        hdim=hdim,
        fast_sampling=True,
    )
    ksp_xdim = self.vc_model.kdim
    ksp_ynum = len(P.dataset.speaker_ids)
    ksp_hdim = hdim
    self.club_ksp = CLUBSampleForCategorical3(
        xdim=ksp_xdim,
        ynum=ksp_ynum,
        hdim=hdim,
        fast_sampling=True,
        logvar=nn.Sequential(
            nn.Linear(ksp_xdim, ksp_hdim),
            nn.ReLU(),
            Transpose(1, 2),
            nn.Conv1d(ksp_hdim, ksp_hdim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(ksp_hdim, ksp_hdim, kernel_size=3, padding=1),
            Transpose(1, 2),
            nn.ReLU(),
            nn.Linear(ksp_hdim, ksp_ynum),
        ),
    )

    self.speaker_d = ACDiscriminator(
        BasicDiscriminator(
            dims=[64, 128, 512, 128, 64],
            kernels=[3, 5, 5, 5, 3],
            strides=[1, 2, 2, 1, 1],
            use_spectral_norm=False,  # spectral norm の weight / sigma で div by zero になってたので
        ),
        len(P.dataset.speaker_ids) * 2,  # ADC-GAN
        norm_feats=False,
    )

    self.batch_rand = Random(94324203)
    self.clip_rand = Random(76482573)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.self_ratio = LinearSchedule(self_ratio)
    self.lr = lr
    self.lr_club = lr_club
    self.lr_spd = lr_spd
    self.grad_clip = grad_clip
    self.e2e_ratio = LinearSchedule(e2e_ratio)
    self.voc_train = BinarySchedule(voc_train)
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
      batch: Entry09,
      self_ratio: float,
      step: int,
      e2e: bool,
      e2e_frames: Optional[int],
      train=False,
      log: Optional[str] = None,
  ):
    h = self.hifi_gan
    opt_model, opt_club, opt_voc, opt_d, opt_spd = self.optimizers()

    # NOTE: step 0 ではバグ確認のためすべてのロスを使って backward する。
    debug = step == 0 and train

    train_voc = self.voc_train(step)
    e2e_ratio = self.e2e_ratio(step)
    if debug: e2e_ratio = 0.1
    if log:
      self.log(f"Charts (General)/{log}_voc", 1.0 if train_voc else 0.0)
      self.log(f"Charts (General)/{log}_e2e_ratio", e2e_ratio)

    e2e = e2e or e2e_ratio > 0

    # inputs

    mel = batch.src.mel
    y = batch.src.audio

    # vc model

    seq_len = mel.shape[1]
    src_ref_len = int(self_ratio * seq_len)
    src_ref_start = self.batch_rand.randint(0, seq_len - src_ref_len)
    mel_hat, (ref_key, ref_value, ref_pitch_i, vc_attn), _ = self.vc_model(batch, src_ref_start, src_ref_len)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"Charts (Main)/{log}_reconst", vc_reconst)

    # speaker discriminator

    spd_real, _ = self.speaker_d(mel.detach())
    spd_fake, _ = self.speaker_d(mel_hat.detach())
    spd_loss_real = aux_loss(spd_real, batch.src.speaker * 2)
    spd_loss_fake = aux_loss(spd_fake, batch.src.speaker * 2 + 1)

    spd_loss = spd_loss_real + spd_loss_fake

    if train:
      step_optimizer(self, opt_spd, spd_loss, self.grad_clip)

    if log:
      self.log(f"Charts (SPD)/{log}_spd", spd_loss)
      self.log(f"Charts (SPD)/{log}_spd_real", spd_loss_real)
      self.log(f"Charts (SPD)/{log}_spd_fake", spd_loss_fake)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = ref_pitch_i[:, :, 0]
    club_sp_y = batch.src.speaker.repeat(1, ref_key.shape[1])
    # club_sp_n = shuffle_dim0(batch.src.speaker).repeat(1, ref_key.shape[1])
    club_sp_n = batch.src.speaker.reshape(-1)
    club_sp_n = club_sp_n[torch.randint_like(club_sp_y, 0, len(club_sp_n))]

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

      # discriminator

      y_df_hat_r, y_df_hat_g, _, _ = self.vocoder_mpd(e2e_y, y_g_hat.detach())
      loss_disc_f = VOC.discriminator_loss(y_df_hat_r, y_df_hat_g)

      y_ds_hat_r, y_ds_hat_g, _, _ = self.vocoder_msd(e2e_y, y_g_hat.detach())
      loss_disc_s = VOC.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

      total_disc = loss_disc_s + loss_disc_f

      if train:
        step_optimizer(self, opt_d, total_disc, self.grad_clip)

      if log:
        self.log(f"Charts (Main)/{log}_e2e_disc", total_disc)
        self.log(f"Charts (E2E)/{log}_e2e_disc_f", loss_disc_f)
        self.log(f"Charts (E2E)/{log}_e2e_disc_s", loss_disc_s)

      # e2e generator loss

      loss_mel = F.l1_loss(e2e_mel, y_g_hat_mel)

      y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.vocoder_mpd(e2e_y, y_g_hat)
      y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.vocoder_msd(e2e_y, y_g_hat)
      loss_fm_f = VOC.feature_loss(fmap_f_r, fmap_f_g)
      loss_fm_s = VOC.feature_loss(fmap_s_r, fmap_s_g)
      loss_gen_f, _ = VOC.generator_loss(y_df_hat_g)
      loss_gen_s, _ = VOC.generator_loss(y_ds_hat_g)

      e2e_model_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45.0

      if log:
        self.log(f"Charts (E2E)/{log}_e2e_gen_f", loss_gen_f)
        self.log(f"Charts (E2E)/{log}_e2e_gen_s", loss_gen_s)
        self.log(f"Charts (E2E)/{log}_e2e_fm_f", loss_fm_f)
        self.log(f"Charts (E2E)/{log}_e2e_fm_s", loss_fm_s)
        self.log(f"Charts (Main)/{log}_e2e_reconst", loss_mel)

      y_g_hat = y_g_hat.squeeze(1)
      y_g_hat_mel = y_g_hat_mel.transpose(1, 2)

    # generator (speaker discriminator loss)

    with torch.no_grad():
      _, spd_f_real = self.speaker_d(mel)
    spd_c_fake, spd_f_fake = self.speaker_d(mel_hat)

    spd_g_fm = fm_loss(spd_f_real, spd_f_fake, F.l1_loss)
    spd_g_pos = aux_loss(spd_c_fake, batch.src.speaker * 2)
    spd_g_neg = aux_loss(spd_c_fake, batch.src.speaker * 2 + 1)

    if log:
      self.log(f"Charts (SPD)/{log}_spd_g_fm", spd_g_fm)
      self.log(f"Charts (SPD)/{log}_spd_g_pos", spd_g_pos)
      self.log(f"Charts (SPD)/{log}_spd_g_neg", spd_g_neg)

    # generator

    total_model = vc_reconst * 150.0
    if e2e_ratio > 0: total_model = mix(x=e2e_model_loss, y=total_model, ratio_x=e2e_ratio)
    total_model += mi_val
    total_model += mi_ksp
    total_model += spd_g_fm + spd_g_pos - spd_g_neg

    if train:
      if train_voc or debug:
        step_optimizers(self, [opt_model, opt_voc], total_model, self.grad_clip)
      else:
        step_optimizer(self, opt_model, total_model, self.grad_clip)

    if log:
      self.log(f"Charts (Main)/{log}_loss", total_model)

    return mel_hat, y_g_hat, y_g_hat_mel, vc_attn, []

  def vocoder_forward(self, mel: Tensor):
    return self.vocoder(mel.transpose(1, 2)).squeeze(1)

  def training_step(self, batch: Entry09, batch_idx: int):
    step = self.batches_that_stepped()

    self_ratio = self.self_ratio(step)
    self.log("Charts (General)/self_ratio", self_ratio)

    opt_model, opt_club, opt_voc, opt_d, opt_spd = self.optimizers()
    sch_model, sch_club, sch_voc, sch_d, sch_spd = self.lr_schedulers()
    self.log("Charts (General)/lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_club", opt_club.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_voc", opt_voc.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_d", opt_d.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_spd", opt_spd.optimizer.param_groups[0]["lr"])

    self._process_batch(batch, self_ratio, step, e2e=False, e2e_frames=self.e2e_frames, train=True, log="train")

    with hide_warns():
      sch_club.step()
      sch_model.step()
      if step % 25000 == 0: sch_voc.step()  # 25000: do_02500000 の steps/epoch
      if step % 25000 == 0: sch_d.step()  # 25000: do_02500000 の steps/epoch
      sch_spd.step()

  def validation_step(self, batch: Entry09, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch.src.mel
    y = batch.src.audio

    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics
    _, y_c, _, _, _ = self._process_batch(batch, self_ratio=1.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="vcheat")
    _, y_v, _, _, _ = self._process_batch(batch, self_ratio=0.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

    # MOSNet を入れるとモデルの学習が非常に遅くなる
    # 多分メモリ不足によるものだけど、そこまでするほど MOSNet が正確なのか知らない
    # どうやら x-vector cos sim のほうが MOS との相関が強かったらしいし
    # see: https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_34.pdf

    if complex_metrics:
      spksim = log_spksim1(self, y, y_v, y_c)
      self.log("valid_spksim", spksim["valid_spksim"].mean())
      self.log("vcheat_spksim", spksim["vcheat_spksim"].mean())
      self.val_outputs.append(spksim)

    if batch_idx == 0:
      mel_c, y_c, ymel_c, attn_c, _ = self._process_batch(batch, self_ratio=1.0, step=step, e2e=True, e2e_frames=None)
      mel_v, y_v, ymel_v, attn_v, _ = self._process_batch(batch, self_ratio=0.0, step=step, e2e=True, e2e_frames=None)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_c, "Attention (Cheat)")
      log_attentions(self, names, attn_v, "Attention")

      log_spectrograms(self, names, mel, mel_c, ymel_c, "Spectrogram (Cheat)")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_audios2(self, names, 22050, y, y_v, y_c)

  def on_validation_epoch_end(self):
    if len(self.val_outputs) > 0:
      v_spksim = torch.cat([x["valid_spksim"] for x in self.val_outputs]).cpu().numpy()
      c_spksim = torch.cat([x["vcheat_spksim"] for x in self.val_outputs]).cpu().numpy()
      self.log_wandb({"Charts (SpkSim)/valid_hist": wandb.Histogram(v_spksim)})
      self.log_wandb({"Charts (SpkSim)/vcheat_hist": wandb.Histogram(c_spksim)})

    self.val_outputs.clear()

  def on_validation_end(self):
    # チェックポイントから学習を再開したときに、学習速度が 1/4 くらいになった。
    # メモリ使用率が高かったので試しにここでメモリを開放してみたところ、速さが（ほぼ）もとに戻った。
    P.release_spkemb()
    P.release_mosnet()
    torch.cuda.empty_cache()

  def configure_optimizers(self):
    h = self.hifi_gan

    # TODO: 今はとりあえず vocoder の重みの更新はしないで様子を見る

    opt_model = AdamW(self.vc_model.parameters(), lr=self.lr)
    opt_club = AdamW([*self.club_val.parameters(), *self.club_key.parameters(), *self.club_ksp.parameters()], lr=self.lr_club)
    opt_voc = AdamW(self.vocoder.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    opt_d = AdamW([*self.vocoder_msd.parameters(), *self.vocoder_mpd.parameters()], h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    opt_spd = AdamW(self.speaker_d.parameters(), lr=self.lr_spd)

    # TODO: こんな方法とタイミングで事前学習済みモデルを読み込むのが最善とは思えない
    last_epoch = -1
    if self.hifi_gan_ckpt is not None:
      if self.trainer.ckpt_path is not None:
        print("Skipped loading pretrained HiFi-GAN weights because the training was resumed from a checkpoint.")
      else:
        print("Loading pretrained HiFi-GAN weights...")
        g, do = self.hifi_gan_ckpt
        g_data = torch.load(g, map_location=self.device)
        do_data = torch.load(do, map_location=self.device)
        last_epoch = do_data['epoch']
        self.vocoder.load_state_dict(g_data["generator"])
        self.vocoder_mpd.load_state_dict(do_data['mpd'])
        self.vocoder_msd.load_state_dict(do_data['msd'])
        opt_voc.load_state_dict(do_data['optim_g'])
        opt_d.load_state_dict(do_data['optim_d'])

    # TODO: resume 時に step に渡される値がリセットされていそう
    sch_model = S.ChainedScheduler([
        get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps),
    ])
    sch_club = S.LambdaLR(opt_club, lambda step: 1.0)
    sch_voc = S.ExponentialLR(opt_voc, gamma=h.lr_decay, last_epoch=last_epoch)
    sch_d = S.ExponentialLR(opt_d, gamma=h.lr_decay, last_epoch=last_epoch)
    sch_spd = S.MultiplicativeLR(opt_spd, lambda step: 1.0)

    return [opt_model, opt_club, opt_voc, opt_d, opt_spd], [sch_model, sch_club, sch_voc, sch_d, sch_spd]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem.split("_")[0].split(" ")[0]
  assert PROJECT.startswith("attempt08")
  PROJECT = "attempt08"

  setup_train_environment()

  P.set_device("cuda")
  datamodule = DataModule09(
      P, frames=256, frames_ref=32, n_refs=64, ref_max_kth=64, batch_size=8, n_batches=1000, n_batches_val=200, same_density=True, num_workers=12)

  total_steps = 100000
  total_actual_steps = 50000
  complex_metrics_batches = 50  # see: validation_step

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  VC_PHONEME_TOPK = 2
  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_club=1e-4,
      lr_spd=1e-4,
      warmup_steps=500,
      total_steps=total_steps,
      self_ratio=[(0, 0.0)],
      e2e_frames=64,  # same as JETS https://arxiv.org/pdf/2203.16852.pdf
      e2e_ratio=[(0, 0.0), (20000, 0.0), (24000, 1.0)],
      voc_train=[(0, False), (30000, True)],
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
      # not benchmark and deterministic にしたら spkemb の計算の conv1d が大部分の時間を占めた
      # detect_anomaly=True,
      # deterministic=True,
      # profiler=profilers.PyTorchProfiler(
      #     DATA_DIR / "profiler",
      #     schedule=torch.profiler.schedule(wait=0, warmup=30, active=6, repeat=1),
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

  # print(prof.report())
  # print(torch._dynamo.utils.compile_times())

  # [optional] finish the wandb run, necessary in notebooks
  wandb.finish()

  print(f"Saved model to {Path(trainer.checkpoint_callback.last_model_path).relative_to(DATA_DIR)}")
