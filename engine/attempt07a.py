# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

# Based on attempt07

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
from engine.attempt07_dataset import (IntraDomainDataModuleA07, IntraDomainEntryA07)
from engine.attempt07a_prepare import AUG_ROOT
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.club import CLUBSampleForCategorical
from engine.lib.fastspeech import FFNBlock, PosFFT
from engine.lib.layers import Buckets, Transpose
from engine.lib.utils import AttrDict, clamp, hide_warns
from engine.prepare import Preparation
from engine.utils import (DATA_DIR, BaseLightningModule, log_attentions, log_audios2, log_audios3, log_spectrograms, log_spksim, new_checkpoint_callback_wandb,
                          new_wandb_logger, rotate_dim0, setup_train_environment, shuffle_dim0, step_optimizer)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim // 2
    mel_dim = hdim // 2
    kv_dim = hdim // 2

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
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.mel_encode = nn.Linear(80, mel_dim)
    self.encode_value = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        Transpose(1, 2),
        nn.Conv1d(hdim, kv_dim, kernel_size=3, padding=1),
        Transpose(1, 2),
        nn.ReLU(),
        nn.LayerNorm(kv_dim),
    )

    self.lookup = nn.MultiheadAttention(kv_dim, 4, dropout=0.2, batch_first=True)

    self.decode = nn.Sequential(
        nn.Linear(kv_dim + energy_dim + pitch_dim, hdim),
        PosFFT(hdim, layers=2, heads=2, hdim=256, kernels=(3, 3), dropout=0.2, posenc_len=2048),
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

  def forward(self, batch: IntraDomainEntryA07, src_ref_start: int, src_ref_len: int, use_aug: bool, n_rotated=0):
    # key: 似たような発音ほど近い表現になってほしい
    #      話者性が多少残ってても lookup 後の value への影響は間接的なので多分問題ない

    # value: 可能な限り多くの発音情報や話者性を含む表現になってほしい
    #        ただし、ピッチや音量によらない表現になってほしい
    #        （デコード時にピッチと音量を調節するので、そこと情報の衝突が起きないでほしい）

    n_frags = len(batch.ref) + 2  # number of audio fragments
    n_batch = len(batch.src.energy)
    seq_len = batch.src.energy.shape[1]

    assert seq_len >= src_ref_start + src_ref_len, f"seq_len={seq_len}, src_ref_start={src_ref_start}, src_ref_len={src_ref_len}"
    src_ref_end = src_ref_start + src_ref_len

    # (n_refs*n_batch, seq_len, dim)
    batch_energy = torch.stack([batch.src.energy, batch.aug.energy, *[o.energy for o in batch.ref]]).flatten(0, 1)
    batch_pitch_i = torch.stack([batch.src.pitch_i, batch.aug.pitch_i, *[o.pitch_i for o in batch.ref]]).flatten(0, 1)
    batch_w2v2 = torch.stack([batch.src.soft, batch.aug.soft, *[o.soft for o in batch.ref]]).flatten(0, 1)
    batch_mel = torch.stack([batch.src.mel, batch.aug.mel, *[o.mel for o in batch.ref]]).flatten(0, 1)

    # (n_refs*n_batch, seq_len, dim)
    batch_energy = self.forward_energy(batch_energy)
    batch_pitch = self.forward_pitch(batch_pitch_i)
    batch_w2v2 = self.forward_w2v2(batch_w2v2)
    batch_mel = self.forward_mel(batch_mel)
    batch_key = self.forward_key(batch_energy, batch_pitch, batch_w2v2)
    batch_value = self.forward_value(batch_energy, batch_pitch, batch_mel)

    # (n_refs, n_batch, seq_len, feat_dim)
    batch_energy = batch_energy.unflatten(0, (n_frags, n_batch))
    batch_pitch = batch_pitch.unflatten(0, (n_frags, n_batch))
    batch_key = batch_key.unflatten(0, (n_frags, n_batch))
    batch_value = batch_value.unflatten(0, (n_frags, n_batch))
    batch_pitch_i = batch_pitch_i.unflatten(0, (n_frags, n_batch))

    # (batch, len, feat_dim)
    src_key = batch_key[0]
    src_energy = batch_energy[0]
    src_pitch = batch_pitch[0]
    aug_key = batch_key[1]
    aug_energy = batch_energy[1]
    aug_pitch = batch_pitch[1]
    ref_key = torch.cat([batch_key[0, :, src_ref_start:src_ref_end], batch_key[2:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_value = torch.cat([batch_value[0, :, src_ref_start:src_ref_end], batch_value[2:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_pitch_i = torch.cat([batch_pitch_i[0, :, src_ref_start:src_ref_end], batch_pitch_i[2:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)

    the_key = aug_key if use_aug else src_key

    assert ref_key.shape[1] == seq_len * (n_frags - 2), f"ref_key.shape={ref_key.shape}, seq_len={seq_len}, n_refs={n_frags}"

    tgt_value, attn = self.lookup(the_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    # 単に rotate すると pitch mismatch のせいで spd_rot loss によりピッチが無視され始めるので、ピッチを適当にでもいいから近づけておく
    # 雑にピッチ分布の平均だけシフトさせて試してみようと思ったので、ピッチの平均値をはじめに出しておく
    rot_pitch_i = batch[0].pitch_i[:, :, 0]  # (n_batch, seq_len)
    rot_pitch_v = batch[0].pitch_v[:, :, 0]  # (n_batch, seq_len)
    rot_pitch_mask = rot_pitch_v > 0.5
    rot_pitch_means = (rot_pitch_i * rot_pitch_mask).sum(dim=-1) / rot_pitch_mask.sum(dim=-1)  # (n_batch,)

    rotated_mels: list[Tensor] = []
    the_key_rotated = the_key
    src_energy_rotated = src_energy
    src_pitch_i_rotated = batch[0].pitch_i.to(src_energy)  # (n_batch, seq_len, 1)
    rot_pitch_means_rotated = rot_pitch_means
    for i in range(n_rotated):
      the_key_rotated = rotate_dim0(the_key_rotated)
      src_energy_rotated = rotate_dim0(src_energy_rotated)
      src_pitch_i_rotated = rotate_dim0(src_pitch_i_rotated)
      rot_pitch_means_rotated = rotate_dim0(rot_pitch_means_rotated)
      src_pitch_i_rotated_converted = src_pitch_i_rotated + (rot_pitch_means - rot_pitch_means_rotated).unsqueeze(-1).unsqueeze(-1)
      src_pitch_rotated = self.forward_pitch(src_pitch_i_rotated_converted.clamp(0, 359).to(torch.int64))

      tgt_value_rotated, _ = self.lookup(the_key_rotated, ref_key, ref_value, need_weights=False)
      tgt_mel_rotated = self.decode(torch.cat([tgt_value_rotated, src_energy_rotated, src_pitch_rotated], dim=-1))
      rotated_mels.append(tgt_mel_rotated)

    return tgt_mel, (ref_key, ref_value, ref_pitch_i, attn), rotated_mels

class VCModule(BaseLightningModule):
  def __init__(self,
               hdim: int,
               lr: float,
               lr_club: float,
               warmup_steps: int,
               total_steps: int,
               milestones: tuple[int, int, int, int],
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
    self.milestones = milestones
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
      batch: IntraDomainEntryA07,
      self_ratio: float,
      step: int,
      e2e: bool,
      e2e_frames: Optional[int],
      use_aug: bool,
      train=False,
      log: Optional[str] = None,
      with_rot=False,
  ):
    opt_model, opt_club = self.optimizers()

    # NOTE: step 0 ではバグ確認のためすべてのロスを使って backward する。
    debug = step == 0 and train

    # inputs

    mel = batch.src.mel
    y = batch.src.audio

    # vc model

    seq_len = mel.shape[1]
    src_ref_len = int(self_ratio * seq_len)
    src_ref_start = self.batch_rand.randint(0, seq_len - src_ref_len)
    mel_hat, (ref_key, ref_value, ref_pitch_i, vc_attn), rotated_mel_hat = self.vc_model(
        batch, src_ref_start, src_ref_len, use_aug, n_rotated=4 if with_rot else 0)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"Charts (Main)/{log}_reconst", vc_reconst)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = ref_pitch_i[:, :, 0]
    club_sp_y = batch.aug.speaker.repeat(1, ref_key.shape[1])
    club_sp_n = shuffle_dim0(batch.aug.speaker).repeat(1, ref_key.shape[1])

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

    return mel_hat, y_g_hat, y_g_hat_mel, vc_attn, rotated_mel_hat

  def vocoder_forward(self, mel: Tensor):
    return self.vocoder(mel.transpose(1, 2)).squeeze(1)

  def training_step(self, batch: IntraDomainEntryA07, batch_idx: int):
    step = self.batches_that_stepped()
    milestones = self.milestones
    progress2 = (step - milestones[2]) / (milestones[3] - milestones[2])
    self_ratio = 1.0 - clamp(progress2, 0.0, 1.0) * 0.4  # TODO
    self.log("Charts (General)/self_ratio", self_ratio)

    opt_model, opt_club = self.optimizers()
    sch_model, sch_club = self.lr_schedulers()
    self.log("Charts (General)/lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_club", opt_club.optimizer.param_groups[0]["lr"])

    use_aug = step % 2 == 0  # TODO

    self._process_batch(batch, self_ratio, step, e2e=False, e2e_frames=self.e2e_frames, train=True, log="train", use_aug=use_aug)

    with hide_warns():
      sch_club.step()
      sch_model.step()

  def validation_step(self, batch: IntraDomainEntryA07, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch.src.mel
    y = batch.src.audio
    yaug = batch.aug.audio

    # vcheat: validation with cheating
    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics
    with_rot = batch_idx == 0 or complex_metrics
    _, y_c, _, _, melrot_c = self._process_batch(
        batch, self_ratio=1.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="vcheat", with_rot=with_rot, use_aug=False)
    _, y_v, _, _, melrot_v = self._process_batch(
        batch, self_ratio=0.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid", with_rot=with_rot, use_aug=False)

    # MOSNet を入れるとモデルの学習が非常に遅くなる
    # 多分メモリ不足によるものだけど、そこまでするほど MOSNet が正確なのか知らない
    # どうやら x-vector cos sim のほうが MOS との相関が強かったらしいし
    # see: https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_34.pdf

    if complex_metrics:
      yc_rot = self.vocoder_forward(melrot_c[0])
      yv_rot = self.vocoder_forward(melrot_v[0])

      log_spksim(self, P, yaug, y_v, y_c, yv_rot, yc_rot, folder="Charts (SpkSim Aug)")
      spksim = log_spksim(self, P, y, y_v, y_c, yv_rot, yc_rot)
      self.log("valid_spksim", spksim["valid_spksim"].mean())
      self.log("vcheat_spksim", spksim["vcheat_spksim"].mean())
      self.log("valid_vc_spksim", spksim["valid_spksim_vc"].mean())
      self.log("vcheat_vc_spksim", spksim["vcheat_spksim_vc"].mean())
      self.val_outputs.append(spksim)

    if batch_idx == 0:
      mel_c, y_c, ymel_c, attn_c, melrot_c = self._process_batch(batch, self_ratio=1.0, step=step, e2e=True, e2e_frames=None, with_rot=True, use_aug=False)
      mel_v, y_v, ymel_v, attn_v, melrot_v = self._process_batch(batch, self_ratio=0.0, step=step, e2e=True, e2e_frames=None, with_rot=True, use_aug=False)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_c, "Attention (Cheat)")
      log_attentions(self, names, attn_v, "Attention")
      log_spectrograms(self, names, mel, mel_c, ymel_c, "Spectrogram (Cheat)")
      log_spectrograms(self, names, rotate_dim0(mel), melrot_v[0], melrot_c[0], "Spectrogram (Rotated 0)")
      log_spectrograms(self, names, rotate_dim0(rotate_dim0(mel)), melrot_v[1], melrot_c[1], "Spectrogram (Rotated 1)")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_audios3(self, P, names, 22050, y, yaug, y_v, y_c)
      log_audios2(self, P, names, 22050, y, self.vocoder_forward(melrot_v[0]), self.vocoder_forward(melrot_c[0]), folder="Audio (Rotated 0)")
      log_audios2(self, P, names, 22050, y, self.vocoder_forward(melrot_v[1]), self.vocoder_forward(melrot_c[1]), folder="Audio (Rotated 1)")
      # log_audios2(self, P, names, 22050, y, yaug, batch.ref[0].audio, folder="Audio gf")

  def on_validation_epoch_end(self):
    if len(self.val_outputs) > 0:
      v_spksim = torch.cat([x["valid_spksim"] for x in self.val_outputs]).cpu().numpy()
      c_spksim = torch.cat([x["vcheat_spksim"] for x in self.val_outputs]).cpu().numpy()
      v2_spksim = torch.cat([x["valid_spksim_vc"] for x in self.val_outputs]).cpu().numpy()
      c2_spksim = torch.cat([x["vcheat_spksim_vc"] for x in self.val_outputs]).cpu().numpy()
      self.log_wandb({"Charts (SpkSim)/valid_hist": wandb.Histogram(v_spksim)})
      self.log_wandb({"Charts (SpkSim)/vcheat_hist": wandb.Histogram(c_spksim)})
      self.log_wandb({"Charts (SpkSim)/valid_hist_vc": wandb.Histogram(v2_spksim)})
      self.log_wandb({"Charts (SpkSim)/vcheat_hist_vc": wandb.Histogram(c2_spksim)})

    self.val_outputs.clear()

  def on_validation_end(self):
    # チェックポイントから学習を再開したときに、学習速度が 1/4 くらいになった。
    # メモリ使用率が高かったので試しにここでメモリを開放してみたところ、速さが（ほぼ）もとに戻った。
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
  assert PROJECT.startswith("attempt07")
  PROJECT = "attempt07"

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModuleA07(P, AUG_ROOT, frames=256, n_samples=16, batch_size=8, n_batches=1000, n_batches_val=200)

  total_steps = 20000
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
      milestones=(0, 1, 2000, 6000),
      e2e_frames=64,  # same as JETS https://arxiv.org/pdf/2203.16852.pdf
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
              filename="{step:08d}-{valid_spksim:.4f}-{vcheat_spksim:.4f}-{valid_vc_spksim:.4f}-{vcheat_vc_spksim:.4f}",
              monitor="valid_spksim",
              mode="max",
          ),
      ],
      accelerator="gpu",
      precision="16-mixed",
      benchmark=True,
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
