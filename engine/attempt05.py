# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

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
from wandb.wandb_run import Run

import engine.hifi_gan.models as VOC
from engine.dataset_feats import (FeatureEntry4, IntraDomainDataModule4, IntraDomainEntry4)
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.club import CLUBSampleForCategorical
from engine.lib.fastspeech import FFNBlock, PosFFT
from engine.lib.layers import Buckets, GetNth
from engine.lib.utils import AttrDict, clamp, hide_warns
from engine.prepare import Preparation
from engine.utils import (DATA_DIR, log_attentions, log_audios2, log_spectrograms, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    phoneme_dim = hdim // 2
    mel_dim = hdim // 2
    kv_dim = hdim // 2

    self.kv_dim = kv_dim

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.phoneme_embed = nn.Embedding(400, phoneme_dim)
    self.encode_key = nn.Sequential(
        nn.Linear(energy_dim + pitch_dim + phoneme_dim, hdim),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.RNN(hdim, hdim, batch_first=True),
        GetNth(0),
        nn.ReLU(),
        nn.LayerNorm(hdim),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, kv_dim),
    )

    self.mel_encode = nn.Linear(80, mel_dim)
    self.encode_value = nn.Sequential(
        nn.Linear(energy_dim + mel_dim, hdim),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, kv_dim),
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
    return self.encode_value(torch.cat([energy, mel], dim=-1))

  def forward(self, batch: IntraDomainEntry4, src_ref_start: int, src_ref_len: int):
    # key: 似たような発音ほど近い表現になってほしい
    #      話者性が多少残ってても lookup 後の value への影響は間接的なので多分問題ない

    # value: 可能な限り多くの発音情報や話者性を含む表現になってほしい
    #        ただし、ピッチや音量によらない表現になってほしい
    #        （デコード時にピッチと音量を調節するので、そこと情報の衝突が起きないでほしい）

    n_refs = len(batch)
    n_batch = len(batch[0].energy)
    seq_len = batch[0].energy.shape[1]

    assert seq_len >= src_ref_start + src_ref_len, f"seq_len={seq_len}, src_ref_start={src_ref_start}, src_ref_len={src_ref_len}"
    src_ref_end = src_ref_start + src_ref_len

    batch_energy = torch.stack([o.energy for o in batch]).flatten(0, 1)
    batch_pitch_i = torch.stack([o.pitch_i for o in batch]).flatten(0, 1)
    batch_phoneme_i = torch.stack([o.phoneme_i for o in batch]).flatten(0, 1)
    batch_phoneme_v = torch.stack([o.phoneme_v for o in batch]).flatten(0, 1)
    batch_mel = torch.stack([o.mel for o in batch]).flatten(0, 1)

    batch_energy = self.forward_energy(batch_energy)
    batch_pitch = self.forward_pitch(batch_pitch_i)
    batch_phoneme = self.forward_phoneme(batch_phoneme_i, batch_phoneme_v)
    batch_mel = self.forward_mel(batch_mel)
    batch_key = self.forward_key(batch_energy, batch_pitch, batch_phoneme)
    batch_value = self.forward_value(batch_energy, batch_pitch, batch_mel)

    # (n_refs, n_batch, seq_len, feat_dim)
    batch_energy = batch_energy.unflatten(0, (n_refs, n_batch))
    batch_pitch = batch_pitch.unflatten(0, (n_refs, n_batch))
    batch_key = batch_key.unflatten(0, (n_refs, n_batch))
    batch_value = batch_value.unflatten(0, (n_refs, n_batch))
    batch_pitch_i = batch_pitch_i.unflatten(0, (n_refs, n_batch))

    # (batch, len, feat_dim)
    src_key = batch_key[0]
    src_energy = batch_energy[0]
    src_pitch = batch_pitch[0]
    ref_key = torch.cat([batch_key[0, :, src_ref_start:src_ref_end], batch_key[1:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_value = torch.cat([batch_value[0, :, src_ref_start:src_ref_end], batch_value[1:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_pitch_i = torch.cat([batch_pitch_i[0, :, src_ref_start:src_ref_end], batch_pitch_i[1:].transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)

    assert ref_key.shape[1] == seq_len * (n_refs - 1), f"ref_key.shape={ref_key.shape}, seq_len={seq_len}, n_refs={n_refs}"

    tgt_value, attn = self.lookup(src_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(torch.cat([tgt_value, src_energy, src_pitch], dim=-1))

    return tgt_mel, (ref_key, ref_value, ref_pitch_i, attn)

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

    self.club_val = CLUBSampleForCategorical(xdim=self.vc_model.kv_dim, ynum=360, hdim=hdim, fast_sampling=True)
    self.club_key = CLUBSampleForCategorical(xdim=self.vc_model.kv_dim, ynum=360, hdim=hdim, fast_sampling=True)

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

    self.val_outputs = []

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

  def _process_batch(self,
                     batch: IntraDomainEntry4,
                     self_ratio: float,
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
    # if debug: print('process_batch: debug mode')

    # inputs
    mel = batch[0].mel
    y = batch[0].audio

    # vc model

    seq_len = mel.shape[1]
    src_ref_len = int(self_ratio * seq_len)
    src_ref_start = self.batch_rand.randint(0, seq_len - src_ref_len)
    mel_hat, (ref_key, ref_value, ref_pitch_i, vc_attn) = self.vc_model(batch, src_ref_start, src_ref_len)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"{log}_reconst", vc_reconst)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = ref_pitch_i[:, :, 0]

    mi_val = self.club_val(club_x_val, club_y)
    mi_key = self.club_key(club_x_key, club_y)

    club_val = self.club_val.learning_loss(club_x_val.detach(), club_y.detach())
    club_key = self.club_key.learning_loss(club_x_key.detach(), club_y.detach())

    total_club = club_val + club_key

    if train:
      self.toggle_optimizer(opt_club)
      opt_club.zero_grad()
      self.manual_backward(total_club, retain_graph=True)
      self.clip_gradients(opt_club, gradient_clip_val=self.grad_clip)
      opt_club.step()
      self.untoggle_optimizer(opt_club)

    if log:
      self.log(f"{log}_mi_val", mi_val)
      self.log(f"{log}_mi_key", mi_key)
      self.log(f"{log}_loss_club_val", club_val)
      self.log(f"{log}_loss_club_key", club_key)

    # discriminator, e2e

    total_model = vc_reconst * 60
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
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        self.manual_backward(total_disc)
        self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip)
        opt_d.step()  # この行をコメントアウトしない場合、 e2e_milestones[0] 以降で generator のスコアも一気に低下していた。
        self.untoggle_optimizer(opt_d)

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
      loss_gen_f, _ = VOC.generator_loss(y_df_hat_g)
      loss_gen_s, _ = VOC.generator_loss(y_ds_hat_g)

      e2e_model_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45.0

      e2e_ratio = clamp((step - self.e2e_milestones[1]) / (self.e2e_milestones[2] - self.e2e_milestones[1]), 0.0, 1.0)
      if debug: e2e_ratio = 0.1
      total_model = total_model * (1.0 - e2e_ratio) + e2e_model_loss * e2e_ratio

      if log:
        self.log(f"{log}_e2e_gen_f", loss_gen_f)
        self.log(f"{log}_e2e_gen_s", loss_gen_s)
        self.log(f"{log}_e2e_fm_f", loss_fm_f)
        self.log(f"{log}_e2e_fm_s", loss_fm_s)
        self.log(f"{log}_e2e_mel", loss_mel)

      y_g_hat = y_g_hat.squeeze(1)
      y_g_hat_mel = y_g_hat_mel.transpose(1, 2)

    # generator

    total_model += 0.01 * mi_val

    if train:
      self.toggle_optimizer(opt_model)
      opt_model.zero_grad()
      self.manual_backward(total_model)
      self.clip_gradients(opt_model, gradient_clip_val=self.grad_clip)
      opt_model.step()
      self.untoggle_optimizer(opt_model)

    if log:
      self.log(f"{log}_loss", total_model)

    return mel_hat, y_g_hat, y_g_hat_mel, vc_attn

  def training_step(self, batch: IntraDomainEntry4, batch_idx: int):
    step = self.batches_that_stepped()
    milestones = self.milestones
    progress2 = (step - milestones[2]) / (milestones[3] - milestones[2])
    self_ratio = 1.0 - clamp(progress2, 0.0, 1.0) * 0.8
    self.log("self_ratio", self_ratio)

    opt_model, opt_club, opt_d = self.optimizers()
    sch_model, sch_club, sch_d = self.lr_schedulers()
    self.log("lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("lr_club", opt_club.optimizer.param_groups[0]["lr"])
    self.log("lr_d", opt_d.optimizer.param_groups[0]["lr"])

    self._process_batch(batch, self_ratio, step, e2e=step >= self.e2e_milestones[0], e2e_frames=self.e2e_frames, train=True, log="train")

    with hide_warns():
      sch_club.step()
      sch_model.step()
      if step % 25000 == 0: sch_d.step()  # 25000: do_02500000 の steps/epoch

  def validation_step(self, batch: IntraDomainEntry4, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch[0].mel
    y = batch[0].audio

    # vcheat: validation with cheating
    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics or step >= self.e2e_milestones[0]
    _, y_c, _, _ = self._process_batch(batch, self_ratio=1.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="vcheat")
    _, y_v, _, _ = self._process_batch(batch, self_ratio=0.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

    # MOSNet を入れるとモデルの学習が非常に遅くなる
    # 多分メモリ不足によるものだけど、そこまでするほど MOSNet が正確なのか知らない
    # どうやら x-vector cos sim のほうが MOS との相関が強かったらしいし
    # see: https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_34.pdf

    if complex_metrics:
      y_spkemb = P.spkemb(y, 22050)
      y_v_spkemb = P.spkemb(y_v, 22050)
      y_c_spkemb = P.spkemb(y_c, 22050)
      v_spksim = F.cosine_similarity(y_spkemb, y_v_spkemb)
      c_spksim = F.cosine_similarity(y_spkemb, y_c_spkemb)
      self.log("valid_spksim", v_spksim.mean())
      self.log("vcheat_spksim", c_spksim.mean())

      # inter-speaker
      rotate = lambda x: torch.cat([x[1:], x[:1]], dim=0)
      inter_batch = [o for o in batch]
      inter_batch[0] = FeatureEntry4(*[rotate(o) for o in inter_batch[0]])

      _, yc2, _, _ = self._process_batch(inter_batch, self_ratio=1.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="vcheat")
      _, yv2, _, _ = self._process_batch(inter_batch, self_ratio=0.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

      yv2_spkemb = P.spkemb(yv2, 22050)
      yc2_spkemb = P.spkemb(yc2, 22050)
      v2_spksim = F.cosine_similarity(y_spkemb, yv2_spkemb)
      c2_spksim = F.cosine_similarity(y_spkemb, yc2_spkemb)
      v2_spksim_leak = F.cosine_similarity(rotate(y_spkemb), yv2_spkemb)
      c2_spksim_leak = F.cosine_similarity(rotate(y_spkemb), yc2_spkemb)
      v2_spksim_irrelevent = F.cosine_similarity(y_spkemb, rotate(yv2_spkemb))
      c2_spksim_irrelevent = F.cosine_similarity(y_spkemb, rotate(yc2_spkemb))
      self.log("valid_spksim_vc", v2_spksim.mean())
      self.log("vcheat_spksim_vc", c2_spksim.mean())
      self.log("valid_spksim_leak", v2_spksim_leak.mean())
      self.log("vcheat_spksim_leak", c2_spksim_leak.mean())
      self.log("valid_spksim_base", v2_spksim_irrelevent.mean())
      self.log("vcheat_spksim_base", c2_spksim_irrelevent.mean())

      self.val_outputs.append({"valid_spksim": v_spksim, "vcheat_spksim": c_spksim, "valid_spksim_vc": v2_spksim, "vcheat_spksim_vc": c2_spksim})

    if batch_idx == 0:
      mel_c, y_c, ymel_c, attn_c = self._process_batch(batch, self_ratio=1.0, step=step, e2e=True, e2e_frames=None)
      mel_v, y_v, ymel_v, attn_v = self._process_batch(batch, self_ratio=0.0, step=step, e2e=True, e2e_frames=None)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_c, "Attention (Cheat)")
      log_attentions(self, names, attn_v, "Attention")
      log_spectrograms(self, names, mel, mel_c, ymel_c, "Spectrogram (Cheat)")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_audios2(self, P, names, 22050, y, y_v, y_c)

  def on_validation_epoch_end(self):
    if len(self.val_outputs) > 0:
      v_spksim = torch.cat([x["valid_spksim"] for x in self.val_outputs]).cpu().numpy()
      c_spksim = torch.cat([x["vcheat_spksim"] for x in self.val_outputs]).cpu().numpy()
      v2_spksim = torch.cat([x["valid_spksim_vc"] for x in self.val_outputs]).cpu().numpy()
      c2_spksim = torch.cat([x["vcheat_spksim_vc"] for x in self.val_outputs]).cpu().numpy()
      self.log_wandb({"Spksim/valid_hist": wandb.Histogram(v_spksim)})
      self.log_wandb({"Spksim/vcheat_hist": wandb.Histogram(c_spksim)})
      self.log_wandb({"Spksim/valid_hist_vc": wandb.Histogram(v2_spksim)})
      self.log_wandb({"Spksim/vcheat_hist_vc": wandb.Histogram(c2_spksim)})

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
    opt_club = AdamW([*self.club_val.parameters(), *self.club_key.parameters()], lr=self.lr_club)
    opt_d = AdamW([*self.vocoder_msd.parameters(), *self.vocoder_mpd.parameters()], h.learning_rate, betas=[h.adam_b1, h.adam_b2])

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

    sch_model = S.ChainedScheduler([
        get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps),
        S.MultiplicativeLR(opt_model, lambda step: min(1.0, (step - self.e2e_milestones[0]) / 1000) if step >= self.e2e_milestones[0] else 1.0),
        S.MultiplicativeLR(opt_model, lambda step: min(1.0, (step - self.e2e_milestones[2]) / 1000) if step >= self.e2e_milestones[2] else 1.0),
    ])
    sch_club = S.MultiplicativeLR(opt_club, lambda step: 1.0)
    sch_d = S.ExponentialLR(opt_d, gamma=h.lr_decay, last_epoch=last_epoch)

    return [opt_model, opt_club, opt_d], [sch_model, sch_club, sch_d]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem

  setup_train_environment()

  P = Preparation("cuda")

  datamodule = IntraDomainDataModule4(P, frames=256, n_samples=8, batch_size=8, n_batches=1000, n_batches_val=200)

  total_steps = 100000
  complex_metrics_batches = 50  # see: validation_step

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_club=1e-4,
      warmup_steps=500,
      total_steps=total_steps,
      milestones=(0, 1, 10000, 20000),
      e2e_milestones=(22000, 30000, 40000),
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
          new_checkpoint_callback_wandb(
              PROJECT,
              wandb_logger,
              filename="{step:08d}-{valid_spksim:.4f}-{vcheat_spksim:.4f}",
              monitor="valid_spksim",
              mode="max",
          ),
      ],
      accelerator="gpu",
      precision="16-mixed",
      benchmark=True,
      # deterministic=True,
      # detect_anomaly=True,
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
