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
from engine.attempt09_dataset import DataModule09, Dataset09, Entry09
from engine.fragment_vc.utils import get_cosine_schedule_with_warmup
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.acgan import ACDiscriminator, BasicDiscriminator, aux_loss
from engine.lib.attention import MultiHeadAttention
from engine.lib.fastspeech import FFNBlock, PosFFT
from engine.lib.layers import Buckets, GetNth, Transpose
from engine.lib.utils import AttrDict, clamp, hide_warns
from engine.prepare import FEATS_DIR, Preparation
from engine.utils import (DATA_DIR, BaseLightningModule, fm_loss, log_attentions, log_audios2, log_spectrograms, log_spksim0, log_spksim1,
                          new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment, shuffle_dim0, step_optimizer)

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    kdim = hdim
    vdim = hdim

    self.kdim = kdim
    self.vdim = vdim

    self.mel_encode = nn.Linear(80, hdim)

    # shape: (batch, src_len, hdim)
    self.encode_key = nn.Sequential(
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, kdim),
    )

    self.encode_value = nn.Sequential(
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, vdim),
    )

    self.lookup = MultiHeadAttention(kdim, vdim, 16, dropout=0.2, hard=False)

    self.decode = nn.Sequential(
        nn.Linear(vdim, hdim),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        FFNBlock(hdim, hdim, kernels=(3, 3), dropout=0.2),
        nn.Linear(hdim, 80),
    )

  def forward(self, refs: list[Entry09], src_mel: Tensor):
    n_refs = len(refs)
    n_batch = len(src_mel)
    ref_len = refs[0].mel.shape[1]

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_mel = torch.stack([o.mel for o in refs]).flatten(0, 1)

    # (...) -> (n_refs*n_batch, seq_len, dim)
    ref_mel = self.mel_encode(ref_mel)
    ref_key = self.encode_key(ref_mel)
    ref_value = self.encode_value(ref_mel)

    src_mel = self.mel_encode(src_mel)
    src_key = self.encode_key(src_mel)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim) -> (n_batch, n_refs*seq_len, feat_dim)
    ref_key = ref_key.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)
    ref_value = ref_value.unflatten(0, (n_refs, n_batch)).transpose(0, 1).flatten(1, 2)

    assert ref_key.shape[1] == ref_len * n_refs, f"ref_key.shape={ref_key.shape}, ref_len={ref_len}, n_refs={n_refs}"

    tgt_value, attn = self.lookup(src_key, ref_key, ref_value, need_weights=True)

    # shape: (batch, src_len, 80)
    tgt_mel = self.decode(tgt_value)

    return tgt_mel, (ref_key, ref_value, None, attn), []

class VCModule(BaseLightningModule):
  def __init__(self,
               hdim: int,
               lr: float,
               lr_spd: float,
               warmup_steps: int,
               total_steps: int,
               grad_clip: float,
               e2e_frames: int,
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.vc_model = VCModel(hdim=hdim)
    self.vocoder = VOC.Generator(hifi_gan)

    self.speaker_d = ACDiscriminator(
        BasicDiscriminator(
            dims=[64, 128, 512, 128, 64],
            kernels=[3, 5, 5, 5, 3],
            strides=[1, 2, 2, 1, 1],
            use_spectral_norm=False,  # spectral norm の weight / sigma で div by zero になってたので
            avg_pool=nn.AvgPool2d((4, 4), stride=(2, 2))),  # 複数の部分ごとに話者判定する
        len(P.dataset.speaker_ids) * 2,  # ADC-GAN
        norm_feats=False,
    )

    self.batch_rand = Random(94324203)
    self.clip_rand = Random(76482573)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.lr = lr
    self.lr_spd = lr_spd
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
      batch: Entry09,
      step: int,
      e2e: bool,
      e2e_frames: Optional[int],
      train=False,
      log: Optional[str] = None,
  ):
    opt_model, opt_spd = self.optimizers()

    # NOTE: step 0 ではバグ確認のためすべてのロスを使って backward する。
    debug = step == 0 and train

    # inputs

    mel = batch.src.mel
    y = batch.src.audio

    # vc model

    tgt_mel_hat, (tgt_ref_key, tgt_ref_value, _, tgt_vc_attn), _ = self.vc_model(batch.tgt_ref, mel)
    src_mel_hat, (src_ref_key, src_ref_value, _, src_vc_attn), _ = self.vc_model(batch.ref, mel)
    mel_hat, (ref_key, ref_value, _, vc_attn), _ = self.vc_model(batch.ref, tgt_mel_hat)

    vc_reconst = F.l1_loss(mel_hat, mel)
    src2src_reconst = F.l1_loss(src_mel_hat, mel)

    if log:
      self.log(f"Charts (Main)/{log}_reconst", vc_reconst)
      self.log(f"Charts (Main)/{log}_reconst_src2src", src2src_reconst)

    # speaker discriminator

    spd_real, _ = self.speaker_d(mel.detach())
    spd_fake, _ = self.speaker_d(tgt_mel_hat.detach())
    spd_loss_real = aux_loss(spd_real, batch.src.speaker * 2)
    spd_loss_fake = aux_loss(spd_fake, batch.tgt_speaker * 2 + 1)

    spd_loss = spd_loss_real + spd_loss_fake

    if train:
      step_optimizer(self, opt_spd, spd_loss, self.grad_clip)

    if log:
      self.log(f"Charts (SPD)/{log}_spd", spd_loss)
      self.log(f"Charts (SPD)/{log}_spd_real", spd_loss_real)
      self.log(f"Charts (SPD)/{log}_spd_fake", spd_loss_fake)

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

    # generator (speaker discriminator loss)

    spd_c_fake, _ = self.speaker_d(tgt_mel_hat)

    spd_g_pos = aux_loss(spd_c_fake, batch.tgt_speaker * 2)
    spd_g_neg = aux_loss(spd_c_fake, batch.tgt_speaker * 2 + 1)

    if log:
      self.log(f"Charts (SPD)/{log}_spd_g_pos", spd_g_pos)
      self.log(f"Charts (SPD)/{log}_spd_g_neg", spd_g_neg)

    # generator

    assert mel.shape[-1] == 80
    feat_norm_loss = F.l1_loss(tgt_mel_hat.mean(dim=-1), mel.mean(dim=-1))

    total_model = vc_reconst * 10
    total_model += src2src_reconst * 10
    # total_model += feat_norm_loss
    # total_model += mi_ksp
    total_model += spd_g_pos - spd_g_neg

    if train:
      step_optimizer(self, opt_model, total_model, self.grad_clip)

    if log:
      self.log(f"Charts (Main)/{log}_loss", total_model)
      self.log(f"Charts (Main)/{log}_feat_norm", feat_norm_loss)

    return mel_hat, tgt_mel_hat, y_g_hat, y_g_hat_mel, vc_attn, []

  def vocoder_forward(self, mel: Tensor):
    return self.vocoder(mel.transpose(1, 2)).squeeze(1)

  def training_step(self, batch: Entry09, batch_idx: int):
    step = self.batches_that_stepped()

    opt_model, opt_spd = self.optimizers()
    sch_model, sch_spd = self.lr_schedulers()
    self.log("Charts (General)/lr", opt_model.optimizer.param_groups[0]["lr"])
    self.log("Charts (General)/lr_spd", opt_spd.optimizer.param_groups[0]["lr"])

    self._process_batch(batch, step, e2e=False, e2e_frames=self.e2e_frames, train=True, log="train")

    with hide_warns():
      sch_model.step()
      sch_spd.step()

  def validation_step(self, batch: Entry09, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch.src.mel
    y = batch.src.audio

    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics
    _, _, y_v, _, _, _ = self._process_batch(batch, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

    if complex_metrics:
      spksim = log_spksim0(self, P, y, y_v)
      self.log("valid_spksim", spksim["valid_spksim"].mean())
      self.val_outputs.append(spksim)

    if batch_idx == 0:
      mel_v, tgt_mel_v, y_v, ymel_v, attn_v, _ = self._process_batch(batch, step=step, e2e=True, e2e_frames=None)
      tgt_y_v = self.vocoder_forward(tgt_mel_v)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_v, "Attention")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_spectrograms(self, names, mel, tgt_mel_v, None, "Spectrogram (Conv)")
      log_audios2(self, P, names, 22050, y, tgt_y_v, y_v, cols=["index", "original", "converted", "cycled"])

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
    opt_spd = AdamW(self.speaker_d.parameters(), lr=self.lr_spd)

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
    sch_model = get_cosine_schedule_with_warmup(opt_model, self.warmup_steps, self.total_steps)
    sch_spd = S.MultiplicativeLR(opt_spd, lambda step: 1.0)

    return [opt_model, opt_spd], [sch_model, sch_spd]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem.split("_")[0].split(" ")[0]
  assert PROJECT.startswith("attempt09")
  PROJECT = "attempt09"

  setup_train_environment()

  P = Preparation("cuda")

  # データ読み込みがネックになってるせいか、 num_workers をある程度減らしたほうが速かった
  datamodule = DataModule09(P, frames=256, frames_ref=64, n_refs=32, ref_max_kth=64, batch_size=8, n_batches=2000, n_batches_val=200, same_density=True)

  total_steps = 40000
  total_actual_steps = 20000
  complex_metrics_batches = 50  # see: validation_step

  g_ckpt = DATA_DIR / "vocoder" / "g_02500000"
  do_ckpt = DATA_DIR / "vocoder" / "do_02500000"

  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_spd=1e-4,
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
      #     schedule=torch.profiler.schedule(wait=0, warmup=20, active=10, repeat=1),
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
