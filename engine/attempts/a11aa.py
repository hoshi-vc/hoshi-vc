# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# 試行: ref sampling に使うキーを適宜更新してみる
# 経緯:
#   今までは hubert-soft の近いフレームほど参照したくなるだろうと予想して ref sampling を行っていた。
#   しかし attention map を見たところ、モデルはバラバラなフレームを参照しているようで、連続したフレームを活用する様子は部分的にしか見られなかった。
#   hubert-soft と key に要求される情報の性質が異なることが原因ではないかと予想した。（例えば key はピッチを軽視して良い）
#   このミスマッチのため、音声の切り貼りより、単なる再合成に近い処理をモデルが学習しているのではないかと考えた。
#   この試行の結果として、より変換に適した key を作成可能となり、モデルがより長いフレームを参照するようになることを期待している。
# 評価方法:
#   より長く連続したフラグメントを参照するようになったら予想通りです！

# %%

from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Optional, Type

import lightning.pytorch as L
import torch
import torch._dynamo
import torch.functional as F
import torch.nn.functional as F
import torch.optim.lr_scheduler as S
import wandb
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm

from engine.attempts.a10_dataset import DataModule10, Entry10, FeatsList
from engine.attempts.a11a_dataset import DataModule11A, prepare_key_index
from engine.attempts.utils import (BaseLightningModule, LinearSchedule, club_ksp_net, default_hifigan, log_attentions, log_audios, log_lr, log_spectrograms,
                                   log_spksim1, new_checkpoint_callback_wandb, new_wandb_logger, setup_train_environment, step_optimizer)
from engine.fragment_vc.utils import (cosine_schedule_with_warmup_lambda, get_cosine_schedule_with_warmup)
from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.attention import MultiHeadAttention
from engine.lib.club import CLUBSampleForCategorical, CLUBSampleForCategorical3
from engine.lib.layers import Buckets, Transpose
from engine.lib.utils import hide_warns
from engine.singleton import DATA_DIR, P

class VCModel(nn.Module):
  def __init__(self, hdim: int):
    super().__init__()

    # TODO: dropout, etc.

    energy_dim = hdim // 4
    pitch_dim = hdim // 4
    w2v2_dim = hdim // 2
    mel_dim = hdim // 2

    self.kdim = kdim = 256
    self.vdim = vdim = hdim

    self.mel_encode = nn.Linear(80, mel_dim)

    self.energy_bins = Buckets(-11.0, -3.0, 128)
    self.energy_embed = nn.Embedding(128, energy_dim)
    self.pitch_embed = nn.Embedding(360, pitch_dim)
    self.w2v2_embed = nn.Linear(256, w2v2_dim)

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

  def forward_lookup(self, key: Tensor, ref_key: Tensor, ref_value: Tensor, need_weights: bool = False):
    return self.lookup(key, ref_key, ref_value, need_weights=need_weights)

  def forward_decode(self, value: Tensor, energy: Tensor, pitch: Tensor):
    return self.decode(torch.cat([value, energy, pitch], dim=-1))

  def forward(self, batch: Entry10, src_ref_start: int, src_ref_len: int):
    n_refs = len(batch.ref)
    n_batch = len(batch.src.energy)
    ref_len = batch.ref[0].energy.shape[1]

    src_ref_end = src_ref_start + src_ref_len

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
    src_value = self.forward_value(src_energy, src_pitch, src_mel)

    # (...) -> (n_refs, n_batch, seq_len, feat_dim)
    ref_energy = ref_energy.unflatten(0, (n_refs, n_batch))
    ref_pitch = ref_pitch.unflatten(0, (n_refs, n_batch))
    ref_key = ref_key.unflatten(0, (n_refs, n_batch))
    ref_value = ref_value.unflatten(0, (n_refs, n_batch))
    ref_pitch_i = ref_pitch_i.unflatten(0, (n_refs, n_batch))

    # (...) -> (n_batch, n_refs*seq_len, feat_dim)
    pitck_cat = lambda src, ref: torch.cat([src[:, src_ref_start:src_ref_end], ref.transpose(0, 1).flatten(1, 2)[:, src_ref_len:]], dim=1)
    ref_key = pitck_cat(src_key, ref_key)
    ref_value = pitck_cat(src_value, ref_value)
    ref_pitch_i = pitck_cat(src_pitch_i, ref_pitch_i)

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
               warmup_steps: int,
               total_steps: int,
               self_ratio: list[tuple[int, float]],
               grad_clip: float,
               e2e_frames: int,
               hifi_gan: Any,
               hifi_gan_ckpt=None):
    super().__init__()
    self.vc_model = VCModel(hdim=hdim)

    n_speakers = len(P.dataset.speaker_ids)
    kdim = self.vc_model.kdim
    vdim = self.vc_model.vdim

    ksp_hdim = hdim
    self.club_val = CLUBSampleForCategorical(vdim, 360, hdim=hdim)
    self.club_key = CLUBSampleForCategorical(kdim, 360, hdim=hdim)
    self.club_ksp = CLUBSampleForCategorical3(kdim, n_speakers, hdim=hdim, logvar=club_ksp_net(kdim, n_speakers, ksp_hdim))

    self.batch_rand = Random(94324203)
    self.clip_rand = Random(76482573)
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.self_ratio = LinearSchedule(self_ratio)
    self.lr = lr
    self.lr_club = lr_club
    self.grad_clip = grad_clip
    self.e2e_frames = e2e_frames
    self.hifi_gan = hifi_gan
    self.hifi_gan_ckpt = hifi_gan_ckpt

    self.val_outputs = []

    self.save_hyperparameters()

    # activates manual optimization.
    # https://lightning.ai/docs/pytorch/stable/common/optimization.html
    self.automatic_optimization = False

  def _process_batch(
      self,
      batch: Entry10,
      self_ratio: float,
      step: int,
      e2e: bool,
      e2e_frames: Optional[int],
      train=False,
      log: Optional[str] = None,
  ):
    h = self.hifi_gan
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
    mel_hat, (ref_key, ref_value, ref_pitch_i, vc_attn), _ = self.vc_model(batch, src_ref_start, src_ref_len)

    vc_reconst = F.l1_loss(mel_hat, mel)

    if log:
      self.log(f"Charts (Main)/{log}_reconst", vc_reconst)

    # CLUB

    club_x_val = ref_value
    club_x_key = ref_key
    club_y = ref_pitch_i[:, :, 0]
    club_sp_y = batch.src.speaker.repeat(1, ref_key.shape[1])
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

    e2e_y_hat = None
    e2e_y_hat_mel = None
    if debug or e2e:

      # vocoder

      if e2e_frames is None:
        e2e_start = 0
        e2e_end = mel.shape[1]
      else:
        e2e_start = self.clip_rand.randint(0, mel.shape[1] - e2e_frames)
        e2e_end = e2e_start + e2e_frames

      e2e_mel_hat = mel_hat[:, e2e_start:e2e_end].transpose(1, 2)

      e2e_y_hat = P.vocoder._model(e2e_mel_hat)
      e2e_y_hat_mel = mel_spectrogram(
          e2e_y_hat.squeeze(1), sampling_rate=22050, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=8000, fast=True)

      e2e_y_hat = e2e_y_hat.squeeze(1)
      e2e_y_hat_mel = e2e_y_hat_mel.transpose(1, 2)

    # generator

    total_model = vc_reconst * 150.0
    total_model += mi_val
    total_model += mi_ksp

    if train:
      step_optimizer(self, opt_model, total_model, self.grad_clip)

    if log:
      self.log(f"Charts (Main)/{log}_loss", total_model)

    return mel_hat, e2e_y_hat, e2e_y_hat_mel, vc_attn, []

  def vocoder_forward(self, mel: Tensor):
    return P.vocoder._model(mel.transpose(1, 2)).squeeze(1)

  def training_step(self, batch: Entry10, batch_idx: int):
    step = self.batches_that_stepped()
    self.log("Charts (General)/step", step)

    self_ratio = self.self_ratio(step)
    self.log("Charts (General)/self_ratio", self_ratio)

    opt_model, opt_club = self.optimizers()
    sch_model, sch_club = self.lr_schedulers()
    log_lr(self, opt_model, "lr")
    log_lr(self, opt_club, "lr_club")

    self._process_batch(batch, self_ratio, step, e2e=False, e2e_frames=self.e2e_frames, train=True, log="train")

    with hide_warns():
      sch_club.step()
      sch_model.step()

  def validation_step(self, batch: Entry10, batch_idx: int):
    step = self.batches_that_stepped()
    mel = batch.src.mel
    y = batch.src.audio

    complex_metrics = batch_idx < complex_metrics_batches
    e2e = batch_idx == 0 or complex_metrics
    _, y_c, _, _, _ = self._process_batch(batch, self_ratio=1.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="cheat")
    _, y_v, _, _, _ = self._process_batch(batch, self_ratio=0.0, step=step, e2e=e2e, e2e_frames=self.e2e_frames, log="valid")

    if complex_metrics:
      spksim = log_spksim1(self, y, y_v, y_c)
      self.log("valid_spksim", spksim["valid_spksim"].mean())
      self.log("cheat_spksim", spksim["cheat_spksim"].mean())
      self.val_outputs.append(spksim)

    if batch_idx == 0:
      mel_c, y_c, ymel_c, attn_c, _ = self._process_batch(batch, self_ratio=1.0, step=step, e2e=True, e2e_frames=None)
      mel_v, y_v, ymel_v, attn_v, _ = self._process_batch(batch, self_ratio=0.0, step=step, e2e=True, e2e_frames=None)
      names = [f"{i:02d}" for i in range(len(mel))]
      log_attentions(self, names, attn_c, "Attention (Cheat)")
      log_attentions(self, names, attn_v, "Attention")
      log_spectrograms(self, names, mel, mel_c, ymel_c, "Spectrogram (Cheat)")
      log_spectrograms(self, names, mel, mel_v, ymel_v, "Spectrogram")
      log_audios(self, names, 22050, [y, y_v, y_c], ["original", "valid", "cheat"])

  def on_validation_epoch_end(self):
    if len(self.val_outputs) > 0:
      v_spksim = torch.cat([x["valid_spksim"] for x in self.val_outputs]).cpu().numpy()
      c_spksim = torch.cat([x["cheat_spksim"] for x in self.val_outputs]).cpu().numpy()
      self.log_wandb({"Charts (SpkSim)/valid_hist": wandb.Histogram(v_spksim)})
      self.log_wandb({"Charts (SpkSim)/cheat_hist": wandb.Histogram(c_spksim)})

    self.val_outputs.clear()

  def on_validation_end(self):
    P.release_spkemb()
    P.release_mosnet()
    torch.cuda.empty_cache()

  def configure_optimizers(self):
    print("configure_optimizers")

    opt_model = AdamW(self.vc_model.parameters(), lr=self.lr)
    opt_club = AdamW([*self.club_val.parameters(), *self.club_key.parameters(), *self.club_ksp.parameters()], lr=self.lr_club)

    sch_model_lambda = cosine_schedule_with_warmup_lambda(self.warmup_steps, self.total_steps)
    sch_model = S.LambdaLR(opt_model, lambda _: sch_model_lambda(self.batches_that_stepped()))
    sch_club = S.LambdaLR(opt_club, lambda _: 1.0)

    return [opt_model, opt_club], [sch_model, sch_club]

if __name__ == "__main__":

  PROJECT = Path(__file__).stem.split("_")[0].split(" ")[0]
  assert PROJECT.startswith("a11")
  PROJECT = "a11"

  setup_train_environment()

  P.set_device("cuda")

  def load_datamodule(cls: Type[DataModule10]):
    if datamodule is not None: datamodule.accessor.clear_cache()
    data_req = FeatsList(audio=True, speaker=True, energy=True, mel=True, pitch_i=True, soft=True)
    return cls(frames=256, frames_ref=32, n_refs=32, ref_max_kth=64, batch_size=8, n_batches=1000, n_batches_val=200, same_density=True, req=data_req)

  prepare_key_index(VCModule.load_from_checkpoint(DATA_DIR / "a11/checkpoints/cool-feather-22/qyul94dq/last.ckpt", map_location=P.device))
  datamodule = None
  datamodule = load_datamodule(DataModule11A)

  total_steps = 20000  # LR の調節に使われるだけなので iter_steps * iter_count と異なってもいい
  iter_steps = 10000
  iter_count = 2
  complex_metrics_batches = 50  # see: validation_step

  model = VCModule(
      hdim=512,
      lr=1e-4,
      lr_club=1e-4,
      warmup_steps=500,
      total_steps=total_steps,
      self_ratio=[(0, 0.0)],
      e2e_frames=64,
      grad_clip=1.0,
      hifi_gan=default_hifigan().config,
      hifi_gan_ckpt=default_hifigan().ckpts,
  )

  wandb_logger = new_wandb_logger(PROJECT)

  trainer = L.Trainer(
      max_epochs=int(ceil(iter_steps / datamodule.n_batches)),
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
      # detect_anomaly=True,
      # deterministic=True,
      # profiler=L.profilers.PyTorchProfiler(
      #     DATA_DIR / "profiler",
      #     schedule=torch.profiler.schedule(wait=0, warmup=30, active=6, repeat=1),
      #     on_trace_ready=torch.profiler.tensorboard_trace_handler(DATA_DIR / "profiler"),
      #     with_stack=False,
      # ),
  )

  for i in tqdm(range(iter_count), desc="iteration", ncols=0):
    if i >= 1:
      prepare_key_index(model)
      datamodule = load_datamodule(DataModule11A)

    if i != 0: trainer.num_sanity_val_steps = 0
    trainer.fit_loop.max_epochs = int(ceil(iter_steps * (i + 1) / datamodule.n_batches))

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=None,
    )

  wandb.finish()

  print(f"Saved model to {Path(trainer.checkpoint_callback.last_model_path).relative_to(DATA_DIR)}")
