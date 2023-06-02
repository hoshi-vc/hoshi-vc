# TODO: 現状、ピッチに依らない表現を獲得できていない。あとでちゃんとやる。
#       confusion network だけで pitch 情報を取り除くのが難しい気がする。
#       同一話者、同一発音の別の音声から同じ表現が得られるようなエンコーダーを学習させるとか試す？

# %% [markdown]
# This notebook (\*.ipynb) was generated from the corresponding python file (\*.py).
#
# ## Attempt 02: 音程によらない表現をつかう
#
# 先の Attempt 01 では、音程があっていない音声を無理やりつなげているせいでうまくいかないように見えた。
#
# なので今度は、音程を調節してから切り貼りするようにしたら良くなるかためしてみる。

# %%

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.dataset_feats import FeatureDataset, FeatureEntry
from engine.lib.grad_rev import GradientReversal
from engine.lib.utils import DATA_DIR, NPArray, clamp, save_ckpt
from engine.lib.utils_ui import play_audio, plot_spectrogram
from engine.preparation import (CREPE_MODEL, FEATS_DIR, PITCH_TOPK, Preparation, pad_clip)

device = "cuda"

P = Preparation(device)

P.prepare_feats()
P.prepare_faiss()

#
#    mel ---+---+---> embedding ---+---> mel_hat (reconstruction loss)
#           |   |                  |
#  pitch ---+   |         pitch ---+
#               |
#               +----[grad_rev]--------> pitch_hat (adversarial loss)
#

train_dataset = [FEATS_DIR / "parallel100" / sid for sid in P.dataset.speaker_ids]
valid_dataset = [FEATS_DIR / "nonpara30" / sid for sid in P.dataset.speaker_ids]
train_dataset = FeatureDataset(train_dataset, frames=256, start_hop=128)
valid_dataset = FeatureDataset(valid_dataset, frames=256, start_hop=128)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False)

class Model(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    hdim = 512
    self.embed_pitch = nn.Embedding(360, hdim // 2)
    self.embed_pitch2 = nn.Embedding(360, hdim // 2)
    self.embed_mel = nn.Linear(80, hdim // 2)
    self.encode = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
    )
    self.decode = nn.Sequential(
        nn.Linear(hdim + hdim // 2, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
    )
    self.pred_pitch_rev = GradientReversal(0.0)
    self.pred_pitch = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
    )
    self.last_mel = nn.Linear(hdim, 80)
    self.last_pitch = nn.Linear(hdim, 360)

  def forward(self, mel: Tensor, pitch_i: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mel = self.embed_mel(mel)  #           shape: (batch, seq_len, hdim // 2)
    pitch = self.embed_pitch(pitch_i)  #   shape: (batch, seq_len, hdim // 2)
    pitch2 = self.embed_pitch2(pitch_i)  # shape: (batch, seq_len, hdim // 2)

    x = torch.cat([mel, pitch], dim=-1)
    x = self.encode(x)
    embed = x
    x = torch.cat([x, pitch2], dim=-1)
    x = self.decode(x)
    x = self.last_mel(x)
    mel_hat = x

    p = embed
    p = self.pred_pitch_rev(p)
    p = self.pred_pitch(p)
    p = self.last_pitch(p)
    pitch_dist_hat = p

    # mel_hat:        (batch, seq_len, 80)
    # pitch_dist_hat: (batch, seq_len, 360)
    # embed:          (batch, seq_len, hdim)
    return mel_hat, pitch_dist_hat, embed

def loss_fn(mel: Tensor, pitch_i: Tensor, mel_hat: Tensor, pitch_dist_hat: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
  mel_loss = F.mse_loss(mel_hat, mel)
  pitch_loss = F.cross_entropy(pitch_dist_hat.transpose(1, 2), pitch_i)
  loss = mel_loss + pitch_loss
  return loss, (mel_loss, pitch_loss)

def train(model: nn.Module, optimizer: Optimizer):
  model.train()
  for batch_i, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), ncols=0, desc="train")):
    batch: FeatureEntry

    mel, pitch_i_topk, pitch_v_topk = batch.mel.to(device), batch.pitch_i.to(device, torch.int64), batch.pitch_v.to(device)
    pitch_i = pitch_i_topk[:, :, 0]

    mel_hat, pitch_dist_hat, _ = model(mel, pitch_i)
    loss, (mel_loss, pitch_loss) = loss_fn(mel, pitch_i, mel_hat, pitch_dist_hat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if torch.isnan(loss):
      raise RuntimeError("Loss is NaN.")

    if batch_i % 10 == 0:
      pbar.set_postfix_str(f"mel_loss {mel_loss.item():.3f}, pitch_loss {pitch_loss.item():.3f}")

def validate(model: nn.Module):
  model.eval()

  mel_loss_list: list[float] = []
  pitch_loss_list: list[float] = []

  with torch.no_grad():
    for batch_i, batch in (pbar := tqdm(enumerate(valid_loader), total=len(valid_loader), ncols=0, desc="valid")):
      batch: FeatureEntry

      mel, pitch_i_topk, pitch_v_topk = batch.mel.to(device), batch.pitch_i.to(device, torch.int64), batch.pitch_v.to(device)
      pitch_i = pitch_i_topk[:, :, 0]

      mel_hat, pitch_dist_hat, _ = model(mel, pitch_i)
      loss, (mel_loss, pitch_loss) = loss_fn(mel, pitch_i, mel_hat, pitch_dist_hat)

      mel_loss_list.append(mel_loss.item())
      pitch_loss_list.append(pitch_loss.item())

      if torch.isnan(loss):
        raise RuntimeError("Loss is NaN.")

      if batch_i % 10 == 0:
        pbar.set_postfix_str(f"mel_loss {mel_loss.item():.3f}, pitch_loss {pitch_loss.item():.3f}")

  print(f"mel_loss:   {np.mean(mel_loss_list):.3f} (std {np.std(mel_loss_list):.3f})")
  print(f"pitch_loss: {np.mean(pitch_loss_list):.3f} (std {np.std(pitch_loss_list):.3f})")

  return np.mean(mel_loss_list), np.mean(pitch_loss_list)

def adjust_rev_scale(model: nn.Module, mel_loss: float, pitch_loss: float):
  # 何もしなかったら mel_loss が小さくなってくれなかったので、 rev_scale を調節してみることにした。
  # はじめに melspec の恒等変換を覚えてから、少しずつピッチ非依存の表現に寄せていくイメージ。

  # ひとまず、雑に rev_scale を調節してみる。
  # TODO: ある程度まで大きくしたら、 rev_scale を増やすのを速くしてもいいかもしれない。

  rev_scale = model.pred_pitch_rev.scale.item()
  if mel_loss < 0.1:
    if pitch_loss > 3.5:
      rev_scale -= 0.001
    else:
      rev_scale += 0.001  #  + floor(rev_scale * 10) * 0.01
  elif mel_loss > 0.2:
    rev_scale -= 0.001
  model.pred_pitch_rev.update_scale(clamp(rev_scale, 0.0, 10.0))
  print(f"pred_pitch_rev.scale: {model.pred_pitch_rev.scale:.3f}")

model = Model().to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)

for epoch in range(100):
  print(f"Epoch {epoch}")
  train(model, optimizer)
  mel_loss, pitch_loss = validate(model)
  adjust_rev_scale(model, mel_loss, pitch_loss)
  save_ckpt(DATA_DIR / "attempt02" / "model.pt", model=model, optimizer=optimizer)

# %%
# モデルが再構成できるかを確認してみる。

item = P.dataset[1000]
print(item.name)

audio, sr = item.audio[0], item.sr
mel = P.extract_melspec(audio, sr)
pitch_i_topk, _ = P.extract_pitch_topk(audio, sr, CREPE_MODEL, PITCH_TOPK, device)
pitch_i_topk = torch.as_tensor(pad_clip(mel.cpu().numpy(), pitch_i_topk.cpu().numpy()))
pitch_i = pitch_i_topk[:, 0]
pitch_i = (pitch_i - 20).maximum(torch.as_tensor(1))
pitch_i = pitch_i * 0 + 100

with torch.no_grad():
  mel_hat, _, _ = model(mel.unsqueeze(0).to(device), pitch_i.unsqueeze(0).to(device, torch.int64))

audio_hat, sr_hat = P.vocoder(mel_hat[0])

plot_spectrogram(mel)
plot_spectrogram(mel_hat[0])
play_audio(audio, sr)
play_audio(audio_hat, sr_hat)

# %%
# TODO: 先に作ったモデルを使って、音声の切り貼りをうまくしてみる。

index = P.get_index("jvs001")
target_mel = P.get_mel("jvs001")
target_pitch = P.get_pitch("jvs001")

keys = P.extract_wav2vec2(audio, sr).cpu().numpy()

top_k = 128

hat = []
for i in range(len(keys)):
  D, I = index.search(keys[None, i], top_k)
  items: list[NPArray] = []
  for j in range(top_k):
    items.append(target_mel[I[0][j]])
  hat.append(np.mean(np.stack(items), axis=0))
hat = torch.as_tensor(np.vstack(hat))

audio_hat, sr_hat = P.vocoder(hat)
plot_spectrogram(mel)
plot_spectrogram(hat)
play_audio(audio, sr)
play_audio(audio_hat, sr_hat)

# %%
