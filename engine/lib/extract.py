# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
from os import PathLike

import numpy as np
import torch
import torch.nn.functional as F
import torchcrepe
from torch import Tensor
from torchaudio.functional import resample
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor

from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.utils import Device, hide_warns

class Wav2Vec2:
  def __init__(self, preprocessor: Wav2Vec2Processor, model: Wav2Vec2Model):
    self.preprocessor = preprocessor
    self.model = model

  def __call__(self, audio: Tensor, sr: int) -> Tensor:
    if sr != 16000:
      audio = resample(audio, sr, 16000)

    # hifi-gan の出力と長さを合わせるための padding :: 試行錯誤してこの設定ならうまく行きそうだった
    audio = F.pad(audio, (39, 40))

    with torch.no_grad():
      audio = self.preprocessor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
      outputs = self.model(audio)
      embeddings = outputs.last_hidden_state[0]

    return embeddings.float()  # shape: (seq_len, 768)

  def to(self, device: torch.device):
    self.preprocessor = self.preprocessor
    self.model = self.model.to(device)
    return self

  @property
  def device(self) -> torch.device:
    return self.model.device

  @staticmethod
  def load(pretrained_model_name_or_path: str | PathLike):
    with hide_warns():
      preprocessor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path)
      model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path)
    model.eval()
    return Wav2Vec2(preprocessor, model)

class Phoneme:
  def __init__(self, preprocessor: Wav2Vec2Processor, model: Wav2Vec2ForCTC):
    self.preprocessor = preprocessor
    self.model = model

  def __call__(self, audio: Tensor, sr: int, top_k: int) -> Tensor:
    if sr != 16000:
      audio = resample(audio, sr, 16000)

    # hifi-gan の出力と長さを合わせるための padding :: 試行錯誤してこの設定ならうまく行きそうだった
    audio = F.pad(audio, (39, 40))

    with torch.no_grad():
      audio = self.preprocessor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
      outputs = self.model(audio)

      log_probs = F.log_softmax(outputs.logits, dim=-1)
      k_log_probs, k_ids = torch.topk(log_probs, k=top_k, dim=-1)
      k_ids = k_ids[0]
      k_log_probs = k_log_probs[0]

    return k_ids.to(torch.int16), k_log_probs.float()  # shape: (seq_len, topk)

  def to(self, device: torch.device):
    self.preprocessor = self.preprocessor
    self.model = self.model.to(device)
    return self

  @property
  def device(self) -> torch.device:
    return self.model.device

  @staticmethod
  def load(pretrained_model_name_or_path: str | PathLike):
    with hide_warns():
      preprocessor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path)
      model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path)
    model.eval()
    return Phoneme(preprocessor, model)

def extract_melspec(audio: Tensor, sr: int) -> Tensor:
  if sr != 22050:
    audio = resample(audio, sr, 22050)

  audio = audio.float()
  audio = audio.unsqueeze(0)

  # 441 = 320 / 16000 * 22050 :: wav2vec2 の 16000 Hz, hop_size 320 に合わせる
  mel = mel_spectrogram(audio, sampling_rate=22050, n_fft=1024, num_mels=80, hop_size=441, win_size=1024, fmin=0, fmax=8000)

  mel = mel[0].transpose(0, 1)
  return mel.float()  # shape: (seq_len, 80)

def extract_energy(audio: Tensor, sr: int) -> Tensor:
  mel = extract_melspec(audio, sr)
  energy = torch.mean(mel, dim=1)
  return energy.unsqueeze(-1).float()  # shape: (seq_len, 1)

def extract_pitch_matrix(audio: Tensor, sr: int, model: str, device: Device) -> Tensor:
  if sr != 16000:
    audio = resample(audio, sr, 16000)

  with torch.no_grad():
    audio = F.pad(audio, (351, 352))
    batch = next(torchcrepe.preprocess(audio.unsqueeze(0), sr, hop_length=320, device=device, pad=False))
    matrix = torchcrepe.infer(batch, model=model)

  return matrix.float()  # shape: (seq_len, 360)

def extract_pitch_topk(audio: Tensor, sr: int, model: str, topk: int, device: Device) -> tuple[Tensor, Tensor]:
  pitch = extract_pitch_matrix(audio, sr, model, device)
  pitch = pitch.topk(topk)
  return pitch.indices.to(torch.int16), pitch.values.float()  # shape: (seq_len, topk)

if __name__ == "__main__":
  from tqdm import tqdm

  from engine.prepare import Preparation

  device = "cuda"
  P = Preparation(device)

  item = P.dataset[0]
  audio0 = item.audio.repeat((1, 3))[0]

  # すべての features が同じ長さになっているか確認
  for j in tqdm(range(480 * 2, len(audio0), 480)):
    lens = []
    for i in range(j, j + 2):
      # TODO: あとから気がついたけど、この resample を外すとだめだった。
      #       よくわからないから preparetion.py で適当に出力を padding して対処することにした。
      audio, sr = audio0[:i], item.sr
      audio, sr = resample(audio, sr, 16000), 16000

      mel = P.extract_melspec(audio, sr)
      w2v2 = P.extract_wav2vec2(audio, sr)
      pi, pv = P.extract_pitch_topk(audio, sr, "tiny", 2, device)
      assert mel.shape[0] == w2v2.shape[0] == pi.shape[0] == pv.shape[0]
      lens.append(mel.shape[0])
    assert lens[0] + 1 == lens[1]
    assert lens[0] == j / 480 - 1
