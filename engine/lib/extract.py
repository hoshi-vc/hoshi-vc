# TODO: MelSpectrogram のパラメーターを調節する : hop_length は wav2vec2 にあわせて、 n_mels は hifi-gan に合わせた。

import warnings
from logging import warning
from os import PathLike

import torch
import torch.nn.functional as F
from torch import Tensor
from torchaudio.functional import resample
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.utils import hide_warns

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

def extract_melspec(audio: Tensor, sr: int) -> Tensor:
  if sr != 22050:
    audio = resample(audio, sr, 22050)

  audio = audio.float()
  audio = audio.unsqueeze(0)

  # 441 = 320 / 16000 * 22050 :: wav2vec2 の 16000 Hz, hop_size 320 に合わせる
  mel = mel_spectrogram(audio, sampling_rate=22050, n_fft=1024, num_mels=80, hop_size=441, win_size=1024, fmin=0, fmax=8000)

  mel = mel[0].transpose(0, 1)
  return mel.float()  # shape: (seq_len, 80)
