# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

# hop_size を揃えていないと扱いにくいので、 hifi-gan に合わせることにした（ 22050 Hz, 256 frame ）
# wav2vec 2.0 の hop_size は異なるが、前処理の段階で interpolate して長さを揃えることにした
# NOTE: phoneme の場合に、同じ音素が二連続に入ることがあるけど、モデルの方でうまくやってくれると願う

from os import PathLike
from typing import Optional

import torch
import torch.nn.functional as F
import torchcrepe
from torch import Tensor
from torchaudio.functional import resample
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor

from engine.hifi_gan.meldataset import mel_spectrogram
from engine.lib.utils import hide_warns

def feat_len(audio: Tensor, sr: int):
  if sr != 22050: print("performance warning: sr != 22050")
  audio = resample(audio, sr, 22050)
  return audio.shape[0] // 256

def to22050(audio: Tensor, sr: int):
  return resample(audio, sr, 22050), 22050

def assert_torch(x: Tensor, dtype: torch.dtype, *shape: tuple[int, ...]):
  assert x.dtype == dtype, f"expected {dtype}, but got {x.dtype}"
  assert x.shape == shape, f"expected {shape}, but got {x.shape}"

def interp_nearest(x: Tensor, size: int):  # (len, dim) -> (size, dim)
  x = x.unsqueeze(0).transpose(1, 2)
  x = F.interpolate(x, size=size, mode="nearest-exact")
  x = x.squeeze(0).transpose(0, 1)
  return x

def pad_clip(x: Tensor, size: int, max_diff: Optional[int] = None):
  if max_diff is not None:
    assert abs(len(x) - size) <= max_diff

  if len(x) < size:
    pad_size = size - len(x)
    x = x.transpose(0, 1)
    x = F.pad(x, (0, pad_size), 'replicate')
    x = x.transpose(0, 1)
  elif len(x) > size:
    x = x[:size]
  return x

class Audio:  # (feat_len * 256,)
  def __call__(self, audio: Tensor, sr: int):
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    out = audio[:flen * 256].float()

    assert_torch(out, torch.float32, flen * 256)
    return out

class MelSpec:  # (feat_len, 80)
  def __call__(self, audio: Tensor, sr: int):
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    mel = mel_spectrogram(audio.unsqueeze(0), sampling_rate=sr, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=8000)
    mel = mel[0].transpose(0, 1).float()

    assert_torch(mel, torch.float32, flen, 80)
    return mel

class Energy:  # (feat_len, 1)
  def __call__(self, audio: Tensor, sr: int):
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    mel = MelSpec()(audio, sr)
    energy = torch.mean(mel, dim=1).unsqueeze(-1).float()

    assert_torch(energy, torch.float32, flen, 1)
    return energy

class Pitch:  # indices, values : (feat_len, topk), (feat_len, topk)
  def __init__(self, model: str, topk: int):
    self.model = model
    self.topk = topk

  def __call__(self, audio: Tensor, sr: int):
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    with torch.no_grad():
      audio16k = resample(audio, sr, 16000)
      batch = next(self._torchcrepe_preprocess(audio16k.unsqueeze(0), 16000, hop_length=256 * 16000 / 22050, device=audio.device))
      matrix = torchcrepe.infer(batch, model=self.model)
      pitch = matrix.topk(self.topk)
      indices = pitch.indices
      values = pitch.values.float()

    # "replication_pad1d_cuda" not implemented for 'Short', 'Long' なので。
    indices = pad_clip(indices.float(), flen, max_diff=2).to(torch.int16)
    values = pad_clip(values, flen, max_diff=2)

    assert_torch(indices, torch.int16, flen, self.topk)
    assert_torch(values, torch.float32, flen, self.topk)
    return indices, values

  def _torchcrepe_preprocess(self, audio, sample_rate, hop_length, batch_size=None, device='cpu', pad=True):
    """ Modified version of torchcrepe.core.preprocess """
    # hop_length に小数を入れられるようにした

    from torchcrepe.core import SAMPLE_RATE, WINDOW_SIZE

    # Resample
    assert sample_rate == SAMPLE_RATE

    # Get total number of frames

    # Maybe pad
    if pad:
      total_frames = 1 + int(audio.size(1) // hop_length)
      audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
      total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

      # Batch indices
      start = max(0, int(round(i * hop_length)))
      end = min(audio.size(1), int(round((i + batch_size - 1) * hop_length)) + WINDOW_SIZE)

      # Chunk
      frames = torch.nn.functional.unfold(audio[:, None, None, start:end], kernel_size=(1, WINDOW_SIZE), stride=(1, int(round(hop_length))))

      # shape=(1 + int(time / hop_length, 1024)
      frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)

      # Place on device
      frames = frames.to(device)

      # Mean-center
      frames -= frames.mean(dim=1, keepdim=True)

      # Scale
      # Note: during silent frames, this produces very large values. But
      # this seems to be what the network expects.
      frames /= torch.max(torch.tensor(1e-10, device=frames.device), frames.std(dim=1, keepdim=True))

      yield frames

class Wav2Vec2:  # (feat_len, 768)
  def __init__(self, preprocessor: Wav2Vec2Processor, model: Wav2Vec2Model):
    self.preprocessor = preprocessor
    self.model = model

  def __call__(self, audio: Tensor, sr: int) -> Tensor:
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    with torch.no_grad():
      # F.pad により len(w2v2) == len(audio16k) // 320 にする
      inputs = resample(audio, sr, 16000)
      inputs = F.pad(inputs, (40, 40))
      inputs = self.preprocessor(inputs, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
      outputs = self.model(inputs)
      w2v2 = outputs.last_hidden_state[0].float()

    # 長さを調節する前に、予想通りの縮小率になっていることを確認する
    assert abs(len(w2v2) * 320 / 16000 - flen * 256 / 22050) <= 0.02

    w2v2 = interp_nearest(w2v2, flen)

    assert_torch(w2v2, torch.float32, flen, 768)
    return w2v2

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

class Phoneme:  # indices, logits : (feat_len, topk), (feat_len, topk)
  def __init__(self, preprocessor: Wav2Vec2Processor, model: Wav2Vec2ForCTC, topk: int):
    self.preprocessor = preprocessor
    self.model = model
    self.topk = topk

  def __call__(self, audio: Tensor, sr: int) -> Tensor:
    audio, sr = to22050(audio, sr)
    flen = feat_len(audio, sr)

    with torch.no_grad():
      # F.pad により len(w2v2) == len(audio16k) // 320 にする
      inputs = resample(audio, sr, 16000)
      inputs = F.pad(inputs, (40, 40))
      inputs = self.preprocessor(inputs, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
      outputs = self.model(inputs)

      log_probs = F.log_softmax(outputs.logits, dim=-1)
      k_logits, k_indice = torch.topk(log_probs, k=self.topk, dim=-1)
      k_indice = k_indice[0]
      k_logits = k_logits[0].float()

    # 長さを調節する前に、予想通りの縮小率になっていることを確認する
    assert abs(len(k_indice) * 320 / 16000 - flen * 256 / 22050) <= 0.02
    assert abs(len(k_logits) * 320 / 16000 - flen * 256 / 22050) <= 0.02

    # "upsample_nearest1d_out_frame" not implemented for 'Short', 'Long' なので。
    k_indice = interp_nearest(k_indice.float(), flen).to(torch.int16)
    k_logits = interp_nearest(k_logits, flen)

    assert_torch(k_indice, torch.int16, flen, self.topk)
    assert_torch(k_logits, torch.float32, flen, self.topk)
    return k_indice, k_logits

  def to(self, device: torch.device):
    self.preprocessor = self.preprocessor
    self.model = self.model.to(device)
    return self

  @property
  def device(self) -> torch.device:
    return self.model.device

  @staticmethod
  def load(pretrained_model_name_or_path: str | PathLike, topk: int):
    with hide_warns():
      preprocessor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path)
      model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path)
    model.eval()
    return Phoneme(preprocessor, model, topk)

if __name__ == "__main__":
  from tqdm import tqdm

  from engine.prepare import Preparation

  P = Preparation("cpu")
  item = P.dataset[0]
  audio = item.audio[0].repeat(3)
  sr = 22050

  extract_mel = MelSpec()
  # extract_w2v2 = Wav2Vec2.load("facebook/wav2vec2-base")
  extract_pitch = Pitch("tiny", 1)

  for i in range(50):
    j = 512 - 10 + i
    aslice = audio[:j]
    # print(j, extract_w2v2(aslice, sr).shape)
    print(j, extract_pitch(aslice, sr)[0].shape)
