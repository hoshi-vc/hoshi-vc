# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import librosa
import numpy as np
import torch
from torch import Tensor
from torchaudio.functional import resample

from engine.lib.utils import NPArray, clamp

# TODO: Pytorch で書き直したい

def erode(mask: NPArray, amount: int = 1):  # mask の中の True の部分を収縮する。
  for _ in range(amount):
    mask[1:] = mask[1:] & mask[:-1]
    mask[:-1] = mask[:-1] & mask[1:]
  return mask

def dilate(mask: NPArray, amount: int = 1):  # mask の中の True の部分を膨張する。
  for _ in range(amount):
    mask[1:] = mask[1:] | mask[:-1]
    mask[:-1] = mask[:-1] | mask[1:]
  return mask

def trim_silence(
    audio: Tensor,
    sr: int,
    mode="split-join",
    top_db=40.0,
    frame_length=2048,
    hop_length=512,
) -> Tensor:

  # とりあえず正規化しておく
  # resampy だと遅いので、torchaudio で resample する。
  normalized = librosa.util.normalize(resample(audio, sr, 44100).numpy())
  conv_sr = lambda frame: clamp(int(frame * sr / 44100), 0, len(audio))

  # librosa の trim のコードが使っていた処理でデシベル値を得る。
  mse = librosa.feature.rms(y=normalized, frame_length=frame_length, hop_length=hop_length)
  db = librosa.core.amplitude_to_db(mse[..., 0, :])

  mask = db > -top_db  # 音がある部分を True にする。

  # 短い無音区間を塞ぐ : 促音とかを残す -- jvs/jvs001/parallel100/VOICEACTRESS100_062 など
  mask = dilate(mask, 10)
  mask = erode(mask, 10)

  # 短い音声区間を除去 : クリック音とかを消す -- jvs/jvs009/parallel100/VOICEACTRESS100_086 など
  mask = erode(mask, 10)
  mask = dilate(mask, 10)

  # 話し始めや話し終わりの部分の余韻を含める
  mask = dilate(mask, 5)

  if mode == "trim":
    nonzero, = np.where(mask)

    if len(nonzero) > 0:
      # Compute the start and end positions
      # End position goes one frame past the last non-zero
      start = int(librosa.core.frames_to_samples(nonzero[0], hop_length=hop_length))
      end = int(librosa.core.frames_to_samples(nonzero[-1] + 1, hop_length=hop_length))
    else:
      # The signal only contains zeros
      start, end = 0, 0

    return audio[conv_sr(start):conv_sr(end)]

  elif mode == "split-join":
    edges, = np.where(np.diff(np.concatenate([[False], mask, [False]])))

    # edges には、 (有音区間の開始インデックス), (終了インデックス + 1), ... が入ってる。
    # print(mask * np.arange(len(mask)))
    # print(edges)

    if len(edges) == 0: return audio[0:0]

    out = []
    assert len(edges) % 2 == 0
    for i in range(0, len(edges), 2):
      start = edges[i]
      end = edges[i + 1]

      # end + 1 としないのは、すでに end は有音区間の終了インデックス + 1 になっているから。
      start = int(librosa.core.frames_to_samples(start, hop_length=hop_length))
      end = int(librosa.core.frames_to_samples(end, hop_length=hop_length))

      out.append(audio[conv_sr(start):conv_sr(end)])

    return torch.cat(out)

  else:
    raise ValueError(f"Unknown mode: {mode}")

# %%
if __name__ == "__main__":
  from torch import as_tensor as T

  from engine.lib.utils_ui import play_audio, plot_spectrogram
  from engine.singleton import P

  item = P.dataset[100]

  audio_trimmed = trim_silence(item.audio[0], item.sr)

  print(item.name, len(item.audio[0]), len(audio_trimmed))
  play_audio(item.audio, item.sr)
  play_audio(audio_trimmed, item.sr)
  plot_spectrogram(P.extract_melspec(item.audio[0], item.sr), factor=2.0)
  plot_spectrogram(P.extract_melspec(T(audio_trimmed), item.sr), factor=2.0)
