import librosa
import numpy as np
from resampy import resample

from engine.lib.utils import NPArray, clamp

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
    audio: NPArray,
    sr: int,
    mode="split-join",
    top_db=30.0,
    frame_length=2048,
    hop_length=512,
) -> NPArray:

  # とりあえず正規化しておく
  audio = resample(audio, sr, 16000)
  normalized = librosa.util.normalize(audio)
  conv_sr = lambda frame: clamp(int(frame * sr / 16000), 0, len(audio))

  # librosa の trim のコードが使っていた処理でデシベル値を得る。
  mse = librosa.feature.rms(y=normalized, frame_length=frame_length, hop_length=hop_length)
  db = librosa.core.amplitude_to_db(mse[..., 0, :])

  mask = db > -top_db  # 音がある部分を True にする。

  # # 話し始めや話し終わりの部分の無音区間を含める
  mask = dilate(mask, 10)

  # # 短い無音区間を塞ぐ
  mask = dilate(mask, 10)
  mask = erode(mask, 10)

  # # 短い音声区間を除去する
  mask = erode(mask, 10)
  mask = dilate(mask, 10)

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

    return np.concatenate(out)

  else:
    raise ValueError(f"Unknown mode: {mode}")
